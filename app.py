import os
import tempfile
import requests
import pandas as pd
import networkx as nx
import streamlit as st

from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.Pipeline import Pipeline, create_workflow
from matchms.filtering.default_pipelines import DEFAULT_FILTERS
from matchms.networking import SimilarityNetwork
from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore

# ---------------------------------------------------------
# 1. MODEL UTILITIES
# ---------------------------------------------------------
def download_file(link, file_name):
    response = requests.get(link, stream=True)
    if os.path.exists(file_name):
        return
    total_size = int(response.headers.get("content-length", 0))
    with open(file_name, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

@st.cache_resource(show_spinner="Downloading and loading MS2DeepScore model...")
def get_ms2deepscore_model():
    model_file_name = "ms2deepscore_model.pt"
    model_url = "https://zenodo.org/records/14290920/files/ms2deepscore_model.pt?download=1"
    if not os.path.exists(model_file_name):
        download_file(model_url, model_file_name)
    return load_model(model_file_name, allow_legacy=True)

# ---------------------------------------------------------
# 2. MOLNETENHANCER CONSENSUS FUNCTIONS (Python translation)
# ---------------------------------------------------------
def compute_highest_score(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculates majority class consensus and consensus score per component."""
    valid = df[df[column].notna() & (df[column] != "") & (df[column] != "NA")]
    
    if valid.empty:
        return pd.DataFrame(columns=["componentindex", f"{column}_Consensus", f"{column}_Score"])
    
    counts = valid.groupby(["componentindex", column]).size().reset_index(name="count")
    total_per_comp = counts.groupby("componentindex")["count"].transform("sum")
    counts["score"] = counts["count"] / total_per_comp
    
    # Select highest scoring class per component
    top_classes = counts.sort_values(["componentindex", "count"], ascending=[True, False]).drop_duplicates("componentindex")
    
    return top_classes[["componentindex", column, "score"]].rename(
        columns={column: f"{column}_Consensus", "score": f"{column}_Score"}
    )

def define_consensus_classes(data: pd.DataFrame) -> pd.DataFrame:
    """Propagates consensus classes to network components (excluding singletons)."""
    df = data.copy()
    
    # Include both NPC and ClassyFire ontology levels
    levels = [
        "NPC_Pathway", "NPC_Superclass", "NPC_Class",
        "CF_Superclass", "CF_Class", "CF_Subclass"
    ]
    
    for level in levels:
        consensus_df = compute_highest_score(df, level)
        cons_col = f"{level}_Consensus"
        score_col = f"{level}_Score"
        
        # SAFEGUARD: Ensure columns are added even if consensus_df is empty
        if consensus_df.empty:
            df[cons_col] = pd.NA
            df[score_col] = pd.NA
        else:
            df = df.merge(consensus_df, on="componentindex", how="left")
        
        # Keep original if singleton (-1)
        df.loc[df["componentindex"] == -1, cons_col] = df[level]
        
        # Propagate consensus to empty original entries
        fill_mask = (df[level].isna() | df[level].isin(["", "NA"])) & df[cons_col].notna()
        df.loc[fill_mask, level] = df.loc[fill_mask, cons_col]
        
    return df

# ---------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# ---------------------------------------------------------
st.set_page_config(page_title="MS2DeepScore & MolNetEnhancer", layout="wide")
st.title("MS2DeepScore Molecular Networking & Annotation Pipeline")

tab1, tab2 = st.tabs(["Step 1: Network & Spectral Mapping", "Step 2: Annotations & MolNetEnhancer"])

# ---------------------------------------------------------
# TAB 1: MS2 Processing & GraphML Generation
# ---------------------------------------------------------
with tab1:
    st.subheader("1. Generate MS2DeepScore Molecular Network")
    st.markdown("Upload raw/preprocessed positive and negative `.mgf` spectral files.")
    
    col1, col2 = st.columns(2)
    with col1:
        pos_file = st.file_uploader("Upload POS MGF", type=["mgf"], key="pos_mgf")
    with col2:
        neg_file = st.file_uploader("Upload NEG MGF", type=["mgf"], key="neg_mgf")
        
    st.markdown("""
    ### ⚙️ Network Parameters Guide
    * **Score Cutoff**: The minimum MS2DeepScore similarity (0.0 to 1.0) required to draw an edge between two spectra. Higher values (e.g., 0.85) produce stricter, highly confident connections and more isolated sub-graphs. Lowering it connects more distant chemical relatives but risks creating an unreadable "hairball".
    * **Max Links per Node**: Limits how many connections a single spectrum can have. Keeping this low (e.g., 10) ensures the network remains sparse and visually interpretable by only retaining the absolute top matches for each node.
    """)

   col_param1, col_param2, col_param3 = st.columns(3)
   
    with col_param1:
        score_cutoff = st.slider(
            "Score Cutoff", 
            min_value=0.50, max_value=0.99, value=0.85, step=0.01,
            help="Higher numbers produce more isolated sub-graphs."
        )
    with col_param2:
        max_links = st.slider(
            "Max Links per Node", 
            min_value=1, max_value=30, value=10, step=1,
            help="Lower numbers make sparser networks."
        )
    with col_param3:
        min_peaks = st.slider(
            "Minimum Peaks per Spectrum", 
            min_value=1, max_value=20, value=5, step=1,
            help="Filters out low-quality spectra with fewer than this many peaks."
        )
    st.markdown("---")

    if pos_file and neg_file:
        if st.button("Run Spectral Pipeline", key="btn_step1"):
            with st.spinner("Processing spectra, computing embeddings, and constructing graph..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    pos_path = os.path.join(tmpdir, "POS.mgf")
                    neg_path = os.path.join(tmpdir, "NEG.mgf")
                    with open(pos_path, "wb") as f: f.write(pos_file.getvalue())
                    with open(neg_path, "wb") as f: f.write(neg_file.getvalue())

                    spectra_neg = list(load_from_mgf(neg_path))
                    spectra_pos = list(load_from_mgf(pos_path))
                    combined_path = os.path.join(tmpdir, "combined.mgf")
                    save_as_mgf(spectra_neg + spectra_pos, combined_path)

                    # MatchMS Quality Filtering
                    workflow_clean = create_workflow(
                        query_filters=DEFAULT_FILTERS + [("require_minimum_number_of_peaks", {"n_required": min_peaks})],
                    )
                    pipeline_clean = Pipeline(workflow_clean)
                    pipeline_clean.run(combined_path)

                    # Separate & Add Persistent Identifiers
                    pos_cleaned, neg_cleaned = [], []
                    for spectrum in pipeline_clean.spectra_queries: # Use spectra_queries to avoid the deprecation warning!
                        if spectrum.get("ionmode") == "positive":
                            pos_cleaned.append(spectrum)
                        else:
                            neg_cleaned.append(spectrum)

                    mapping_records = []
                    
                    def get_safe_precursor_mz(spec):
                        """Safely extracts precursor m/z whether filtered or raw."""
                        prec_mz = spec.get("precursor_mz")
                        if prec_mz is None and spec.get("pepmass"):
                            pep = spec.get("pepmass")
                            prec_mz = pep[0] if isinstance(pep, (tuple, list)) else pep
                        return prec_mz

                    for i, spectrum in enumerate(pos_cleaned):
                        query_id = f"pos_{i + 1}"
                        spectrum.set("query_spectrum_nr", query_id)
                        scans = spectrum.get("scans") or spectrum.get("feature_id") or (i + 1)
                        mapping_records.append({
                            "SCANS": int(scans) if str(scans).isdigit() else scans,
                            "QUERY_SPECTRUM_NR": query_id,
                            "IONMODE": "positive",
                            "PRECURSOR_MZ": get_safe_precursor_mz(spectrum),
                            "RT": spectrum.get("rtinminutes") or spectrum.get("retention_time")
                        })

                    for i, spectrum in enumerate(neg_cleaned):
                        query_id = f"neg_{i + 1}"
                        spectrum.set("query_spectrum_nr", query_id)
                        scans = spectrum.get("scans") or spectrum.get("feature_id") or (i + 1)
                        mapping_records.append({
                            "SCANS": int(scans) if str(scans).isdigit() else scans,
                            "QUERY_SPECTRUM_NR": query_id,
                            "IONMODE": "negative",
                            "PRECURSOR_MZ": get_safe_precursor_mz(spectrum),
                            "RT": spectrum.get("rtinminutes") or spectrum.get("retention_time")
                        })

                    numbered_path = os.path.join(tmpdir, "cleaned_numbered.mgf")
                    all_cleaned = pos_cleaned + neg_cleaned
                    save_as_mgf(all_cleaned, numbered_path)

                    # Scoring Pipeline
                    model = get_ms2deepscore_model()
                    workflow_score = create_workflow(
                        query_filters=[],
                        score_computations=[[MS2DeepScore, {"model": model}]],
                    )
                    pipeline_score = Pipeline(workflow_score)
                    pipeline_score.run(numbered_path)

                    # Network Construction
                    ms2ds_network = SimilarityNetwork(
                        identifier_key="query_spectrum_nr",
                        score_cutoff=score_cutoff,
                        max_links=max_links,
                        link_method="mutual",
                    )
                    ms2ds_network.create_network(pipeline_score.scores, score_name="MS2DeepScore")

                    graphml_path = os.path.join(tmpdir, "ms2ds_graph.graphml")
                    ms2ds_network.export_to_graphml(graphml_path)

                    # Store session state files
                    with open(graphml_path, "rb") as f:
                        st.session_state["graphml_data"] = f.read()
                    with open(numbered_path, "rb") as f:
                        st.session_state["cleaned_mgf_data"] = f.read()
                        
                    mapping_df = pd.DataFrame(mapping_records)
                    st.session_state["data_cytoscape_df"] = mapping_df
                    st.session_state["data_cytoscape_csv"] = mapping_df.to_csv(index=False).encode("utf-8")
                    
                    # Mark step 1 as complete so buttons persist!
                    st.session_state["step1_complete"] = True

            st.success("Step 1 Complete! Download your files below.")
            
    # OUTSIDE the 'if st.button' block, display the download buttons
    if st.session_state.get("step1_complete"):
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.download_button(
                label="Download Network (.graphml)",
                data=st.session_state["graphml_data"],
                file_name="ms2ds_graph.graphml",
                mime="application/xml",
                key="btn_dl_graphml_tab1"
            )
        with dcol2:
            st.download_button(
                label="Download Node Mapping (data_cytoscape.csv)",
                data=st.session_state["data_cytoscape_csv"],
                file_name="data_cytoscape.csv",
                mime="text/csv",
                key="btn_dl_nodes_tab1"
            )
        with dcol3:
            st.download_button(
                label="Download Cleaned MGF (.mgf)",
                data=st.session_state["cleaned_mgf_data"],
                file_name="cleaned_spectra_pos_neg_numbered.mgf",
                mime="text/plain",
                key="btn_dl_mgf_tab1"
            )

# ---------------------------------------------------------
# TAB 2: Annotations Merging & MolNetEnhancer Execution
# ---------------------------------------------------------
with tab2:
    st.subheader("2. Run MolNetEnhancer (CANOPUS Only)")
    st.markdown(
        "Upload the SIRIUS/CANOPUS structure predictions (`canopus_structure_summary.tsv`) for each ionization mode. "
        "The graph component assignments will be derived directly from the generated `.graphml`."
    )
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Positive Mode Input**")
        pos_canopus_file = st.file_uploader("POS CANOPUS Structure (TSV)", type=["tsv", "txt"], key="pos_canopus", help="Upload canopus_structure_summary.tsv")
    with col_a2:
        st.markdown("**Negative Mode Input**")
        neg_canopus_file = st.file_uploader("NEG CANOPUS Structure (TSV)", type=["tsv", "txt"], key="neg_canopus", help="Upload canopus_structure_summary.tsv")

    # Allow uploading a previously saved data_cytoscape.csv or GraphML if not in state
    if "data_cytoscape_df" not in st.session_state or "graphml_data" not in st.session_state:
        st.warning("Step 1 data not found in current session memory. Please provide them below:")
        upl_col1, upl_col2 = st.columns(2)
        with upl_col1:
            fallback_map = st.file_uploader("Upload data_cytoscape.csv", type=["csv"], key="fallback_map")
            if fallback_map:
                st.session_state["data_cytoscape_df"] = pd.read_csv(fallback_map)
        with upl_col2:
            fallback_graph = st.file_uploader("Upload ms2ds_graph.graphml", type=["graphml", "xml"], key="fallback_graph")
            if fallback_graph:
                st.session_state["graphml_data"] = fallback_graph.getvalue()

    all_inputs_ready = (
        pos_canopus_file and neg_canopus_file and
        "data_cytoscape_df" in st.session_state and
        "graphml_data" in st.session_state
    )

    if all_inputs_ready:
        if st.button("Run MolNetEnhancer Enrichment", key="btn_step2"):
            with st.spinner("Mapping CANOPUS topologies and computing consensus..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    # 1. Parse Graph Structure & Connected Components
                    graph_file_path = os.path.join(tmpdir, "temp_graph.graphml")
                    with open(graph_file_path, "wb") as f:
                        f.write(st.session_state["graphml_data"])
                    
                    G = nx.read_graphml(graph_file_path)
                    
                    # Extract node table & components
                    node_rows = []
                    components = list(nx.connected_components(G))
                    
                    for comp_idx, comp_nodes in enumerate(components, start=1):
                        for node_id in comp_nodes:
                            degree = G.degree(node_id)
                            c_idx = -1 if degree == 0 else comp_idx
                            node_rows.append({"id": str(node_id), "componentindex": c_idx})
                            
                    net_df = pd.DataFrame(node_rows)

                    # 2. Process MS2 Base Data
                    ms2_data = st.session_state["data_cytoscape_df"].copy()
                    ms2_data = ms2_data.rename(columns={"SCANS": "row.ID", "QUERY_SPECTRUM_NR": "id"})
                    
                    ms2_data["id"] = ms2_data["id"].astype(str).str.strip()
                    ms2_data["row.ID"] = ms2_data["row.ID"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

                    # 3. Process CANOPUS Structure Summary data
                    canopus_pos = pd.read_csv(pos_canopus_file, sep="\t")
                    canopus_pos["IONMODE"] = "positive"
                    canopus_neg = pd.read_csv(neg_canopus_file, sep="\t")
                    canopus_neg["IONMODE"] = "negative"
                    
                    canopus_all = pd.concat([canopus_pos, canopus_neg], ignore_index=True)
                    
                    canopus_cols_map = {
                        "mappingFeatureId": "row.ID",
                        "NPC#pathway": "NPC_Pathway",
                        "NPC#superclass": "NPC_Superclass",
                        "NPC#class": "NPC_Class",
                        "ClassyFire#superclass": "CF_Superclass",
                        "ClassyFire#class": "CF_Class",
                        "ClassyFire#subclass": "CF_Subclass"
                    }
                    canopus_all = canopus_all.rename(columns=canopus_cols_map)
                    canopus_all["row.ID"] = canopus_all["row.ID"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                    
                    canopus_cols = ["row.ID", "IONMODE"] + [val for val in canopus_cols_map.values() if val != "row.ID"]
                    canopus_all = canopus_all[[c for c in canopus_cols if c in canopus_all.columns]]

                    # 4. Dual-Key Merging (Base = ms2_data to keep everything)
                    all_data = ms2_data.copy()
                    all_data = all_data.merge(canopus_all, on=["row.ID", "IONMODE"], how="left")
                    all_data = all_data.drop_duplicates(subset=["id"])

                    canopus_final = all_data[["id", "NPC_Pathway", "NPC_Superclass", "NPC_Class",
                                              "CF_Superclass", "CF_Class", "CF_Subclass"]].copy()

                    # 5. Join with Graph Components
                    final_table = canopus_final.merge(net_df, on="id", how="right")

                    # 6. Compute and Assign Consensus
                    defined_classes = define_consensus_classes(final_table)

                    output_cols = [
                        "id", "componentindex", 
                        "NPC_Pathway", "NPC_Superclass", "NPC_Class",
                        "NPC_Pathway_Consensus", "NPC_Pathway_Score",
                        "NPC_Superclass_Consensus", "NPC_Superclass_Score",
                        "NPC_Class_Consensus", "NPC_Class_Score",
                        "CF_Superclass", "CF_Class", "CF_Subclass",
                        "CF_Superclass_Consensus", "CF_Superclass_Score",
                        "CF_Class_Consensus", "CF_Class_Score",
                        "CF_Subclass_Consensus", "CF_Subclass_Score"
                    ]
                    
                    for col in output_cols:
                        if col not in defined_classes.columns:
                            defined_classes[col] = pd.NA
                            
                    output_df = defined_classes[output_cols].copy()
                    
                    # --- ISOLATE GRAPHML COLUMNS ---
                    # We only push the final consensus labels to the graph to keep it clean
                    graph_cols = [
                        "id", "componentindex",
                        "NPC_Pathway_Consensus", "NPC_Superclass_Consensus", "NPC_Class_Consensus",
                        "CF_Superclass_Consensus", "CF_Class_Consensus", "CF_Subclass_Consensus"
                    ]
                    graph_df = output_df[graph_cols].copy()
                    
                    # --- GRAPHML TYPE COMPATIBILITY FIX ---
                    text_cols_graph = [c for c in graph_cols if c not in ["id", "componentindex"]]
                    for col in text_cols_graph:
                        graph_df[col] = graph_df[col].fillna("").astype(str)
                        
                    graph_df["componentindex"] = pd.to_numeric(graph_df["componentindex"], errors='coerce').fillna(-1).astype(int)
                    # --------------------------------------

                    # Inject ONLY the consensus annotations back into the networkx Graph
                    node_attributes = graph_df.set_index("id").to_dict("index")
                    nx.set_node_attributes(G, node_attributes)
                    
                    # Export the enriched graph
                    enriched_graph_path = os.path.join(tmpdir, "enriched_ms2ds_graph.graphml")
                    nx.write_graphml(G, enriched_graph_path)
                    
                    # Save all outputs to session state
                    with open(enriched_graph_path, "rb") as f:
                        st.session_state["enriched_graphml"] = f.read()
                        
                    # Save the full DataFrame (with scores and raw classes) to the CSV
                    st.session_state["molnet_csv"] = output_df.to_csv(index=False).encode("utf-8")
                    st.session_state["step2_complete"] = True

            st.success("Step 2 Complete! MolNetEnhancer consensus successfully computed using CANOPUS data.")
            
    # DISPLAY TAB 2 DOWNLOAD BUTTONS EXACTLY ONCE
    if st.session_state.get("step2_complete"):
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.download_button(
                label="Download Enriched Network (.graphml)",
                data=st.session_state["enriched_graphml"],
                file_name="Enriched_MolNetEnhancer_Graph.graphml",
                mime="application/xml",
                key="btn_dl_enriched_graph_tab2"
            )
        with mcol2:
            st.download_button(
                label="Download Consensus Results (.csv)",
                data=st.session_state["molnet_csv"],
                file_name="Molnetenhancer_Consensus.csv",
                mime="text/csv",
                key="btn_dl_consensus_tab2"
            )