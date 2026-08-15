import os
import tempfile
import requests
from tqdm import tqdm
import streamlit as st


from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.Pipeline import Pipeline, create_workflow
from matchms.filtering.default_pipelines import DEFAULT_FILTERS
from matchms.networking import SimilarityNetwork
from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore

def download_file(link, file_name):
    # Your existing download logic here
    response = requests.get(link, stream=True)
    if os.path.exists(file_name):
        return
    total_size = int(response.headers.get('content-length', 0))
    with open(file_name, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

@st.cache_resource(show_spinner="Downloading and loading MS2DeepScore model... This happens once.")
def get_ms2deepscore_model():
    model_file_name = "ms2deepscore_model.pt"
    model_url = "https://zenodo.org/records/14290920/files/ms2deepscore_model.pt?download=1"
    
    # Download on the fly if it doesn't exist in the cloud environment
    if not os.path.exists(model_file_name):
        download_file(model_url, model_file_name)
        
    return load_model(model_file_name, allow_legacy=True)

st.title("MS2DeepScore Molecular Networking Pipeline")
st.write("Upload your POS and NEG MGF files to generate a molecular network.")

# 2. UI for file uploads
pos_file = st.file_uploader("Upload Positive Ionization MGF (POS.mgf)", type=["mgf"])
neg_file = st.file_uploader("Upload Negative Ionization MGF (NEG.mgf)", type=["mgf"])

if pos_file and neg_file:
    if st.button("Run Pipeline"):
        
        # 3. Processing block with a loading spinner
        with st.spinner("Processing spectra and calculating similarities. This may take a moment..."):
            
            # Create a temporary directory that cleans itself up automatically
            with tempfile.TemporaryDirectory() as tmpdir:
                
                # Write uploaded bytes to temp files
                pos_path = os.path.join(tmpdir, "POS.mgf")
                neg_path = os.path.join(tmpdir, "NEG.mgf")
                with open(pos_path, "wb") as f: f.write(pos_file.getvalue())
                with open(neg_path, "wb") as f: f.write(neg_file.getvalue())

                # Load and merge spectra
                spectra_neg = list(load_from_mgf(neg_path))
                spectra_pos = list(load_from_mgf(pos_path))
                all_spectra = spectra_neg + spectra_pos
                
                combined_path = os.path.join(tmpdir, "combined_spectra.mgf")
                save_as_mgf(all_spectra, combined_path)

                # Clean using matchms
                workflow_clean = create_workflow(
                    query_filters=DEFAULT_FILTERS + [("require_minimum_number_of_peaks", {"n_required": 5})],
                )
                pipeline_clean = Pipeline(workflow_clean)
                pipeline_clean.run(combined_path)

                # Separate pos/neg and add identifiers
                pos_cleaned = []
                neg_cleaned = []
                for spectrum in pipeline_clean.spectrums_queries:
                    if spectrum.get("ionmode") == "positive":
                        pos_cleaned.append(spectrum)
                    else:
                        neg_cleaned.append(spectrum)

                for i, spectrum in enumerate(pos_cleaned):
                    spectrum.set("query_spectrum_nr", "pos_" + str(i + 1))
                for i, spectrum in enumerate(neg_cleaned):
                    spectrum.set("query_spectrum_nr", "neg_" + str(i + 1))

                numbered_path = os.path.join(tmpdir, "cleaned_numbered.mgf")
                save_as_mgf(pos_cleaned + neg_cleaned, numbered_path)

                # Load model and compute MS2DeepScore similarities
                model = get_ms2deepscore_model()
                workflow_score = create_workflow(
                    query_filters=[],
                    score_computations=[[MS2DeepScore, {"model": model}]],
                )
                pipeline_score = Pipeline(workflow_score)
                pipeline_score.run(numbered_path)

                # Create the similarity network using your exact parameters
                ms2ds_network = SimilarityNetwork(
                    identifier_key="query_spectrum_nr",
                    score_cutoff=0.85,
                    max_links=10,
                    link_method="mutual",
                )
                ms2ds_network.create_network(pipeline_score.scores, score_name="MS2DeepScore")

                # Export to GraphML
                graphml_path = os.path.join(tmpdir, "ms2ds_graph.graphml")
                ms2ds_network.export_to_graphml(graphml_path)

                # Read the generated file back into memory for downloading
                with open(graphml_path, "rb") as f:
                    graphml_data = f.read()

        st.success("Pipeline complete! Your network is ready.")
        
        # 4. Provide a download button for the result
        st.download_button(
            label="Download Molecular Network (.graphml)",
            data=graphml_data,
            file_name="ms2ds_network.graphml",
            mime="application/xml"
        )