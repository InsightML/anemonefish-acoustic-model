import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import os
from glob import glob
import torch # Added for GPU check

# --- Configuration ---
ANEMONEFISH_SPECS_PATH = "/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/1_binary_training_data/spectograms/anemonefish"
NOISE_SPECS_PATH = "/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/1_binary_training_data/spectograms/noise"
UNLABELED_SPECS_PATH = ""
DATASET_NAME = "spectrogram_embeddings_run_7"
EMBEDDINGS_FIELD = "image_embeddings" # Field to store raw embeddings
BRAIN_KEY_VIS = "spectrogram_viz" # Key for the visualization run

# Off-the-shelf model from FiftyOne Model Zoo
# Common choices: "mobilenet-v2-imagenet-torch", "resnet50-imagenet-torch"
ZOO_MODEL_NAME = "mobilenet-v2-imagenet-torch"


def create_fiftyone_dataset(dataset_name, anemonefish_path, noise_path, unlabelled_path):
    """
    Creates or loads a FiftyOne dataset with spectrogram images.
    """
    if fo.dataset_exists(dataset_name):
        print(f"Dataset '{dataset_name}' already exists. Deleting and recreating.")
        fo.delete_dataset(dataset_name)
    
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True # Make dataset persistent

    samples = []

    # Load anemonefish images
    anemonefish_files = [f for f in glob(os.path.join(anemonefish_path, "*.png")) 
                        if not os.path.basename(f).startswith('.')]
    for filepath in anemonefish_files:
        sample = fo.Sample(filepath=filepath)
        sample["ground_truth"] = fo.Classification(label="anemonefish")
        samples.append(sample)
    print(f"Found {len(anemonefish_files)} anemonefish spectrograms.")

    # Load noise images
    noise_files = [f for f in glob(os.path.join(noise_path, "*.png"))
                  if not os.path.basename(f).startswith('.')]
    for filepath in noise_files:
        sample = fo.Sample(filepath=filepath)
        sample["ground_truth"] = fo.Classification(label="noise")
        samples.append(sample)
    print(f"Found {len(noise_files)} noise spectrograms.")

    # Unlabelled images
    unlabelled_files = []
    try:
        for subdir in os.listdir(unlabelled_path):
            if not os.path.isdir(os.path.join(unlabelled_path, subdir)):
                continue
            unlabelled_files.extend([
                f for f in glob(os.path.join(unlabelled_path, subdir, "*.png"))
                if not os.path.basename(f).startswith('.')
            ])
        for filepath in unlabelled_files:
            sample = fo.Sample(filepath=filepath)
            sample["ground_truth"] = fo.Classification(label="unlabelled")
            samples.append(sample)
        print(f"Found {len(unlabelled_files)} unlabelled spectrograms.")
    except Exception as e:
        print(f"Error loading unlabelled spectrograms: {e}")
        print("Continuing with only anemonefish and noise spectrograms.")

    if not samples:
        print("CRITICAL: No image files found. Please check the paths.")
        return None

    dataset.add_samples(samples)
    print(f"Added {len(samples)} samples to the dataset '{dataset_name}'.")
    return dataset

def main():
    print("Starting spectrogram embedding visualization process...")

    # 1. Create or load the FiftyOne dataset
    dataset = create_fiftyone_dataset(DATASET_NAME, ANEMONEFISH_SPECS_PATH, NOISE_SPECS_PATH, UNLABELED_SPECS_PATH)
    if dataset is None:
        return

    # 2. Load an off-the-shelf model from the FiftyOne Model Zoo
    print(f"Loading pre-trained model: {ZOO_MODEL_NAME}...")
    # Ensure you have the necessary packages for the model, e.g., torch, torchvision
    # This might download model weights on first run
    try:
        # Load model (initially on CPU or default device)
        model = foz.load_zoo_model(ZOO_MODEL_NAME) 
    except Exception as e:
        print(f"Error loading or moving model '{ZOO_MODEL_NAME}': {e}")
        print("Please ensure the model name is correct and necessary dependencies (like PyTorch) are installed.")
        print("For PyTorch on Mac GPU, ensure your PyTorch version supports MPS.")
        return

    # 3. Compute embeddings for the dataset
    # The model will handle its own preprocessing (e.g., resizing, normalization)
    print(f"Computing embeddings using '{ZOO_MODEL_NAME}' and storing in field '{EMBEDDINGS_FIELD}'...")
    
    dataset.compute_embeddings(model, embeddings_field=EMBEDDINGS_FIELD)
    print("Embeddings computation complete.")

    # 4. Compute 2D visualization of the embeddings
    # This uses UMAP by default for dimensionality reduction
    print(f"Computing 2D visualization using UMAP (brain_key='{BRAIN_KEY_VIS}')...")
    fob.compute_visualization(
        dataset,
        embeddings=EMBEDDINGS_FIELD, # Use the field where raw embeddings are stored
        brain_key=BRAIN_KEY_VIS,
        seed=51 # For reproducibility of UMAP
    )
    print("Visualization computation complete.")

    # 5. Launch the FiftyOne App
    print("Launching FiftyOne App...")
    session = fo.launch_app(dataset)
    print(f"Session launched. Explore dataset '{DATASET_NAME}' in the App.")
    print("You can color the points by the 'ground_truth' field to see class clusters.")
    session.wait() # Keep the script running until the App is closed

if __name__ == "__main__":
    main() 