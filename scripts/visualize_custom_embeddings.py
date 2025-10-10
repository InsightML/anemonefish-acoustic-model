import fiftyone as fo
import fiftyone.brain as fob
import os
from glob import glob
import numpy as np
import tensorflow as tf
import sys

# Add src to path to import from anemonefish_acoustics
# This allows the script to find the custom data processing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from anemonefish_acoustics.data_processing import SpectrogramDatasetBuilder, SpectrogramConfig
from anemonefish_acoustics.utils.logger import get_logger

# --- Configuration ---

# NOTE: PLEASE UPDATE THIS PATH to your trained encoder model
# It's typically found in 'models/autoencoder/checkpoints_run_*/best_encoder.keras'
ENCODER_MODEL_PATH = "/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/models/autoencoder/checkpoints_run_4/best_encoder.keras" 

# Data Paths - must match the data used for training the autoencoder
ANEMONEFISH_SPECS_PATH = "/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/1_binary_training_data/spectograms/anemonefish"
NOISE_SPECS_PATH = "/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/1_binary_training_data/spectograms/noise"
UNLABELED_SPECS_PATH = ""  # Optional: path to a directory of unlabeled spectrograms

# FiftyOne Configuration
DATASET_NAME = "custom_spectrogram_embeddings_run_4"
EMBEDDINGS_FIELD = "custom_embedding"  # Field to store custom embeddings
BRAIN_KEY_VIS = "custom_spectrogram_viz" # Key for the visualization run

# --- Logger Setup ---
logging = get_logger(name='visualize_custom_embeddings', workspace_root='/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics')


def create_fiftyone_dataset(dataset_name, anemonefish_path, noise_path, unlabelled_path):
    """
    Creates or loads a FiftyOne dataset with spectrogram images from specified paths.
    """
    if fo.dataset_exists(dataset_name):
        logging.info(f"Dataset '{dataset_name}' already exists. Deleting and recreating.")
        fo.delete_dataset(dataset_name)
    
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

    samples = []

    # Load anemonefish images
    anemonefish_files = [f for f in glob(os.path.join(anemonefish_path, "*.png")) if not os.path.basename(f).startswith('.')]
    for filepath in anemonefish_files:
        sample = fo.Sample(filepath=filepath)
        sample["ground_truth"] = fo.Classification(label="anemonefish")
        samples.append(sample)
    logging.info(f"Found {len(anemonefish_files)} anemonefish spectrograms.")

    # Load noise images
    noise_files = [f for f in glob(os.path.join(noise_path, "*.png")) if not os.path.basename(f).startswith('.')]
    for filepath in noise_files:
        sample = fo.Sample(filepath=filepath)
        sample["ground_truth"] = fo.Classification(label="noise")
        samples.append(sample)
    logging.info(f"Found {len(noise_files)} noise spectrograms.")

    # Load unlabelled images if path is provided
    if unlabelled_path and os.path.isdir(unlabelled_path):
        try:
            unlabelled_files = []
            for subdir in os.listdir(unlabelled_path):
                subdir_path = os.path.join(unlabelled_path, subdir)
                if not os.path.isdir(subdir_path): continue
                unlabelled_files.extend([
                    f for f in glob(os.path.join(subdir_path, "*.png"))
                    if not os.path.basename(f).startswith('.')
                ])
            for filepath in unlabelled_files:
                sample = fo.Sample(filepath=filepath)
                sample["ground_truth"] = fo.Classification(label="unlabelled")
                samples.append(sample)
            logging.info(f"Found {len(unlabelled_files)} unlabelled spectrograms.")
        except Exception as e:
            logging.error(f"Error loading unlabelled spectrograms: {e}")
    else:
        logging.info("No valid path provided for unlabelled spectrograms, skipping.")

    if not samples:
        logging.critical("CRITICAL: No image files found. Please check the data paths.")
        return None

    dataset.add_samples(samples)
    logging.info(f"Added {len(samples)} samples to the dataset '{dataset_name}'.")
    return dataset


def load_custom_encoder(model_path):
    """
    Loads the trained Keras encoder model.
    """
    if not os.path.exists(model_path):
        logging.error(f"Encoder model not found at '{model_path}'. Please update ENCODER_MODEL_PATH.")
        return None
    
    logging.info(f"Loading custom encoder from: {model_path}")
    try:
        encoder = tf.keras.models.load_model(model_path)
        logging.info("✅ Custom encoder loaded successfully.")
        encoder.summary(print_fn=logging.info)
        return encoder
    except Exception as e:
        logging.error(f"Error loading Keras model: {e}")
        return None


def generate_custom_embeddings(dataset, encoder, embeddings_field):
    """
    Generates embeddings for each sample in the dataset using the custom encoder.
    """
    logging.info(f"Generating embeddings using custom encoder. This may take a while...")
    
    # Use the same preprocessing config as the training script for consistency
    config = SpectrogramConfig()
    config.ENABLE_AUGMENTATION = False  # Disable augmentation for inference
    builder = SpectrogramDatasetBuilder(config)

    # Manually iterate, preprocess, and generate embeddings for each sample
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            image_path = sample.filepath
            
            # Preprocess the image exactly as in training: parse to [0,1], then scale to [-1,1]
            img_tensor_0_1 = builder.preprocessor.parse_image(image_path)
            img_tensor_minus1_1 = builder.preprocessor.scale_image(img_tensor_0_1)

            # Add batch dimension for the model
            img_batch = np.expand_dims(img_tensor_minus1_1, axis=0)

            # Generate embedding using the encoder
            embedding = encoder.predict(img_batch, verbose=0)[0]

            # Store the embedding on the sample and save
            sample[embeddings_field] = embedding.tolist()
            sample.save()
            
    logging.info(f"✅ Embeddings generated and stored in field '{embeddings_field}'.")


def main():
    """
    Main workflow to generate and visualize embeddings.
    """
    logging.info("Starting custom spectrogram embedding visualization process...")

    # 1. Load the custom encoder model
    encoder = load_custom_encoder(ENCODER_MODEL_PATH)
    if encoder is None:
        return

    # 2. Create or load the FiftyOne dataset
    dataset = create_fiftyone_dataset(
        DATASET_NAME, ANEMONEFISH_SPECS_PATH, NOISE_SPECS_PATH, UNLABELED_SPECS_PATH
    )
    if dataset is None:
        return

    # 3. Generate and store embeddings using the custom encoder
    generate_custom_embeddings(dataset, encoder, EMBEDDINGS_FIELD)

    # 4. Compute 2D visualization (UMAP) of the custom embeddings
    logging.info(f"Computing 2D visualization using UMAP (brain_key='{BRAIN_KEY_VIS}')...")
    fob.compute_visualization(
        dataset,
        embeddings=EMBEDDINGS_FIELD,
        brain_key=BRAIN_KEY_VIS,
        seed=42  # For reproducibility of UMAP
    )
    logging.info("Visualization computation complete.")

    # 5. Launch the FiftyOne App for exploration
    logging.info("Launching FiftyOne App...")
    session = fo.launch_app(dataset)
    logging.info(f"Session launched. Explore dataset '{DATASET_NAME}' in the App.")
    logging.info("To see class clusters, color the points by the 'ground_truth' field.")
    session.wait()

if __name__ == "__main__":
    main() 