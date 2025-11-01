# %% [markdown]
# # Data preprocessing: Audio file to Spectrogram
# 
# This is the second step in the data preprocessing pipeline. We convert segmented audio files into spectrogram images.
# 
# **Multi-Class Support**: This notebook now processes all classes defined in the YAML configuration file. It automatically generates spectrograms for each class (e.g., anemonefish, noise, biological) and organizes them into the appropriate output directories.
# 
# **Inputs**: Segmented `.wav` files from step 1 (stored in `_cache/1_generate_training_audio/{class_name}/`)
# **Outputs**: Spectrogram PNG images for each class (stored in `2_training_datasets/{dataset_version}/{class_name}/`)
# 
# The preprocessing configuration is also saved to the output dataset directory for reproducibility.

# %% [markdown]
# # 1. Introduction
# 
# This notebook processes segmented audio files from multiple classes, converting them into spectrograms suitable for training machine learning models. The goal is to visualize the frequency content over time, focusing on the relevant frequency range (e.g., 0-2000Hz for anemonefish calls). 
# 
# All parameters are loaded from a YAML configuration file, ensuring consistency with step 1 (audio segmentation) and enabling reproducibility. The notebook automatically processes all classes defined in the configuration.

# %% [markdown]
# # 2. Setup and Imports
# 
# This section imports the necessary Python libraries for audio processing, numerical operations, plotting, and file system interactions.

# %%
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import shutil
from pathlib import Path
from IPython.display import Image, display
from tqdm import tqdm

# %% [markdown]
# # 3. Configuration
# 
# **YAML-Based Configuration**: Load the same configuration file used in step 1 (audio segmentation). This ensures consistency across the preprocessing pipeline.
# 
# **Important**: Update the `CONFIG_PATH` variable below to point to the same YAML config file you used in notebook 1-1.
# 
# The configuration specifies:
# - Classes to process
# - Input/output directories
# - Spectrogram parameters (frequency range, FFT settings, output resolution)

# %%
# --- Load Configuration from YAML ---

# !!! UPDATE THIS PATH TO MATCH THE CONFIG USED IN NOTEBOOK 1-1 !!!
CONFIG_PATH = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/2_training_datasets/v2_biological/preprocessing_config_v2_biological.yaml'

# Load configuration
print(f"Loading configuration from: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Extract configuration values
WORKSPACE_BASE_PATH = Path(config['workspace_base_path'])
DATASET_VERSION = config['dataset_version']
CLASSES = config['classes']

# Construct paths
INPUT_AUDIO_BASE_DIR = WORKSPACE_BASE_PATH / 'data' / '_cache' / '1_generate_training_audio'
OUTPUT_BASE_DIR = WORKSPACE_BASE_PATH / 'data' / '2_training_datasets' / DATASET_VERSION

# Spectrogram parameters
FMAX_HZ = config['spectrogram']['fmax_hz']
N_FFT = config['spectrogram']['n_fft']
HOP_LENGTH = config['spectrogram']['hop_length']
WIDTH_PIXELS = config['spectrogram']['width_pixels']
HEIGHT_PIXELS = config['spectrogram']['height_pixels']

# Create output directory structure
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Configuration Loaded ===")
print(f"Dataset Version: {DATASET_VERSION}")
print(f"Classes to process: {CLASSES}")
print(f"Input directory: {INPUT_AUDIO_BASE_DIR}")
print(f"Output directory: {OUTPUT_BASE_DIR}")
print(f"\nSpectrogram parameters:")
print(f"  - Frequency range: 0-{FMAX_HZ} Hz")
print(f"  - FFT size: {N_FFT}")
print(f"  - Hop length: {HOP_LENGTH}")
print(f"  - Output size: {WIDTH_PIXELS}x{HEIGHT_PIXELS} pixels")

# Validate input directory exists
if not INPUT_AUDIO_BASE_DIR.exists():
    print(f"\nWARNING: Input directory {INPUT_AUDIO_BASE_DIR} does not exist. Please run notebook 1-1 first.")

# %% [markdown]
# # 4. Function to Generate and Save Spectrogram
# 
# This function takes an audio file path, loads the audio, computes its spectrogram, and saves it as a PNG image.

# %%
def create_and_save_spectrogram(audio_path, output_image_path, sr_target=None, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX_HZ):
    """
    Generates a spectrogram from an audio file and saves it as an image
    suitable for CNN input (no axes, labels, colorbar).

    Args:
        audio_path (Path or str): Path to the input audio file.
        output_image_path (Path or str): Path to save the output spectrogram image.
        sr_target (int, optional): Target sampling rate. If None, uses native.
                                   Consider sr_target= (e.g., 2 * fmax, so 4000 or 8000 for fmax=2000Hz).
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        fmax (int): Maximum frequency relevant for the STFT and display.
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr_target)

        # Compute Short-Time Fourier Transform (STFT)
        # The STFT will consider frequencies up to sr/2.
        # We are interested in fmax, so ensure sr is adequate.
        # If sr_target is set (e.g., 4000Hz for fmax=2000Hz), this is fine.
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

        # Convert amplitude to decibels
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Create a figure and axes with specific DPI for desired resolution
        dpi = 100  # Dots per inch - adjust as needed
        
        # Convert pixels to inches for figsize
        width_inches = WIDTH_PIXELS / dpi
        height_inches = HEIGHT_PIXELS / dpi
        
        fig, ax = plt.subplots(1, figsize=(width_inches, height_inches), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # Remove padding around the plot

        # Plot spectrogram
        # We use librosa.display.specshow which is convenient, but ensure it doesn't add extra whitespace.
        # The actual frequency range of S_db depends on sr and n_fft.
        # We'll plot the relevant portion up to fmax.
        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None, fmax=fmax, ax=ax)
        
        # Ensure the y-axis is limited to fmax if specshow's fmax isn't perfectly cropping the data display.
        # This cuts the *displayed* data, not the underlying STFT calculation.
        num_frequency_bins = S_db.shape[0]
        if fmax is not None and sr is not None:
            # Calculate the bin index corresponding to fmax
            fmax_bin = int(fmax / (sr / 2.0) * num_frequency_bins)
            if fmax_bin < num_frequency_bins : # Ensure fmax_bin is within bounds
                 ax.set_ylim(0, fmax_bin) # Set y-limit in terms of bins for specshow
            # If fmax is higher than Nyquist, librosa handles it by showing up to Nyquist.

        # Turn off all axes, labels, titles, colorbar
        ax.axis('off')
        
        # Save the figure at the specified resolution
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig) # Close the figure to free memory
        
        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False

# Example usage (optional, will be used in the next step)
# test_audio_file = AUDIO_DIR / 'your_sample_audio.wav' # Replace with an actual file name
# test_output_image = SPECTROGRAM_DIR / 'your_sample_audio_spectrogram.png'
# if test_audio_file.exists():
#    create_and_save_spectrogram(test_audio_file, test_output_image, fmax=2000)
# else:
#    print(f"Test audio file {test_audio_file} not found. Skipping example generation here.")

# %% [markdown]
# # 5. Process Audio Files for All Classes
# 
# This section processes all classes defined in the configuration:
# 1. For each class, it reads audio files from `_cache/1_generate_training_audio/{class_name}/`
# 2. Generates spectrograms for each audio file
# 3. Saves spectrograms to `2_training_datasets/{dataset_version}/{class_name}/`
# 4. Copies the preprocessing config to the output directory for reproducibility

# %%
# Initialize statistics
total_processed = 0
total_errors = 0
class_stats = {}
example_spectrogram_path = None

# Process each class
for class_name in CLASSES:
    print(f"\n{'='*60}")
    print(f"Processing class: {class_name.upper()}")
    print(f"{'='*60}")
    
    # Define input and output directories for this class
    class_input_dir = INPUT_AUDIO_BASE_DIR / class_name
    class_output_dir = OUTPUT_BASE_DIR / class_name
    
    # Create output directory
    class_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input directory exists
    if not class_input_dir.exists():
        print(f"WARNING: Input directory not found: {class_input_dir}")
        print(f"Skipping class '{class_name}'. Please run notebook 1-1 first.")
        continue
    
    # Find all audio files for this class
    audio_files = [f for f in class_input_dir.glob('*.wav') if not f.name.startswith('.')]
    
    if not audio_files:
        print(f"No .wav files found in {class_input_dir}")
        continue
    
    print(f"Found {len(audio_files)} audio files to process")
    print(f"Input:  {class_input_dir}")
    print(f"Output: {class_output_dir}")
    
    # Process files for this class
    class_processed = 0
    class_errors = 0
    
    for audio_file_path in tqdm(audio_files, desc=f"Processing {class_name}"):
        output_filename = audio_file_path.stem + '_spectrogram.png'
        output_image_path = class_output_dir / output_filename
        
        if create_and_save_spectrogram(audio_file_path, output_image_path, fmax=FMAX_HZ):
            class_processed += 1
            total_processed += 1
            if example_spectrogram_path is None:  # Store first successful spectrogram
                example_spectrogram_path = output_image_path
        else:
            class_errors += 1
            total_errors += 1
    
    # Store class statistics
    class_stats[class_name] = {
        'processed': class_processed,
        'errors': class_errors,
        'total_files': len(audio_files)
    }
    
    print(f"✓ Processed {class_processed}/{len(audio_files)} files for '{class_name}'")
    if class_errors > 0:
        print(f"✗ Encountered {class_errors} errors")

# Copy configuration file to output directory for reproducibility
config_output_path = OUTPUT_BASE_DIR / 'preprocessing_config.yaml'
shutil.copy2(CONFIG_PATH, config_output_path)
print(f"\n{'='*60}")
print(f"Saved preprocessing config to: {config_output_path}")

# Print final summary
print(f"\n{'='*60}")
print(f"PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Dataset Version: {DATASET_VERSION}")
print(f"Total spectrograms generated: {total_processed}")
if total_errors > 0:
    print(f"Total errors: {total_errors}")

print(f"\nBreakdown by class:")
for class_name in CLASSES:
    if class_name in class_stats:
        stats = class_stats[class_name]
        print(f"  {class_name}: {stats['processed']}/{stats['total_files']} files")
    else:
        print(f"  {class_name}: Not processed (no input files found)")

if example_spectrogram_path:
    print(f"\nExample spectrogram saved at: {example_spectrogram_path}")


# %% [markdown]
# # 6. Display an Example Spectrogram
# 
# If spectrograms were successfully generated, this section displays the first one that was created.

# %%
if example_spectrogram_path and example_spectrogram_path.exists():
    print(f"Displaying example spectrogram:")
    print(f"Path: {example_spectrogram_path}")
    display(Image(filename=str(example_spectrogram_path), width=800))
elif total_processed > 0:
    print("Spectrograms were generated, but unable to display an example.")
    print(f"Check the output directory: {OUTPUT_BASE_DIR}")
else:
    print("No spectrograms were generated.")


