# %% [markdown]
# # 4. Generate Spectrograms from Unlabelled Long Audio Files
# 
# This notebook processes multiple datasets of unlabelled long WAV audio files. It iterates through all subdirectories in the unlabelled data folder (e.g., `20230208_First_B48_24h_audio_R`, etc.), and for each audio file in each dataset, it performs the following steps:
# 1.  Splits the long audio file into short, 0.4-second segments (aligned with training pipeline).
# 2.  For each 0.4-second audio segment, it generates a spectrogram image using native sample rates.
# 3.  Saves these spectrogram images to a specified output directory, organized by dataset and original file names.
# 
# This is useful for preparing multiple unlabelled audio datasets for further analysis or model training where spectrograms of fixed-length audio chunks are required. The spectrogram generation parameters are aligned with the training pipeline (`scripts/generate_training_audio.py`) for consistency across the codebase.

# %% [markdown]
# ## 1. Setup and Imports
# 
# This section imports the necessary Python libraries for audio processing, spectrogram generation, numerical operations, plotting, file system interactions, and logging.

# %%
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import soundfile as sf # For reading audio chunks efficiently
import logging
from IPython.display import Image, display
import shutil # For cleaning up temporary directory

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %% [markdown]
# ## 2. Configuration
# 
# Define the input and output directories, and parameters for chunking and spectrogram generation.
# 
# **Note: Configuration parameters are aligned with the training pipeline (`scripts/generate_training_audio.py`) for consistency across the codebase.**
# 
# - `INPUT_UNLABELLED_BASE_DIR`: Path to the base directory containing subdirectories with unlabelled audio datasets. Each subdirectory (e.g., `20230208_First_B48_24h_audio_R`) should contain `.wav` audio files and will be processed separately.
# - `OUTPUT_SPECTROGRAM_DIR`: Path to the directory where generated spectrogram images will be saved. Spectrograms will be organized by dataset and original WAV file names. A temporary directory `temp_chunks_for_spectrograms` will be created here during processing and should be removed afterwards.
# - `WINDOW_SIZE_SECONDS`: Duration of each audio window to be converted into a spectrogram (0.4 seconds, matching the training script).
# - `SR_TARGET_SPECTROGRAM`: Target sampling rate for spectrogram generation. Set to `None` to use the native sample rate of each audio file (aligned with training pipeline). This preserves original audio quality and avoids unnecessary resampling artifacts.
# - `FMAX_HZ`: Maximum frequency (in Hz) to display on the spectrogram (e.g., 2000 Hz for anemonefish calls).
# - `N_FFT`: FFT window size. Affects frequency resolution.
# - `HOP_LENGTH`: Hop length for STFT. Affects time resolution. Typically `N_FFT // 4` or `N_FFT // 2`.

# %%
# --- Configuration ---
# !!! IMPORTANT: Adjust these paths and parameters as needed !!!
WORKSPACE_BASE_PATH = Path('/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics') # Adjust if your workspace is different

# Base directory containing subdirectories with unlabelled audio files
# Each subdirectory (e.g., '20230208_First_B48_24h_audio_R') will be processed separately
INPUT_UNLABELLED_BASE_DIR = WORKSPACE_BASE_PATH / 'data' / 'unlabelled'
OUTPUT_SPECTROGRAM_DIR = WORKSPACE_BASE_PATH / 'data' / 'unlabelled_spectrograms'

# Audio window parameters - aligned with training pipeline (scripts/generate_training_audio.py)
WINDOW_SIZE_SECONDS = 0.4  # Duration of each audio window in seconds (matches training script)

# Spectrogram Parameters 
# NOTE: These parameters are aligned with the training pipeline for consistency:
# - SR_TARGET_SPECTROGRAM = None uses native sample rate (same as training script)
# - This preserves original audio quality and avoids unnecessary resampling artifacts
# - FMAX_HZ and FFT parameters remain optimized for anemonefish call detection
SR_TARGET_SPECTROGRAM = None  # Use native sample rate (aligned with training pipeline)
FMAX_HZ = 2000      # Max frequency for the calls of interest
N_FFT = 1024        # FFT window size
HOP_LENGTH = N_FFT // 4 # Hop length, typically 1/4 of N_FFT

# --- End Configuration ---

# Ensure base input directory exists
INPUT_UNLABELLED_BASE_DIR.mkdir(parents=True, exist_ok=True)
# Create output directory if it doesn't exist
OUTPUT_SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)

logging.info(f"Input Unlabelled Base Directory: {INPUT_UNLABELLED_BASE_DIR.resolve()}")
logging.info(f"Output Spectrogram Directory: {OUTPUT_SPECTROGRAM_DIR.resolve()}")
logging.info(f"Window Size: {WINDOW_SIZE_SECONDS}s (aligned with training pipeline)")
logging.info(f"Spectrogram Target SR: {'Native (no resampling)' if SR_TARGET_SPECTROGRAM is None else f'{SR_TARGET_SPECTROGRAM} Hz'}")
logging.info(f"Spectrogram Fmax: {FMAX_HZ} Hz")
logging.info(f"Spectrogram N_FFT: {N_FFT}")
logging.info(f"Spectrogram Hop Length: {HOP_LENGTH}")

if not INPUT_UNLABELLED_BASE_DIR.is_dir():
    logging.critical(f"CRITICAL: Input unlabelled base directory {INPUT_UNLABELLED_BASE_DIR} could not be confirmed as a directory. Please check the path.")
else:
    # Find all subdirectories in the unlabelled base directory
    subdirs = [d for d in INPUT_UNLABELLED_BASE_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if subdirs:
        logging.info(f"Found {len(subdirs)} subdirectories to process: {[d.name for d in subdirs]}")
    else:
        logging.warning(f"No subdirectories found in {INPUT_UNLABELLED_BASE_DIR}. Creating a dummy directory with test WAV file for demonstration.")
        dummy_dir = INPUT_UNLABELLED_BASE_DIR / 'dummy_test_dataset'
        dummy_dir.mkdir(exist_ok=True)
        dummy_wav_path = dummy_dir / 'dummy_test_audio_10s.wav'
        try:
            sr_dummy = 44100; duration_dummy = 10; frequency_dummy = 440
            t_dummy = np.linspace(0, duration_dummy, int(sr_dummy * duration_dummy), False)
            audio_dummy = 0.5 * np.sin(2 * np.pi * frequency_dummy * t_dummy)
            sf.write(str(dummy_wav_path), audio_dummy, sr_dummy)
            logging.info(f"Created dummy directory and WAV file: {dummy_wav_path}")
        except Exception as e:
            logging.error(f"Could not create dummy directory/WAV file: {e}")

# %% [markdown]
# ## 3. Helper Function: Generate and Save Spectrogram
# 
# This function is adapted from `notebooks/1_audio_to_spectogram.ipynb`. It takes an audio file path, loads the audio (optionally resampling it to `sr_target`), computes its spectrogram focusing on the `0-FMAX_HZ` range, and saves it as a PNG image. The image is formatted to be suitable for machine learning input (no axes, labels, or colorbars).

# %%
def create_and_save_spectrogram(audio_path, output_image_path, sr_target=None, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX_HZ):
    """
    Generates a spectrogram from an audio file and saves it as an image
    suitable for CNN input (no axes, labels, colorbar).

    Args:
        audio_path (Path or str): Path to the input audio file.
        output_image_path (Path or str): Path to save the output spectrogram image.
        sr_target (int, optional): Target sampling rate for librosa.load(). If None, uses native.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        fmax (int): Maximum frequency relevant for the STFT and display.
    """
    try:
        y, sr = librosa.load(Path(audio_path), sr=sr_target)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        fig_width_inches = 5 
        fig_height_inches = 5
        
        fig, ax = plt.subplots(1, figsize=(fig_width_inches, fig_height_inches)) 
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) 

        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None, fmax=fmax, ax=ax)
        
        ax.axis('off') 
        
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig) 
        
        return True
    except Exception as e:
        logging.error(f"Error generating spectrogram for {audio_path}: {e}")
        # import traceback # Uncomment for detailed debugging
        # logging.error(traceback.format_exc()) # Uncomment for detailed debugging
        return False

# %% [markdown]
# ## 4. Helper Function: Process Long Audio File into Spectrogram Chunks
# 
# This function orchestrates the processing of a single long audio file:
# 1.  Reads audio file metadata (native sample rate, duration) using `soundfile.info()`.
# 2.  Calculates how many full 0.4-second chunks can be extracted (aligned with training pipeline).
# 3.  Creates a temporary subdirectory (within `OUTPUT_SPECTROGRAM_DIR/temp_chunks_for_spectrograms/`) to store the 0.4-second WAV chunks for the current long file.
# 4.  Iterates through the long audio file, extracting each 0.4-second chunk at its native sample rate using `soundfile.read()`.
# 5.  If the source audio is stereo, it's converted to mono by taking the first channel.
# 6.  Saves each 0.4-second audio chunk as a temporary WAV file.
# 7.  Calls `create_and_save_spectrogram()` for each temporary WAV chunk. This function will use the native sample rate (no resampling) to align with the training pipeline.
# 8.  Spectrograms are saved into a subdirectory within `OUTPUT_SPECTROGRAM_DIR` named after the original long WAV file, for better organization (e.g., `OUTPUT_SPECTROGRAM_DIR/original_wav_stem/`).
# 9.  Deletes each temporary 0.4-second WAV file immediately after its spectrogram is generated.
# 10. After all chunks from the long audio file are processed, the temporary subdirectory for its chunks is removed.

# %%
def process_audio_file_to_spectrogram_chunks(
    long_audio_path: Path, 
    output_spectrogram_base_dir: Path, 
    chunk_duration_s: float, 
    sr_target_spectrogram: int, 
    n_fft_spec: int, 
    hop_length_spec: int, 
    fmax_spec: int
    ) -> int:

    tape_basename = long_audio_path.stem # This is 'wavname'
    parent_dir_name = long_audio_path.parent.name # This is 'dirname'
    processed_chunk_count = 0
    
    # Temporary directory for 0.4-second WAV chunks specific to this long_audio_file
    temp_chunk_file_storage_dir = output_spectrogram_base_dir / "temp_chunks_for_spectrograms" / tape_basename
    temp_chunk_file_storage_dir.mkdir(parents=True, exist_ok=True)

    # Output directory for spectrograms from this specific long_audio_file
    # Spectrograms will be saved in: output_spectrogram_base_dir / tape_basename / *.png
    file_specific_spectrogram_output_dir = output_spectrogram_base_dir / tape_basename
    file_specific_spectrogram_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        audio_info = sf.info(str(long_audio_path))
        sr_native = audio_info.samplerate
        total_duration_native = audio_info.duration
        
        if total_duration_native < chunk_duration_s:
            logging.warning(f"Audio file {long_audio_path.name} ({total_duration_native:.2f}s) is shorter than chunk duration ({chunk_duration_s}s). Skipping.")
            shutil.rmtree(temp_chunk_file_storage_dir) # Clean up temp dir for this file
            return 0

        chunk_len_samples_native = int(chunk_duration_s * sr_native)
        num_full_chunks_possible = int(total_duration_native // chunk_duration_s)

        logging.info(f"Processing {long_audio_path.name}: Native SR={sr_native}Hz, Duration={total_duration_native:.2f}s. Expecting {num_full_chunks_possible} full {chunk_duration_s}s chunks.")

        for i in range(num_full_chunks_possible):
            start_sample_native = i * chunk_len_samples_native
            
            try:
                # Read the chunk data at its native sample rate
                audio_chunk_data, read_sr = sf.read(str(long_audio_path), start=start_sample_native, frames=chunk_len_samples_native, dtype='float32', always_2d=False)
                
                if audio_chunk_data.ndim > 1 : # If stereo or more channels, take first channel to make it mono
                    audio_chunk_data = audio_chunk_data[:, 0]
                
                if len(audio_chunk_data) == chunk_len_samples_native: # Ensure full chunk read
                    # Temporary path for the 0.4-second WAV chunk
                    temp_chunk_filename = f"{tape_basename}_temp_chunk_{i:04d}.wav" # Temporary name, doesn't need parent dir
                    temp_chunk_path = temp_chunk_file_storage_dir / temp_chunk_filename
                    
                    # Save the 0.4-second chunk at its native sample rate
                    sf.write(str(temp_chunk_path), audio_chunk_data, sr_native)

                    # New spectrogram filename: dirname-wavname-chunk_xxxx_spectrogram.png
                    output_spectrogram_filename = f"{parent_dir_name}-{tape_basename}-chunk_{i:04d}_spectrogram.png"
                    output_spectrogram_path = file_specific_spectrogram_output_dir / output_spectrogram_filename
                    
                    # Generate spectrogram from the temporary 0.4-second WAV chunk
                    success = create_and_save_spectrogram(
                        audio_path=temp_chunk_path, 
                        output_image_path=output_spectrogram_path, 
                        sr_target=sr_target_spectrogram, # Uses native sample rate (aligned with training pipeline)
                        n_fft=n_fft_spec, 
                        hop_length=hop_length_spec, 
                        fmax=fmax_spec
                    )
                    if success:
                        processed_chunk_count += 1
                    
                    temp_chunk_path.unlink() # Delete temporary 0.4-second WAV chunk
                else:
                    logging.warning(f"Chunk {i} from {long_audio_path.name}: Expected {chunk_len_samples_native} samples, got {len(audio_chunk_data)}. Skipping.")
            
            except Exception as e_chunk:
                logging.error(f"Error processing chunk {i} from {long_audio_path.name}: {e_chunk}")
                # import traceback; logging.error(traceback.format_exc()) # For detailed debugging

        logging.info(f"Generated {processed_chunk_count} spectrograms from {long_audio_path.name}.")
        
    except Exception as e_file:
        logging.error(f"Major error processing file {long_audio_path.name}: {e_file}")
        # import traceback; logging.error(traceback.format_exc()) # For detailed debugging
    finally:
        # Clean up the temporary directory for this specific long_audio_file's chunks
        if temp_chunk_file_storage_dir.exists():
            try:
                shutil.rmtree(temp_chunk_file_storage_dir)
                logging.debug(f"Cleaned up temporary chunk directory: {temp_chunk_file_storage_dir}")
            except Exception as e_clean:
                logging.error(f"Could not remove temporary chunk directory {temp_chunk_file_storage_dir}: {e_clean}. Please remove manually.")
                
    return processed_chunk_count

# %% [markdown]
# ## 5. Main Processing Loop
# 
# This section iterates through all subdirectories in `INPUT_UNLABELLED_BASE_DIR`. For each subdirectory, it processes all `.wav` (and `.WAV`) audio files within that directory, invoking `process_audio_file_to_spectrogram_chunks` to generate and save spectrograms for all 0.4-second segments (aligned with training pipeline).

# %%
def run_main_processing_for_unlabelled_spectrograms():
    if not INPUT_UNLABELLED_BASE_DIR.is_dir():
        logging.error(f"Input unlabelled base directory {INPUT_UNLABELLED_BASE_DIR} is not valid. Aborting.")
        return

    # Find all subdirectories in the unlabelled base directory
    subdirs = [d for d in INPUT_UNLABELLED_BASE_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not subdirs:
        logging.warning(f"No subdirectories found in {INPUT_UNLABELLED_BASE_DIR}. Nothing to process.")
        return

    logging.info(f"Found {len(subdirs)} dataset subdirectories to process: {[d.name for d in subdirs]}")
    
    # Overall statistics across all datasets
    total_spectrograms_generated_all_datasets = 0
    total_files_successfully_processed = 0
    total_datasets_processed = 0

    for dataset_dir in subdirs:
        logging.info(f"\n{'='*60}")
        logging.info(f"PROCESSING DATASET: {dataset_dir.name}")
        logging.info(f"{'='*60}")
        
        # Find all audio files in this dataset directory
        all_audio_files = sorted(list(set(list(dataset_dir.glob('*.wav')) + list(dataset_dir.glob('*.WAV')))))
        all_audio_files = [f for f in all_audio_files if not f.name.startswith('.')] # Filter out hidden files

        if not all_audio_files:
            logging.warning(f"No .wav or .WAV files (non-hidden) found in {dataset_dir}. Skipping this dataset.")
            continue

        logging.info(f"Found {len(all_audio_files)} audio files to process in dataset '{dataset_dir.name}'.")
        
        # Statistics for this dataset
        dataset_spectrograms_generated = 0
        dataset_files_successfully_processed = 0

        for audio_file_path in all_audio_files:
            logging.info(f"--- Starting processing for: {dataset_dir.name}/{audio_file_path.name} ---")
            spectrograms_from_this_file = process_audio_file_to_spectrogram_chunks(
                long_audio_path=audio_file_path,
                output_spectrogram_base_dir=OUTPUT_SPECTROGRAM_DIR,
                chunk_duration_s=WINDOW_SIZE_SECONDS,
                sr_target_spectrogram=SR_TARGET_SPECTROGRAM,
                n_fft_spec=N_FFT, 
                hop_length_spec=HOP_LENGTH,
                fmax_spec=FMAX_HZ
            )
            dataset_spectrograms_generated += spectrograms_from_this_file
            if spectrograms_from_this_file > 0: # Consider a file processed if at least one spectrogram was made
                dataset_files_successfully_processed += 1
            logging.info(f"--- Finished processing: {dataset_dir.name}/{audio_file_path.name}. Generated {spectrograms_from_this_file} spectrograms. ---")

        # Update overall statistics
        total_spectrograms_generated_all_datasets += dataset_spectrograms_generated
        total_files_successfully_processed += dataset_files_successfully_processed
        if dataset_files_successfully_processed > 0:
            total_datasets_processed += 1

        logging.info(f"\n--- DATASET '{dataset_dir.name}' SUMMARY ---")
        logging.info(f"Successfully processed {dataset_files_successfully_processed} out of {len(all_audio_files)} audio files.")
        logging.info(f"Generated {dataset_spectrograms_generated} spectrograms from dataset '{dataset_dir.name}'.")

    # Final cleanup and overall summary
    logging.info(f"\n{'='*60}")
    logging.info(f"OVERALL PROCESSING COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"Successfully processed {total_datasets_processed} out of {len(subdirs)} datasets.")
    logging.info(f"Total audio files processed: {total_files_successfully_processed}")
    logging.info(f"Total 0.4-second spectrograms generated across all datasets: {total_spectrograms_generated_all_datasets}")
    
    parent_temp_dir_for_all_chunks = OUTPUT_SPECTROGRAM_DIR / "temp_chunks_for_spectrograms"
    if parent_temp_dir_for_all_chunks.exists():
        if not any(parent_temp_dir_for_all_chunks.iterdir()): # Check if it's empty
            try:
                parent_temp_dir_for_all_chunks.rmdir()
                logging.info(f"Successfully removed empty parent temporary directory: {parent_temp_dir_for_all_chunks}")
            except Exception as e_clean_parent:
                logging.warning(f"Parent temporary directory {parent_temp_dir_for_all_chunks} is empty but could not be removed: {e_clean_parent}")
        else: # If it's not empty, it means some file-specific temp dirs were not cleaned up
            logging.warning(f"Parent temporary directory {parent_temp_dir_for_all_chunks} still contains subdirectories. This might indicate an issue during cleanup for specific files. Please check manually.")

# === Execute Main Processing ===
run_main_processing_for_unlabelled_spectrograms()

# %% [markdown]
# ## 6. Display an Example Spectrogram
# 
# If spectrograms were generated, this section attempts to find and display the first one created from the first processed audio file.

# %%
def display_example_generated_spectrogram():
    example_spectrogram_to_display = None
    
    if OUTPUT_SPECTROGRAM_DIR.is_dir():
        # Find the first dataset directory that was processed
        subdirs = [d for d in INPUT_UNLABELLED_BASE_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for dataset_dir in subdirs:
            # Find the first audio file in this dataset
            all_audio_files_for_example = sorted(list(set(list(dataset_dir.glob('*.wav')) + list(dataset_dir.glob('*.WAV')))))
            all_audio_files_for_example = [f for f in all_audio_files_for_example if not f.name.startswith('.')]

            if all_audio_files_for_example:
                first_audio_file_stem_for_example = all_audio_files_for_example[0].stem
                # Spectrograms are in OUTPUT_SPECTROGRAM_DIR / <original_wav_stem> /
                spectrogram_output_subdir_for_example = OUTPUT_SPECTROGRAM_DIR / first_audio_file_stem_for_example
                
                if spectrogram_output_subdir_for_example.is_dir():
                    # Find any .png file in that subdirectory
                    potential_spectrograms = sorted(list(spectrogram_output_subdir_for_example.glob('*_spectrogram.png')))
                    if potential_spectrograms:
                        example_spectrogram_to_display = potential_spectrograms[0]
                        break  # Found an example, stop looking

    if example_spectrogram_to_display and example_spectrogram_to_display.exists():
        logging.info(f"Displaying example spectrogram: {example_spectrogram_to_display}")
        display(Image(filename=str(example_spectrogram_to_display), width=300)) # Display smaller
    else:
        logging.info("No example spectrogram found to display. Check if processing completed and generated files in the expected output structure.")

# === Display Example ===
display_example_generated_spectrogram()


