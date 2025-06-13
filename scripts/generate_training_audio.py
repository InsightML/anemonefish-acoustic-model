# %% [markdown]
# # Training Data Generation from Audio Tapes (Labeled Events & Noise)
# 
# This notebook processes long audio recordings (tapes) and their corresponding Audacity label files (TXT format) to extract segments for training an acoustic model. It generates data for two classes:
# 1.  **Target Sounds (Y=1)**: Extracted from time segments explicitly labeled in the Audacity files.
# 2.  **Noise (Y=0)**: Extracted from time segments that *do not* contain labeled sounds.
# 
# A sliding window approach is used to generate multiple audio clips from both target sound and noise regions.
# 
# **Process:**
# 1.  **Configuration**: Define paths to raw audio tapes, label files, and output directories. Set parameters for the sliding window (window size, slide/hop duration).
# 2.  **Parse Labels**: Read Audacity label files to identify time segments where target sounds are present.
# 3.  **Identify Noise Regions**: Determine the time segments in the audio tapes that *do not* contain labeled sounds.
# 4.  **Extract Segments with Sliding Window**:
#     *   Apply a sliding window to the labeled (target sound) regions to generate positive examples.
#     *   Apply the same sliding window to the identified noise regions to generate negative examples.
# 5.  **Save Segments**: Save the extracted audio clips as individual WAV files into respective output directories (`anemonefish` for target sounds, `noise` for noise).
# 
# The output will populate the `anemonefish` and `noise` folders within `data/1_binary_training_data/audio_files/`.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import glob
import pandas as pd
import soundfile as sf
import numpy as np
import random
import sys
sys.path.append('/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/src')

from anemonefish_acoustics.utils.logger import get_logger

# Setup logging
logging = get_logger()

# %% [markdown]
# ## 2. Configuration
# 
# Adjust the paths and parameters below as needed.
# - `INPUT_TAPES_AND_LABELS_DIR`: Directory containing both your full-length WAV audio tape files and their corresponding Audacity TXT label files. **It is assumed that a WAV file (e.g., `tape1.wav`) will have its label file as `tape1.txt` in this same directory.**
# - `OUTPUT_BASE_DIR`: The base directory where the `audio_files` folder (containing `noise` and `anemonefish` subfolders) is located.
# - `CHUNK_DURATION_SECONDS`: The desired duration for each extracted noise chunk (e.g., 1.0 for 1-second clips).
# - `MIN_NOISE_SEGMENT_DURATION_SECONDS`: Minimum duration for a segment to be considered for chunking noise. This should ideally be at least `CHUNK_DURATION_SECONDS`.

# %%
# --- Configuration ---

# !!! ADJUST THIS PATH TO YOUR DIRECTORY CONTAINING BOTH WAV TAPES AND TXT LABELS !!!
INPUT_TAPES_AND_LABELS_DIR = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/1_binary_training_data/tapes_and_labels' # Example path

# Output directory configuration
WORKSPACE_BASE_PATH = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics'
OUTPUT_AUDIO_FILES_DIR = os.path.join(WORKSPACE_BASE_PATH, 'data', '1_binary_training_data', 'audio_files')
OUTPUT_NOISE_DIR = os.path.join(OUTPUT_AUDIO_FILES_DIR, 'noise')
OUTPUT_ANEMONEFISH_DIR = os.path.join(OUTPUT_AUDIO_FILES_DIR, 'anemonefish') # Will now be populated by this script

# Audio segment extraction parameters
WINDOW_SIZE_SECONDS = 0.4  # Duration of each extracted audio window in seconds
SLIDE_SECONDS = 0.2      # Hop duration for the sliding window in seconds
# Minimum duration for a segment (either labeled or noise) to be considered for windowing.
# Must be at least WINDOW_SIZE_SECONDS to extract at least one window.
MIN_SEGMENT_DURATION_SECONDS = WINDOW_SIZE_SECONDS

# Noise padding parameters to prevent model bias
NOISE_PADDING_RATIO = 0.12  # Percentage of noise segments that will be randomly shortened and padded (0.0 to 1.0)
MIN_NOISE_DURATION_FOR_SHORTENING = 0.05  # Minimum duration in seconds for shortened noise segments
MAX_NOISE_DURATION_FOR_SHORTENING = 0.39  # Maximum duration in seconds for shortened noise segments

# Ensure output directories exist
os.makedirs(OUTPUT_NOISE_DIR, exist_ok=True)
os.makedirs(OUTPUT_ANEMONEFISH_DIR, exist_ok=True)

logging.info(f"Workspace Base Path: {WORKSPACE_BASE_PATH}")
logging.info(f"Input Tapes and Labels Directory: {INPUT_TAPES_AND_LABELS_DIR}")
logging.info(f"Output Noise Directory: {OUTPUT_NOISE_DIR}")
logging.info(f"Output Anemonefish Directory: {OUTPUT_ANEMONEFISH_DIR}")
logging.info(f"Audio Window Size: {WINDOW_SIZE_SECONDS}s")
logging.info(f"Sliding Window Hop: {SLIDE_SECONDS}s")
logging.info(f"Minimum Segment Duration for Processing: {MIN_SEGMENT_DURATION_SECONDS}s")
logging.info(f"Noise Padding Ratio: {NOISE_PADDING_RATIO} ({int(NOISE_PADDING_RATIO*100)}% of noise segments will be shortened and padded)")

# Check if input directory exists
if not os.path.isdir(INPUT_TAPES_AND_LABELS_DIR):
    logging.critical(f"Input tapes and labels directory not found: {INPUT_TAPES_AND_LABELS_DIR}")
    logging.critical("Please ensure your audio tapes and label files are in the correct path.")

# %% [markdown]
# ## 3. Helper Functions

# %%
def parse_audacity_labels(label_file_path):
    """
    Parses an Audacity label file (TXT, tab-separated).
    Assumes columns are: start_time (s), end_time (s), [label_name (optional)].
    Returns a list of tuples representing labeled segments: [(start1, end1), (start2, end2), ...].
    Labels are sorted by start time.
    """
    labeled_segments = []
    try:
        # Read with tab delimiter, no header, use first two columns
        df = pd.read_csv(label_file_path, sep='\t', header=None, usecols=[0, 1], float_precision='round_trip')
        for index, row in df.iterrows():
            start_time = float(row[0])
            end_time = float(row[1])
            if start_time < end_time: # Basic validation
                 labeled_segments.append((start_time, end_time))
            else:
                logging.warning(f"Skipping invalid segment in {label_file_path}: start_time {start_time} >= end_time {end_time}")

        # Sort segments by start time
        labeled_segments.sort(key=lambda x: x[0])
        logging.info(f"Parsed {len(labeled_segments)} segments from {label_file_path}")
    except FileNotFoundError:
        logging.error(f"Label file not found: {label_file_path}")
    except pd.errors.EmptyDataError:
        logging.warning(f"Label file is empty: {label_file_path}")
    except Exception as e:
        logging.error(f"Error parsing label file {label_file_path}: {e}")
    return labeled_segments

# %%
def get_noise_segments(total_duration_seconds, labeled_segments, min_segment_len_seconds):
    """
    Identifies noise segments in an audio file given its total duration and labeled (non-noise) segments.
    Args:
        total_duration_seconds (float): Total duration of the audio file.
        labeled_segments (list of tuples): Sorted list of (start, end) times for labeled regions.
        min_segment_len_seconds (float): Minimum duration for a segment to be considered noise.
                                         Segments shorter than this will be ignored.
    Returns:
        list of tuples: [(noise_start1, noise_end1), ...] for noise regions.
    """
    noise_segments = []
    current_time = 0.0

    # If no labeled segments, the whole file is noise
    if not labeled_segments:
        if total_duration_seconds >= min_segment_len_seconds:
            noise_segments.append((0.0, total_duration_seconds))
            logging.info(f"No labels found. Entire duration {total_duration_seconds:.2f}s considered noise for segmentation.")
        else:
            logging.info(f"No labels found. Entire duration {total_duration_seconds:.2f}s is less than min_segment_len_seconds {min_segment_len_seconds:.2f}s. No noise segments generated.")
        return noise_segments

    # Process segment from start of tape to the first label
    first_label_start = labeled_segments[0][0]
    if first_label_start > current_time:
        duration = first_label_start - current_time
        if duration >= min_segment_len_seconds:
            noise_segments.append((current_time, first_label_start))
        # else: logging.debug(f"Initial noise segment from {current_time:.2f} to {first_label_start:.2f} (duration {duration:.2f}s) too short.")
    current_time = max(current_time, labeled_segments[0][1]) # Move current time to end of first label

    # Process segments between labels
    for i in range(len(labeled_segments) - 1):
        end_current_label = labeled_segments[i][1]
        start_next_label = labeled_segments[i+1][0]
        
        # Ensure current_time is at least at the end of the current label before looking for a gap
        current_time = max(current_time, end_current_label) 

        if start_next_label > current_time: # If there's a gap
            duration = start_next_label - current_time
            if duration >= min_segment_len_seconds:
                noise_segments.append((current_time, start_next_label))
            # else: logging.debug(f"Noise segment between labels (from {current_time:.2f} to {start_next_label:.2f}, duration {duration:.2f}s) too short.")
        current_time = max(current_time, labeled_segments[i+1][1]) # Move current time to end of next label

    # Process segment from the end of the last label to the end of the file
    last_label_end = labeled_segments[-1][1]
    current_time = max(current_time, last_label_end) # Ensure current_time is at least at the end of the last label
    
    if total_duration_seconds > current_time:
        duration = total_duration_seconds - current_time
        if duration >= min_segment_len_seconds:
            noise_segments.append((current_time, total_duration_seconds))
        # else: logging.debug(f"Final noise segment (from {current_time:.2f} to {total_duration_seconds:.2f}, duration {duration:.2f}s) too short.")
            
    if noise_segments:
        logging.info(f"Identified {len(noise_segments)} noise segments meeting minimum duration of {min_segment_len_seconds:.2f}s.")
    else:
        logging.info(f"No noise segments meeting minimum duration of {min_segment_len_seconds:.2f}s were identified.")
    return noise_segments

# %%
def extract_save_segments_sliding_window(tape_audio_path, segments_to_process,
                                         window_duration_s, slide_duration_s, sr,
                                         output_dir_path, tape_basename, segment_type_prefix):
    """
    Extracts audio segments using a sliding window from the given audio tape and saves them.
    For 'anemonefish' type, segments shorter than window_duration_s will be padded with zeros.
    For 'noise' type, sliding windows are extracted normally, then some windows are randomly shortened and padded.
    Args:
        tape_audio_path (str): Path to the full audio tape WAV file.
        segments_to_process (list of tuples): List of (start_sec, end_sec) for regions to process.
        window_duration_s (float): Duration of each window in seconds.
        slide_duration_s (float): Hop duration for the sliding window in seconds.
        sr (int): Sample rate of the audio tape.
        output_dir_path (str): Directory to save the extracted audio windows.
        tape_basename (str): Basename of the tape file, for naming windows.
        segment_type_prefix (str): Prefix for the filename (e.g., 'noise', 'anemonefish').
    Returns:
        dict: Statistics containing counts of total windows, padded windows, and segment lengths.
    """
    saved_windows_count = 0
    padded_windows_count = 0
    segment_lengths = []
    window_len_samples = int(window_duration_s * sr)

    if not segments_to_process:
        logging.info(f"No segments to process for {segment_type_prefix} from {tape_basename}.")
        return {
            'total_windows': 0,
            'padded_windows': 0,
            'segment_lengths': []
        }

    for seg_idx, (seg_start_s, seg_end_s) in enumerate(segments_to_process):
        segment_duration_s = seg_end_s - seg_start_s
        segment_lengths.append(segment_duration_s)
        
        logging.debug(f"Processing segment {seg_idx+1}/{len(segments_to_process)} ({segment_type_prefix}): "
                      f"{seg_start_s:.2f}s - {seg_end_s:.2f}s (duration: {segment_duration_s:.2f}s) from {tape_basename}")

        if segment_duration_s <= 0:
            logging.warning(f"Segment {seg_idx+1} for {segment_type_prefix} from {tape_basename} has zero or negative duration. Skipping.")
            continue

        # Case 1: Segment is shorter than window_duration_s (only for anemonefish)
        if segment_duration_s < window_duration_s and segment_type_prefix == "anemonefish":
            logging.info(f"Segment {seg_idx+1} ({segment_type_prefix}, duration {segment_duration_s:.2f}s) is shorter than window size {window_duration_s:.2f}s. Padding...")
            start_sample = int(seg_start_s * sr)
            frames_to_read = int(segment_duration_s * sr) # Read only the actual segment

            if frames_to_read <= 0:
                logging.warning(f"Segment {seg_idx+1} ({segment_type_prefix}) resulted in {frames_to_read} frames to read. Skipping padding.")
                continue
            
            try:
                audio_segment_data, read_sr = sf.read(tape_audio_path, start=start_sample,
                                                      frames=frames_to_read, dtype='float32', always_2d=False)
                if sr != read_sr: # Should not happen if sf.info worked correctly
                    logging.warning(f"Sample rate mismatch during read: expected {sr}, got {read_sr}. Using original sr for padding calculation.")

                # Ensure audio_segment_data is 1D array
                if audio_segment_data.ndim > 1:
                    audio_segment_data = np.mean(audio_segment_data, axis=1) # Convert to mono by averaging if stereo

                num_padding_samples = window_len_samples - len(audio_segment_data)
                
                if num_padding_samples < 0: # Should ideally not happen if segment_duration_s < window_duration_s
                    logging.warning(f"Calculated negative padding ({num_padding_samples}) for short segment {seg_idx+1}. "
                                    f"Segment len: {len(audio_segment_data)}, target window len: {window_len_samples}. Clipping padding to 0.")
                    num_padding_samples = 0
                    # Potentially truncate if audio_segment_data is somehow longer than window_len_samples
                    audio_segment_data = audio_segment_data[:window_len_samples]

                padded_audio_data = np.pad(audio_segment_data, (0, num_padding_samples), 'constant', constant_values=(0.0, 0.0))

                if len(padded_audio_data) == window_len_samples:
                    window_filename = f"{tape_basename}_{segment_type_prefix}_window_padded_{saved_windows_count:04d}.wav"
                    output_window_path = os.path.join(output_dir_path, window_filename)
                    sf.write(output_window_path, padded_audio_data, sr)
                    saved_windows_count += 1
                    padded_windows_count += 1  # Track padded window
                else:
                    logging.warning(f"Padded audio for short segment {seg_idx+1} ({segment_type_prefix}) has unexpected length {len(padded_audio_data)} (expected {window_len_samples}). Skipping save.")
            except Exception as e:
                logging.error(f"Error processing/padding short segment {seg_idx+1} ({segment_type_prefix}) at {seg_start_s:.2f}s from {tape_audio_path}: {e}", exc_info=True)
            continue # Move to the next segment

        # Case 2: Segment is >= window_duration_s OR noise segment (regardless of length)
        # Apply sliding window approach
        current_window_start_s = seg_start_s
        while current_window_start_s + window_duration_s <= seg_end_s:
            start_sample = int(current_window_start_s * sr)
            
            try:
                audio_window_data, _ = sf.read(tape_audio_path, start=start_sample,
                                               frames=window_len_samples, dtype='float32', always_2d=False)
                
                if audio_window_data.ndim > 1: # Ensure mono
                    audio_window_data = np.mean(audio_window_data, axis=1)

                if len(audio_window_data) == window_len_samples:
                    # For noise segments, randomly decide whether to shorten and pad this window
                    if segment_type_prefix == "noise" and random.random() < NOISE_PADDING_RATIO:
                        # Randomly shorten this window and apply padding
                        random_duration = random.uniform(MIN_NOISE_DURATION_FOR_SHORTENING, MAX_NOISE_DURATION_FOR_SHORTENING)
                        random_samples = int(random_duration * sr)
                        
                        logging.info(f"Randomly shortening noise window at {current_window_start_s:.2f}s (original duration {window_duration_s:.2f}s) to {random_duration:.2f}s for padding.")
                        
                        # Truncate the window to the random duration
                        shortened_audio_data = audio_window_data[:random_samples]
                        
                        # Pad to match the original window length
                        num_padding_samples = window_len_samples - len(shortened_audio_data)
                        
                        if num_padding_samples < 0:
                            logging.warning(f"Calculated negative padding ({num_padding_samples}) for shortened noise window. Truncating.")
                            num_padding_samples = 0
                            shortened_audio_data = shortened_audio_data[:window_len_samples]

                        padded_audio_data = np.pad(shortened_audio_data, (0, num_padding_samples), 'constant', constant_values=(0.0, 0.0))
                        
                        if len(padded_audio_data) == window_len_samples:
                            window_filename = f"{tape_basename}_{segment_type_prefix}_window_padded_{saved_windows_count:04d}.wav"
                            output_window_path = os.path.join(output_dir_path, window_filename)
                            sf.write(output_window_path, padded_audio_data, sr)
                            saved_windows_count += 1
                            padded_windows_count += 1  # Track padded window
                        else:
                            logging.warning(f"Padded audio for shortened noise window has unexpected length {len(padded_audio_data)} (expected {window_len_samples}). Skipping save.")
                    else:
                        # Save the normal window (for anemonefish or non-selected noise windows)
                        window_filename = f"{tape_basename}_{segment_type_prefix}_window_{saved_windows_count:04d}.wav"
                        output_window_path = os.path.join(output_dir_path, window_filename)
                        sf.write(output_window_path, audio_window_data, sr)
                        saved_windows_count += 1
                else:
                    logging.warning(f"Could not read full window of {window_len_samples} samples "
                                    f"at {current_window_start_s:.2f}s from {tape_basename} for sliding window. "
                                    f"Read {len(audio_window_data)} samples. Skipping this window.")

            except Exception as e:
                logging.error(f"Error reading/writing sliding window at {current_window_start_s:.2f}s "
                              f"from {tape_audio_path} for {segment_type_prefix}: {e}", exc_info=True)
            
            current_window_start_s += slide_duration_s
            
    logging.info(f"Extracted and saved {saved_windows_count} '{segment_type_prefix}' windows from {tape_basename} (from {len(segments_to_process)} segments). Padded: {padded_windows_count}")
    
    return {
        'total_windows': saved_windows_count,
        'padded_windows': padded_windows_count,
        'segment_lengths': segment_lengths
    }

# %% [markdown]
# ## 4. Main Processing Loop
# 
# This section iterates through all WAV files in the `INPUT_TAPES_AND_LABELS_DIR`. For each tape:
# 1. It constructs the expected name for its corresponding `.txt` label file (e.g., if the tape is `audio_tape.wav`, it looks for `audio_tape.txt` in the same directory).
# 2. Parses the labels if the file exists; otherwise, it treats the entire tape as noise.
# 3. Gets the total duration of the tape.
# 4. Identifies noise segments.
# 5. Extracts and saves noise chunks from these segments into `OUTPUT_NOISE_DIR`.

# %%
def main_processing():
    if not os.path.isdir(INPUT_TAPES_AND_LABELS_DIR):
        logging.error("Input tapes and labels directory does not exist. Please check Configuration.")
        return

    tape_files = glob.glob(os.path.join(INPUT_TAPES_AND_LABELS_DIR, '*.wav')) + \
                 glob.glob(os.path.join(INPUT_TAPES_AND_LABELS_DIR, '*.WAV'))
    
    if not tape_files:
        logging.warning(f"No .wav or .WAV files found in {INPUT_TAPES_AND_LABELS_DIR}. Processing will not start.")
        return

    logging.info(f"Found {len(tape_files)} audio tapes for processing.")
    
    # Statistics tracking
    total_anemonefish_windows_generated = 0
    total_noise_windows_generated = 0
    total_anemonefish_padded = 0
    total_noise_padded = 0
    all_anemonefish_segment_lengths = []

    for tape_path in tape_files:
        tape_basename_with_ext = os.path.basename(tape_path)
        tape_basename = os.path.splitext(tape_basename_with_ext)[0]
        
        label_file_path = os.path.join(INPUT_TAPES_AND_LABELS_DIR, tape_basename + '.txt')
        
        logging.info(f"--- Processing tape: {tape_path} ---")

        labeled_segments_from_file = [] # Initialize to empty list
        if not os.path.exists(label_file_path):
            logging.warning(f"Label file not found for {tape_path} (expected at {label_file_path}). "
                            "This tape will only be processed for noise if the entire tape is considered noise.")
        else:
            labeled_segments_from_file = parse_audacity_labels(label_file_path)

        try:
            audio_info = sf.info(tape_path)
            total_duration = audio_info.duration
            sample_rate = audio_info.samplerate
            logging.info(f"Tape duration: {total_duration:.2f}s, Sample rate: {sample_rate}Hz")
        except Exception as e:
            logging.error(f"Could not read audio info for {tape_path}: {e}")
            continue # Skip to the next tape

        # Process Labeled Segments (Anemonefish / Y=1)
        if labeled_segments_from_file: # Check if parse_audacity_labels returned any segments
            logging.info(f"Processing {len(labeled_segments_from_file)} labeled (anemonefish) segments from {tape_basename_with_ext} with padding for short ones...")
            
            anemonefish_stats = extract_save_segments_sliding_window(
                tape_path,
                labeled_segments_from_file, # Use the original list of parsed segments
                WINDOW_SIZE_SECONDS,
                SLIDE_SECONDS,
                sample_rate,
                OUTPUT_ANEMONEFISH_DIR,
                tape_basename,
                "anemonefish" 
            )
            total_anemonefish_windows_generated += anemonefish_stats['total_windows']
            total_anemonefish_padded += anemonefish_stats['padded_windows']
            all_anemonefish_segment_lengths.extend(anemonefish_stats['segment_lengths'])
        else:
            # This log can occur if the label file existed but was empty or contained no valid segments.
            logging.info(f"No labeled segments found or parsed from {label_file_path} to process as anemonefish sounds for tape {tape_basename_with_ext}.")

        # Process Noise Segments (Y=0)
        noise_segments = get_noise_segments(total_duration, labeled_segments_from_file, MIN_SEGMENT_DURATION_SECONDS)
        
        if noise_segments:
            logging.info(f"Processing {len(noise_segments)} non-labeled (noise) segments from {tape_basename_with_ext}...")
            noise_stats = extract_save_segments_sliding_window(
                tape_path,
                noise_segments,
                WINDOW_SIZE_SECONDS,
                SLIDE_SECONDS,
                sample_rate,
                OUTPUT_NOISE_DIR,
                tape_basename,
                "noise" 
            )
            total_noise_windows_generated += noise_stats['total_windows']
            total_noise_padded += noise_stats['padded_windows']
        else:
            logging.info(f"No suitable noise segments found in {tape_basename_with_ext} (after considering MIN_SEGMENT_DURATION_SECONDS and labels) to extract noise windows.")
        
        logging.info(f"--- Finished processing tape: {tape_basename_with_ext} ---\n")

    # Calculate and display statistics
    logging.info(f"\n=== Processing Complete ===")
    logging.info(f"Total ANEMONEFISH windows generated across all tapes: {total_anemonefish_windows_generated}")
    logging.info(f"Total NOISE windows generated across all tapes: {total_noise_windows_generated}")
    
    # Padding statistics
    if total_anemonefish_windows_generated > 0:
        anemonefish_padding_percentage = (total_anemonefish_padded / total_anemonefish_windows_generated) * 100
        logging.info(f"ANEMONEFISH padding statistics: {total_anemonefish_padded}/{total_anemonefish_windows_generated} ({anemonefish_padding_percentage:.2f}%) windows used padding")
    else:
        logging.info("ANEMONEFISH padding statistics: No anemonefish windows generated")
    
    if total_noise_windows_generated > 0:
        noise_padding_percentage = (total_noise_padded / total_noise_windows_generated) * 100
        logging.info(f"NOISE padding statistics: {total_noise_padded}/{total_noise_windows_generated} ({noise_padding_percentage:.2f}%) windows used padding")
    else:
        logging.info("NOISE padding statistics: No noise windows generated")
    
    # Anemonefish segment length statistics
    if all_anemonefish_segment_lengths:
        import statistics
        logging.info(f"\n=== Anemonefish Segment Length Statistics ===")
        logging.info(f"Total anemonefish segments processed: {len(all_anemonefish_segment_lengths)}")
        logging.info(f"Segment length statistics (seconds):")
        logging.info(f"  Mean: {statistics.mean(all_anemonefish_segment_lengths):.3f}s")
        logging.info(f"  Median: {statistics.median(all_anemonefish_segment_lengths):.3f}s")
        logging.info(f"  Min: {min(all_anemonefish_segment_lengths):.3f}s")
        logging.info(f"  Max: {max(all_anemonefish_segment_lengths):.3f}s")
        logging.info(f"  Std Dev: {statistics.stdev(all_anemonefish_segment_lengths):.3f}s")
        
        # Count segments shorter than window size (these would use padding)
        short_segments = [length for length in all_anemonefish_segment_lengths if length < WINDOW_SIZE_SECONDS]
        if short_segments:
            short_percentage = (len(short_segments) / len(all_anemonefish_segment_lengths)) * 100
            logging.info(f"  Segments shorter than window size ({WINDOW_SIZE_SECONDS}s): {len(short_segments)}/{len(all_anemonefish_segment_lengths)} ({short_percentage:.2f}%)")
        else:
            logging.info(f"  All segments are >= window size ({WINDOW_SIZE_SECONDS}s)")
    else:
        logging.info("No anemonefish segments were processed")

if __name__ == '__main__':
    main_processing()