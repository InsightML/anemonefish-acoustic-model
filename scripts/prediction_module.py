# %% [markdown]
# # Anemonefish Call Prediction Module
# 
# This notebook implements a prediction module that processes audio files to detect anemonefish calls using the trained binary classifier. The module applies the exact same preprocessing pipeline as training and uses a sliding window approach for temporal analysis.
# 
# ## Pipeline Overview:
# 1. **Audio Loading & Preprocessing** - Load and normalize audio consistently with training
# 2. **Sliding Window Analysis** - Extract 1-second windows with overlap for high temporal resolution
# 3. **Spectrogram Generation** - Create spectrograms with identical parameters as training
# 4. **Model Prediction** - Apply trained classifier to each window
# 5. **Post-Processing** - Smooth predictions and detect events
# 6. **Results Export** - Generate timestamps and confidence scores
# 

# %% [markdown]
# ## 1. Setup and Imports
# 

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image
import json
from scipy import ndimage
from scipy.signal import medfilt
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")


# %% [markdown]
# ## 2. Configuration
# 
# All parameters must match the training setup exactly to ensure consistency.
# 

# %%
# --- CONFIGURATION ---

# Model and paths
MODEL_PATH = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/models/binary_classifier/checkpoints_run_7/best_model.keras'
AUDIO_INPUT_DIR = Path('/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/prediction_input')  # Input audio files
RESULTS_OUTPUT_DIR = Path('/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/results/predictions')  # Output results

# Spectrogram Parameters (MUST match training exactly)
SPECTROGRAM_CONFIG = {
    'n_fft': 1024,
    'hop_length': 256,  # n_fft // 4
    'fmax': 2000,  # Max frequency for anemonefish calls
    'width_pixels': 256,
    'height_pixels': 256,
    'target_sr': None,  # Use original sample rate
    'normalization_mean': [0.485, 0.456, 0.406],  # ImageNet normalization
    'normalization_std': [0.229, 0.224, 0.225]
}

# Sliding Window Parameters
WINDOW_CONFIG = {
    'window_duration': 0.4,  # seconds (matches training)
    'stride_duration': 0.2,  # seconds (high temporal resolution)
    'min_duration': 0.4     # minimum segment duration to process
}

# Prediction Parameters
PREDICTION_CONFIG = {
    'batch_size': 32,
    'probability_threshold': 0.5,
    'min_event_duration': 0.2,  # seconds
    'min_gap_duration': 0.1,    # seconds between events
    'smoothing_window': 5       # for median filtering
}

# Create output directories
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_INPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Configuration loaded:")
print(f"  Model: {MODEL_PATH}")
print(f"  Input Directory: {AUDIO_INPUT_DIR}")
print(f"  Output Directory: {RESULTS_OUTPUT_DIR}")
print(f"  Window Duration: {WINDOW_CONFIG['window_duration']}s")
print(f"  Stride Duration: {WINDOW_CONFIG['stride_duration']}s")
print(f"  Probability Threshold: {PREDICTION_CONFIG['probability_threshold']}")


# %% [markdown]
# ## 3. Audio Processing Functions
# 

# %%
def load_and_preprocess_audio(audio_path, target_sr=None):
    """
    Load and preprocess audio file consistently with training pipeline.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (None to use original)
    
    Returns:
        audio_data: Preprocessed audio array
        sample_rate: Sample rate of audio
        duration: Duration in seconds
    """
    try:
        # Load audio using librosa (matches training pipeline)
        audio_data, sample_rate = librosa.load(audio_path, sr=target_sr)
        
        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)
        
        duration = len(audio_data) / sample_rate
        
        print(f"Loaded audio: {duration:.2f}s at {sample_rate}Hz")
        return audio_data, sample_rate, duration
        
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None, None, None


def generate_sliding_windows(audio_data, sample_rate, window_duration, stride_duration):
    """
    Generate sliding window parameters for audio processing.
    
    Args:
        audio_data: Audio array
        sample_rate: Sample rate
        window_duration: Window size in seconds
        stride_duration: Stride size in seconds
    
    Returns:
        windows: List of (start_sample, end_sample, start_time, end_time) tuples
    """
    window_samples = int(window_duration * sample_rate)
    stride_samples = int(stride_duration * sample_rate)
    total_samples = len(audio_data)
    
    windows = []
    
    start_sample = 0
    while start_sample + window_samples <= total_samples:
        end_sample = start_sample + window_samples
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        
        windows.append((start_sample, end_sample, start_time, end_time))
        start_sample += stride_samples
    
    print(f"Generated {len(windows)} windows with {window_duration:.2f}s window, {stride_duration:.2f}s stride")
    return windows


# %% [markdown]
# ## 4. Spectrogram Generation (Training-Identical)
# 

# %%
def create_spectrogram_for_model(audio_segment, sample_rate, config):
    """
    Create spectrogram identical to training pipeline.
    This function replicates the exact spectrogram generation from notebook 1.
    
    Args:
        audio_segment: 1D audio array
        sample_rate: Sample rate
        config: Spectrogram configuration dictionary
    
    Returns:
        spectrogram: Preprocessed spectrogram ready for model input (H, W, C)
    """
    try:
        # Compute STFT (matches training)
        D = librosa.stft(audio_segment, 
                        n_fft=config['n_fft'], 
                        hop_length=config['hop_length'])
        
        # Convert to dB scale (matches training)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Create figure for rendering (matches training approach)
        dpi = 100
        width_inches = config['width_pixels'] / dpi
        height_inches = config['height_pixels'] / dpi
        
        fig, ax = plt.subplots(1, figsize=(width_inches, height_inches), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Plot spectrogram
        librosa.display.specshow(S_db, sr=sample_rate, 
                               hop_length=config['hop_length'],
                               x_axis=None, y_axis=None, 
                               fmax=config['fmax'], ax=ax)
        
        # Handle frequency limit
        num_frequency_bins = S_db.shape[0]
        if config['fmax'] is not None and sample_rate is not None:
            fmax_bin = int(config['fmax'] / (sample_rate / 2.0) * num_frequency_bins)
            if fmax_bin < num_frequency_bins:
                ax.set_ylim(0, fmax_bin)
        
        ax.axis('off')
        
        # Convert to array (fix for newer matplotlib versions)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
        plt.close(fig)
        
        # Resize to exact dimensions
        spectrogram = cv2.resize(buf, (config['width_pixels'], config['height_pixels']))
        
        # Normalize (matches training)
        spectrogram = spectrogram.astype(np.float32) / 255.0
        mean = np.array(config['normalization_mean'])
        std = np.array(config['normalization_std'])
        spectrogram = (spectrogram - mean) / std
        
        return spectrogram
        
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return None


def process_windows_to_spectrograms(audio_data, windows, sample_rate, config):
    """
    Process all windows to create spectrograms for model input.
    
    Args:
        audio_data: Full audio array
        windows: List of window parameters
        sample_rate: Sample rate
        config: Spectrogram configuration
    
    Returns:
        spectrograms: Array of spectrograms (N, H, W, C)
        timestamps: List of (start_time, end_time) for each window
    """
    spectrograms = []
    timestamps = []
    
    print(f"Processing {len(windows)} windows to spectrograms...")
    
    for start_sample, end_sample, start_time, end_time in tqdm(windows):
        # Extract audio segment
        audio_segment = audio_data[start_sample:end_sample]
        
        # Create spectrogram
        spectrogram = create_spectrogram_for_model(audio_segment, sample_rate, config)
        
        if spectrogram is not None:
            spectrograms.append(spectrogram)
            timestamps.append((start_time, end_time))
    
    if spectrograms:
        spectrograms = np.array(spectrograms)
        print(f"Created {len(spectrograms)} spectrograms with shape: {spectrograms.shape}")
    else:
        spectrograms = np.array([])
        print("No spectrograms created")
    
    return spectrograms, timestamps


# %% [markdown]
# ## 5. Model Loading and Prediction
# 

# %%
def load_trained_model(model_path):
    """
    Load the trained binary classifier model.
    
    Args:
        model_path: Path to the trained model
    
    Returns:
        model: Loaded Keras model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_batch(model, spectrograms, batch_size=32):
    """
    Run batch prediction on spectrograms.
    
    Args:
        model: Loaded Keras model
        spectrograms: Array of spectrograms
        batch_size: Batch size for prediction
    
    Returns:
        predictions: Array of probability scores
    """
    if len(spectrograms) == 0:
        return np.array([])
    
    print(f"Running prediction on {len(spectrograms)} spectrograms...")
    
    try:
        # Predict in batches
        predictions = model.predict(spectrograms, batch_size=batch_size, verbose=1)
        
        # Flatten predictions (from (N, 1) to (N,))
        predictions = predictions.flatten()
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Prediction range: {predictions.min():.3f} - {predictions.max():.3f}")
        
        return predictions
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return np.array([])


# %% [markdown]
# ## 6. Post-Processing and Event Detection
# 

# %%
def smooth_predictions(predictions, window_size=5):
    """
    Apply median filtering to smooth predictions.
    
    Args:
        predictions: Array of prediction scores
        window_size: Size of median filter window
    
    Returns:
        smoothed_predictions: Smoothed prediction array
    """
    if len(predictions) == 0:
        return predictions
    
    # Apply median filter
    smoothed = medfilt(predictions, kernel_size=window_size)
    
    print(f"Applied median filtering with window size {window_size}")
    return smoothed


def detect_events(predictions, timestamps, config):
    """
    Detect events from prediction stream using thresholding and post-processing.
    
    Args:
        predictions: Array of prediction scores
        timestamps: List of (start_time, end_time) for each prediction
        config: Prediction configuration
    
    Returns:
        events: List of detected events with start/end times and confidence
    """
    if len(predictions) == 0:
        return []
    
    threshold = config['probability_threshold']
    min_event_duration = config['min_event_duration']
    min_gap_duration = config['min_gap_duration']
    
    # Apply threshold
    binary_predictions = predictions > threshold
    
    # Find event boundaries
    events = []
    in_event = False
    event_start = None
    event_confidences = []
    
    for i, (is_event, (start_time, end_time)) in enumerate(zip(binary_predictions, timestamps)):
        if is_event and not in_event:
            # Start of new event
            in_event = True
            event_start = start_time
            event_confidences = [predictions[i]]
            
        elif is_event and in_event:
            # Continue current event
            event_confidences.append(predictions[i])
            
        elif not is_event and in_event:
            # End of current event
            in_event = False
            event_end = timestamps[i-1][1] if i > 0 else end_time
            event_duration = event_end - event_start
            
            # Check minimum duration
            if event_duration >= min_event_duration:
                mean_confidence = np.mean(event_confidences)
                max_confidence = np.max(event_confidences)
                
                events.append({
                    'start_time': event_start,
                    'end_time': event_end,
                    'duration': event_duration,
                    'mean_confidence': mean_confidence,
                    'max_confidence': max_confidence
                })
    
    # Handle case where file ends during an event
    if in_event and event_start is not None:
        event_end = timestamps[-1][1]
        event_duration = event_end - event_start
        
        if event_duration >= min_event_duration:
            mean_confidence = np.mean(event_confidences)
            max_confidence = np.max(event_confidences)
            
            events.append({
                'start_time': event_start,
                'end_time': event_end,
                'duration': event_duration,
                'mean_confidence': mean_confidence,
                'max_confidence': max_confidence
            })
    
    # Merge events that are too close together
    if len(events) > 1:
        merged_events = []
        current_event = events[0]
        
        for next_event in events[1:]:
            gap = next_event['start_time'] - current_event['end_time']
            
            if gap < min_gap_duration:
                # Merge events
                current_event['end_time'] = next_event['end_time']
                current_event['duration'] = current_event['end_time'] - current_event['start_time']
                current_event['mean_confidence'] = (current_event['mean_confidence'] + next_event['mean_confidence']) / 2
                current_event['max_confidence'] = max(current_event['max_confidence'], next_event['max_confidence'])
            else:
                # Keep separate
                merged_events.append(current_event)
                current_event = next_event
        
        merged_events.append(current_event)
        events = merged_events
    
    print(f"Detected {len(events)} events after post-processing")
    return events


# %% [markdown]
# ## 7. Results Export and Visualization
# 

# %%
def save_results(events, predictions, timestamps, audio_filename, output_dir):
    """
    Save prediction results in multiple formats.
    
    Args:
        events: List of detected events
        predictions: Array of all prediction scores
        timestamps: List of (start_time, end_time) for predictions
        audio_filename: Name of processed audio file
        output_dir: Output directory path
    
    Returns:
        result_files: Dictionary of created result files
    """
    base_name = Path(audio_filename).stem
    result_files = {}
    
    # 1. Save events as CSV
    if events:
        events_df = pd.DataFrame(events)
        events_csv_path = output_dir / f"{base_name}_events.csv"
        events_df.to_csv(events_csv_path, index=False)
        result_files['events_csv'] = events_csv_path
        print(f"Events saved to: {events_csv_path}")
    
    # 2. Save events as Audacity labels
    if events:
            audacity_path = output_dir / f"{base_name}_audacity_labels.txt"
            with open(audacity_path, 'w') as f:
                for event in events:
                    f.write(f"{event['start_time']:.3f}\t{event['end_time']:.3f}\tanemonefish_call\n")
            result_files['audacity_labels'] = audacity_path
            print(f"Audacity labels saved to: {audacity_path}")
    
    # 3. Save all predictions as CSV
    if len(predictions) > 0:
        pred_data = []
        for i, (pred, (start_time, end_time)) in enumerate(zip(predictions, timestamps)):
            pred_data.append({
                'window_index': i,
                'start_time': start_time,
                'end_time': end_time,
                'center_time': (start_time + end_time) / 2,
                'probability': pred
            })
        
        pred_df = pd.DataFrame(pred_data)
        pred_csv_path = output_dir / f"{base_name}_predictions.csv"
        pred_df.to_csv(pred_csv_path, index=False)
        result_files['predictions_csv'] = pred_csv_path
        print(f"All predictions saved to: {pred_csv_path}")
    
    # 4. Save summary JSON
    summary = {
        'audio_file': audio_filename,
        'processing_time': timestamps[-1][1] if timestamps else 0,
        'total_windows': len(predictions),
        'total_events': len(events),
        'events_summary': []
    }
    
    for i, event in enumerate(events):
        summary['events_summary'].append({
            'event_id': i + 1,
            'start_time': event['start_time'],
            'end_time': event['end_time'],
            'duration': event['duration'],
            'mean_confidence': event['mean_confidence'],
            'max_confidence': event['max_confidence']
        })
    
    summary_json_path = output_dir / f"{base_name}_summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    result_files['summary_json'] = summary_json_path
    print(f"Summary saved to: {summary_json_path}")
    
    return result_files


def plot_prediction_timeline(predictions, timestamps, events, audio_filename, output_dir):
    """
    Create visualization of prediction timeline.
    
    Args:
        predictions: Array of prediction scores
        timestamps: List of (start_time, end_time)
        events: List of detected events
        audio_filename: Name of audio file
        output_dir: Output directory
    """
    if len(predictions) == 0:
        return
    
    # Create time array from timestamps
    time_points = np.array([t[0] for t in timestamps])
    
    plt.figure(figsize=(15, 8))
    
    # Plot prediction scores
    plt.subplot(2, 1, 1)
    plt.plot(time_points, predictions, 'b-', alpha=0.7, label='Prediction Scores')
    plt.axhline(y=PREDICTION_CONFIG['probability_threshold'], color='r', linestyle='--', 
                label=f'Threshold {PREDICTION_CONFIG["probability_threshold"]}')
    
    # Mark detected events
    for event in events:
        plt.axvspan(event['start_time'], event['end_time'], alpha=0.3, color='green')
    
    plt.ylabel('Probability')
    plt.title(f'Anemonefish Call Detection Results: {Path(audio_filename).name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot binary detection
    plt.subplot(2, 1, 2)
    binary_preds = predictions > PREDICTION_CONFIG['probability_threshold']
    plt.plot(time_points, binary_preds.astype(int), 'g-', linewidth=2, label='Detections')
    
    # Mark events
    for i, event in enumerate(events):
        plt.axvspan(event['start_time'], event['end_time'], alpha=0.5, color='green')
        # Add event labels
        mid_time = (event['start_time'] + event['end_time']) / 2
        plt.text(mid_time, 0.5, f'Event {i+1}', ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.ylabel('Detection')
    plt.xlabel('Time (seconds)')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    base_name = Path(audio_filename).stem
    plot_path = output_dir / f"{base_name}_timeline.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Timeline plot saved to: {plot_path}")


# %% [markdown]
# ## 8. Main Processing Pipeline
# 

# %%
def process_audio_file(audio_path, model, output_dir):
    """
    Main processing pipeline for a single audio file.
    
    Args:
        audio_path: Path to audio file
        model: Loaded Keras model
        output_dir: Output directory for results
        
    Returns:
        processing_results: Dictionary with processing results
    """
    print(f"\n{'='*60}")
    print(f"Processing: {audio_path}")
    print(f"{'='*60}")
    
    # 1. Load and preprocess audio
    audio_data, sample_rate, duration = load_and_preprocess_audio(
        audio_path, SPECTROGRAM_CONFIG['target_sr']
    )
    
    if audio_data is None:
        print(f"Failed to load audio file: {audio_path}")
        return None
    
    # 2. Generate sliding windows
    windows = generate_sliding_windows(
        audio_data, sample_rate,
        WINDOW_CONFIG['window_duration'],
        WINDOW_CONFIG['stride_duration']
    )
    
    if not windows:
        print("No windows generated")
        return None
    
    # 3. Create spectrograms
    spectrograms, timestamps = process_windows_to_spectrograms(
        audio_data, windows, sample_rate, SPECTROGRAM_CONFIG
    )
    
    if len(spectrograms) == 0:
        print("No spectrograms created")
        return None
    
    # 4. Run model predictions
    predictions = predict_batch(
        model, spectrograms, PREDICTION_CONFIG['batch_size']
    )
    
    if len(predictions) == 0:
        print("No predictions generated")
        return None
    
    # 5. Apply smoothing
    smoothed_predictions = smooth_predictions(
        predictions, PREDICTION_CONFIG['smoothing_window']
    )
    
    # 6. Detect events
    events = detect_events(smoothed_predictions, timestamps, PREDICTION_CONFIG)
    
    # 7. Save results
    audio_filename = Path(audio_path).name
    result_files = save_results(events, smoothed_predictions, timestamps, 
                               audio_filename, output_dir)
    
    # 8. Create visualization
    plot_prediction_timeline(smoothed_predictions, timestamps, events, 
                           audio_filename, output_dir)
    
    # 9. Summary
    processing_results = {
        'audio_file': audio_filename,
        'duration': duration,
        'sample_rate': sample_rate,
        'total_windows': len(predictions),
        'total_events': len(events),
        'events': events,
        'result_files': result_files
    }
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE: {audio_filename}")
    print(f"Duration: {duration:.2f}s")
    print(f"Windows processed: {len(predictions)}")
    print(f"Events detected: {len(events)}")
    if events:
        total_event_duration = sum(event['duration'] for event in events)
        print(f"Total event duration: {total_event_duration:.2f}s")
        print(f"Coverage: {total_event_duration/duration*100:.1f}%")
    print(f"{'='*60}")
    
    return processing_results


def process_multiple_files(input_dir, model, output_dir):
    """
    Process multiple audio files in a directory.
    
    Args:
        input_dir: Directory containing audio files
        model: Loaded model
        output_dir: Output directory
        
    Returns:
        all_results: List of processing results
    """
    # Find audio files
    audio_extensions = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC']
    audio_files = []
    
    for ext in audio_extensions:
        files = list(input_dir.glob(f'*{ext}'))
        # Filter out hidden files (starting with .)
        files = [f for f in files if not f.name.startswith('.')]
        audio_files.extend(files)
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return []
    
    print(f"Found {len(audio_files)} audio files to process")
    
    all_results = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        try:
            result = process_audio_file(audio_file, model, output_dir)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    # Create overall summary
    if all_results:
        total_files = len(all_results)
        total_events = sum(r['total_events'] for r in all_results)
        total_duration = sum(r['duration'] for r in all_results)
        
        summary = {
            'total_files_processed': total_files,
            'total_audio_duration': total_duration,
            'total_events_detected': total_events,
            'files': all_results
        }
        
        summary_path = output_dir / 'batch_processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"Files processed: {total_files}")
        print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.1f}min)")
        print(f"Total events detected: {total_events}")
        print(f"Batch summary saved to: {summary_path}")
        print(f"{'='*80}")
    
    return all_results


# %% [markdown]
# ## 9. Load Model and Run Predictions
# 

# %%
# Load the trained model
print("Loading trained model...")
model = load_trained_model(MODEL_PATH)

if model is None:
    print("Failed to load model. Please check the MODEL_PATH in configuration.")
else:
    print("Model loaded successfully!")
    
    # Check if input directory has files
    if not AUDIO_INPUT_DIR.exists():
        print(f"Input directory does not exist: {AUDIO_INPUT_DIR}")
        print("Please create the directory and add your audio files.")
    else:
        # Find audio files in input directory
        audio_files = [f for f in list(AUDIO_INPUT_DIR.glob('*.wav')) + list(AUDIO_INPUT_DIR.glob('*.WAV')) if not f.name.startswith('.')]
        
        if not audio_files:
            print(f"No audio files found in {AUDIO_INPUT_DIR}")
            print("Please add .wav files to the input directory.")
            print(f"You can also modify AUDIO_INPUT_DIR in the configuration to point to your audio files.")
        else:
            print(f"Ready to process {len(audio_files)} audio files")
            print("Files found:")
            for f in audio_files:
                print(f"  - {f.name}")


# %% [markdown]
# ## 10. Process Audio Files
# 
# Run this cell to process all audio files in the input directory:
# 

# %%
# Process all audio files
if 'model' in locals() and model is not None:
    # Process all files in the input directory
    results = process_multiple_files(AUDIO_INPUT_DIR, model, RESULTS_OUTPUT_DIR)
    
    if results:
        print(f"\nProcessing completed successfully!")
        print(f"Results saved to: {RESULTS_OUTPUT_DIR}")
        
        # Show brief summary
        print("\nBrief Summary:")
        for result in results:
            print(f"  {result['audio_file']}: {result['total_events']} events detected")
    else:
        print("No files were processed successfully.")
else:
    print("Model not loaded. Please run the previous cell first.")


# %% [markdown]
# ## 11. Individual File Processing (Optional)
# 
# Use this section to process a single file for testing or detailed analysis:
# 

# %%
# Process a single file for testing
# Uncomment and modify the path below to process a specific file

# SINGLE_FILE_PATH = "/path/to/your/test/audio/file.wav"
# 
# if 'model' in locals() and model is not None:
#     from pathlib import Path
#     
#     test_file = Path(SINGLE_FILE_PATH)
#     if test_file.exists():
#         print(f"Processing single file: {test_file}")
#         result = process_audio_file(test_file, model, RESULTS_OUTPUT_DIR)
#         
#         if result:
#             print(f"Processing complete!")
#             print(f"Events detected: {result['total_events']}")
#         else:
#             print("Processing failed.")
#     else:
#         print(f"File not found: {test_file}")
# else:
#     print("Model not loaded.")

print("To process a single file, uncomment and modify the code above.")


# %% [markdown]
# ## Summary
# 
# This prediction module provides a complete pipeline for detecting anemonefish calls in audio recordings:
# 
# ### **Key Features:**
# - **Training-Consistent Preprocessing**: Exact replication of training spectrogram generation
# - **Efficient Sliding Window**: In-memory processing with configurable overlap
# - **Batch Processing**: Handles multiple files with progress tracking
# - **Post-Processing**: Smoothing and intelligent event detection
# - **Multiple Output Formats**: CSV, JSON, and Audacity-compatible labels
# - **Visualization**: Timeline plots showing prediction confidence and detected events
# 
# ### **Output Files Generated:**
# - `{filename}_events.csv`: Detected events with timestamps and confidence scores
# - `{filename}_audacity_labels.txt`: Labels that can be imported into Audacity
# - `{filename}_predictions.csv`: All prediction scores for detailed analysis
# - `{filename}_summary.json`: Processing summary and metadata
# - `{filename}_timeline.png`: Visualization of detection results
# 
# ### **Configuration Options:**
# - **Window Duration**: 1.0 seconds (matches training)
# - **Stride Duration**: 0.1 seconds (configurable for temporal resolution)
# - **Probability Threshold**: 0.5 (adjustable for sensitivity)
# - **Smoothing**: Median filtering to reduce noise
# - **Event Merging**: Intelligent consolidation of nearby detections
# 
# ### **Usage:**
# 1. Place audio files in the configured input directory
# 2. Run the processing cells in sequence
# 3. Results will be saved to the output directory with multiple format options
# 4. Review timeline visualizations to assess detection quality
# 
# The module is designed to be robust, efficient, and provide actionable results for acoustic monitoring of anemonefish populations.
# 


