"""
Visualization utilities for anemonefish acoustics.

This module provides functions for visualizing training progress,
dataset samples, and model results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import librosa
import librosa.display


def plot_training_history(history: Dict[str, List[float]], 
                         output_path: Optional[str] = None,
                         show_figure: bool = False) -> None:
    """
    Plot training history metrics.
    
    Parameters
    ----------
    history : Dict[str, List[float]]
        Dictionary containing training metrics by epoch
    output_path : Optional[str], optional
        Path to save the plot, by default None
    show_figure : bool, optional
        Whether to display the figure, by default False
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Over Time')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Training Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    if 'val_recall' in history:
        axes[1].plot(history['val_recall'], label='Validation Recall')
    if 'val_precision' in history:
        axes[1].plot(history['val_precision'], label='Validation Precision')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric Value')
    axes[1].set_title('Metrics Over Time')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_audio_waveform(audio_data: np.ndarray, 
                       sr: int = 8000, 
                       title: str = 'Audio Waveform',
                       output_path: Optional[str] = None,
                       show_figure: bool = False) -> None:
    """
    Plot audio waveform.
    
    Parameters
    ----------
    audio_data : np.ndarray
        Audio data to plot
    sr : int, optional
        Sample rate, by default 8000
    title : str, optional
        Plot title, by default 'Audio Waveform'
    output_path : Optional[str], optional
        Path to save the plot, by default None
    show_figure : bool, optional
        Whether to display the figure, by default False
    """
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio_data, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_spectrogram(spectrogram: np.ndarray, 
                    sr: int = 8000, 
                    hop_length: int = 512,
                    fmin: float = 0.0,
                    fmax: float = 2000.0,
                    title: str = 'Spectrogram',
                    output_path: Optional[str] = None,
                    show_figure: bool = False,
                    duration: Optional[float] = None) -> None:
    """
    Plot spectrogram.
    
    Parameters
    ----------
    spectrogram : np.ndarray
        Spectrogram data to plot
    sr : int, optional
        Sample rate, by default 8000
    hop_length : int, optional
        Hop length, by default 512
    fmin : float, optional
        Minimum frequency, by default 0.0
    fmax : float, optional
        Maximum frequency, by default 2000.0
    title : str, optional
        Plot title, by default 'Spectrogram'
    output_path : Optional[str], optional
        Path to save the plot, by default None
    show_figure : bool, optional
        Whether to display the figure, by default False
    duration : Optional[float], optional
        Duration of the audio in seconds, if provided will override the x-axis calculation, by default None
    """
    plt.figure(figsize=(12, 5))
    
    # If spectrogram has more than 2 dimensions (e.g., it's a batch), take the first one
    if spectrogram.ndim > 2:
        spectrogram = spectrogram[0]
    
    # If it's a channel-first format, transpose it
    if spectrogram.shape[0] == 1:
        spectrogram = spectrogram[0]
        
    # Convert to dB scale if not already
    if np.max(spectrogram) > 1.0:
        spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    # Plot spectrogram
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, 
                            x_axis='time', y_axis='hz', fmin=fmin, fmax=fmax)
    
    # Set x-axis limit if duration is provided
    if duration is not None:
        plt.xlim([0, duration])
        plt.xlabel(f'Time (seconds) - Duration: {duration:.1f}s')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_feature_comparison(original_features: np.ndarray, 
                           augmented_features: np.ndarray,
                           feature_type: str = 'spectrogram',
                           sr: int = 8000,
                           hop_length: int = 512,
                           output_path: Optional[str] = None,
                           show_figure: bool = False) -> None:
    """
    Plot comparison between original and augmented features.
    
    Parameters
    ----------
    original_features : np.ndarray
        Original feature data
    augmented_features : np.ndarray
        Augmented feature data
    feature_type : str, optional
        Type of feature ('spectrogram', 'mfcc', etc.), by default 'spectrogram'
    sr : int, optional
        Sample rate, by default 8000
    hop_length : int, optional
        Hop length, by default 512
    output_path : Optional[str], optional
        Path to save the plot, by default None
    show_figure : bool, optional
        Whether to display the figure, by default False
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    if feature_type.lower() == 'spectrogram':
        # Convert to dB scale if not already
        if np.max(original_features) > 1.0:
            original_features_db = librosa.amplitude_to_db(original_features, ref=np.max)
        else:
            original_features_db = original_features
            
        if np.max(augmented_features) > 1.0:
            augmented_features_db = librosa.amplitude_to_db(augmented_features, ref=np.max)
        else:
            augmented_features_db = augmented_features
        
        # Plot original spectrogram
        librosa.display.specshow(original_features_db, sr=sr, hop_length=hop_length, 
                                x_axis='time', y_axis='hz', ax=axes[0])
        axes[0].set_title('Original Spectrogram')
        
        # Plot augmented spectrogram
        im = librosa.display.specshow(augmented_features_db, sr=sr, hop_length=hop_length, 
                                    x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title('Augmented Spectrogram')
        
        fig.colorbar(im, ax=axes, format='%+2.0f dB')
    
    elif feature_type.lower() == 'mfcc':
        # Plot original MFCCs
        librosa.display.specshow(original_features, x_axis='time', sr=sr, hop_length=hop_length, ax=axes[0])
        axes[0].set_title('Original MFCCs')
        
        # Plot augmented MFCCs
        im = librosa.display.specshow(augmented_features, x_axis='time', sr=sr, hop_length=hop_length, ax=axes[1])
        axes[1].set_title('Augmented MFCCs')
        
        fig.colorbar(im, ax=axes)
    
    else:
        # Generic feature plot
        axes[0].imshow(original_features, aspect='auto', origin='lower')
        axes[0].set_title(f'Original {feature_type.capitalize()}')
        
        im = axes[1].imshow(augmented_features, aspect='auto', origin='lower')
        axes[1].set_title(f'Augmented {feature_type.capitalize()}')
        
        fig.colorbar(im, ax=axes)
    
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(confusion_matrix: np.ndarray, 
                         class_names: List[str],
                         title: str = 'Confusion Matrix',
                         output_path: Optional[str] = None,
                         show_figure: bool = False) -> None:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix data
    class_names : List[str]
        List of class names
    title : str, optional
        Plot title, by default 'Confusion Matrix'
    output_path : Optional[str], optional
        Path to save the plot, by default None
    show_figure : bool, optional
        Whether to display the figure, by default False
    """
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add axis labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_prediction_samples(audio_data: List[np.ndarray],
                          predictions: List[float],
                          ground_truth: List[int],
                          sr: int = 8000,
                          sample_indices: Optional[List[int]] = None,
                          max_samples: int = 5,
                          output_dir: Optional[str] = None,
                          show_figures: bool = False) -> None:
    """
    Plot sample predictions with their waveforms.
    
    Parameters
    ----------
    audio_data : List[np.ndarray]
        List of audio samples
    predictions : List[float]
        List of prediction probabilities
    ground_truth : List[int]
        List of ground truth labels
    sr : int, optional
        Sample rate, by default 8000
    sample_indices : Optional[List[int]], optional
        Indices of samples to plot, by default None (random selection)
    max_samples : int, optional
        Maximum number of samples to plot, by default 5
    output_dir : Optional[str], optional
        Directory to save plots, by default None
    show_figures : bool, optional
        Whether to display the figures, by default False
    """
    n_samples = len(audio_data)
    
    # Select samples to plot
    if sample_indices is None:
        # Randomly select samples
        num_samples = min(max_samples, n_samples)
        sample_indices = np.random.choice(range(n_samples), num_samples, replace=False)
    else:
        # Use provided indices
        sample_indices = sample_indices[:max_samples]
    
    for i, idx in enumerate(sample_indices):
        plt.figure(figsize=(12, 6))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio_data[idx], sr=sr)
        plt.title(f"Sample {idx}")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot prediction vs ground truth
        plt.subplot(2, 1, 2)
        prediction_label = 'Anemonefish' if predictions[idx] > 0.5 else 'Background'
        truth_label = 'Anemonefish' if ground_truth[idx] == 1 else 'Background'
        
        bars = plt.bar(['Prediction', 'Ground Truth'], 
                      [predictions[idx], ground_truth[idx]],
                      color=['blue', 'green'])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   prediction_label if bar.get_x() == 0 else truth_label,
                   ha='center', va='bottom')
        
        plt.ylim(0, 1.2)
        plt.ylabel('Probability / Label')
        
        plt.tight_layout()
        
        if output_dir:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"prediction_sample_{idx}.png")
            plt.savefig(output_path)
            
        if show_figures:
            plt.show()
        else:
            plt.close()


def plot_preprocessing_stages(audio_stages: Dict[str, np.ndarray],
                             feature_stages: Dict[str, np.ndarray],
                             sr: int = 8000,
                             hop_length: int = 512,
                             fmin: float = 0.0,
                             fmax: float = 2000.0,
                             title: str = 'Preprocessing Pipeline',
                             output_path: Optional[str] = None,
                             show_figure: bool = False,
                             feature_type: str = 'mel_spectrogram',
                             duration: Optional[float] = None) -> None:
    """
    Plot the stages of preprocessing from raw audio to final features.
    
    Parameters
    ----------
    audio_stages : Dict[str, np.ndarray]
        Dictionary containing audio at different preprocessing stages.
        Example keys: 'raw', 'normalized', 'trimmed', etc.
    feature_stages : Dict[str, np.ndarray]
        Dictionary containing extracted features at different preprocessing stages.
        Example keys: 'spectrogram', 'mel_spectrogram', 'augmented', etc.
    sr : int, optional
        Sample rate, by default 8000
    hop_length : int, optional
        Hop length for spectrogram calculation, by default 512
    fmin : float, optional
        Minimum frequency for spectrogram display, by default 0.0
    fmax : float, optional
        Maximum frequency for spectrogram display, by default 2000.0
    title : str, optional
        Main title for the plot, by default 'Preprocessing Pipeline'
    output_path : Optional[str], optional
        Path to save the plot, by default None
    show_figure : bool, optional
        Whether to display the figure, by default False
    feature_type : str, optional
        Type of feature being visualized, by default 'mel_spectrogram'
    duration : Optional[float], optional
        Duration in seconds to set on x-axis for standardized audio, by default None
    """
    # Count the total number of visualizations
    num_audio_stages = len(audio_stages)
    num_feature_stages = len(feature_stages)
    total_stages = num_audio_stages + num_feature_stages
    
    # Figure sizing and setup - rows will be audio waves and then features
    fig = plt.figure(figsize=(15, 3 * total_stages))
    fig.suptitle(title, fontsize=16)
    
    # Counter for subplot positioning
    plot_idx = 1
    
    # Plot audio stages
    for stage_name, audio in audio_stages.items():
        ax = fig.add_subplot(total_stages, 1, plot_idx)
        if audio is not None:
            librosa.display.waveshow(audio, sr=sr, ax=ax)
            ax.set_title(f"Audio: {stage_name}")
            ax.set_ylabel("Amplitude")
            
            # For standardized audio, set the x-axis limit if duration is provided
            if "standardized" in stage_name and duration is not None:
                ax.set_xlim([0, duration])
                ax.set_xlabel(f"Time (s) - Duration: {duration:.1f}s")
            # Only show x-label (time) on the last audio stage if no duration
            elif plot_idx == num_audio_stages:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xlabel("")
        else:
            ax.text(0.5, 0.5, f"No data for audio stage: {stage_name}", 
                   horizontalalignment='center', verticalalignment='center')
        plot_idx += 1
    
    # Plot feature stages
    for stage_name, feature in feature_stages.items():
        ax = fig.add_subplot(total_stages, 1, plot_idx)
        if feature is not None:
            # Handle different feature types
            if feature_type.lower() == 'mfcc':
                im = librosa.display.specshow(feature, x_axis='time', sr=sr, 
                                           hop_length=hop_length, ax=ax)
                ax.set_title(f"Feature: {stage_name} (MFCC)")
            else:
                # For spectrograms, convert to dB if not already
                if np.max(feature) > 1.0 and 'db' not in stage_name.lower():
                    feature_db = librosa.amplitude_to_db(feature, ref=np.max)
                else:
                    feature_db = feature
                
                im = librosa.display.specshow(feature_db, sr=sr, hop_length=hop_length,
                                           x_axis='time', y_axis='hz', 
                                           fmin=fmin, fmax=fmax, ax=ax)
                ax.set_title(f"Feature: {stage_name}")
            
            # For standardized features, set the x-axis limit if duration is provided
            if "standardized" in stage_name and duration is not None:
                ax.set_xlim([0, duration])
                
            fig.colorbar(im, ax=ax, format='%+2.0f dB')
        else:
            ax.text(0.5, 0.5, f"No data for feature stage: {stage_name}", 
                   horizontalalignment='center', verticalalignment='center')
        plot_idx += 1
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_augmentation_comparison(original_audio: np.ndarray,
                               augmented_audio: np.ndarray,
                               original_feature: np.ndarray,
                               augmented_feature: np.ndarray,
                               augmentation_name: str,
                               sr: int = 8000,
                               hop_length: int = 512,
                               fmin: float = 0.0,
                               fmax: float = 2000.0,
                               output_path: Optional[str] = None,
                               show_figure: bool = False,
                               feature_type: str = 'mel_spectrogram',
                               duration: Optional[float] = None) -> None:
    """
    Plot comparison between original and augmented audio and features.
    
    Parameters
    ----------
    original_audio : np.ndarray
        Original audio data
    augmented_audio : np.ndarray
        Augmented audio data
    original_feature : np.ndarray
        Original feature (e.g., spectrogram)
    augmented_feature : np.ndarray
        Augmented feature
    augmentation_name : str
        Name of the augmentation applied
    sr : int, optional
        Sample rate, by default 8000
    hop_length : int, optional
        Hop length for spectrogram calculation, by default 512
    fmin : float, optional
        Minimum frequency for spectrogram display, by default 0.0
    fmax : float, optional
        Maximum frequency for spectrogram display, by default 2000.0
    output_path : Optional[str], optional
        Path to save the plot, by default None
    show_figure : bool, optional
        Whether to display the figure, by default False
    feature_type : str, optional
        Type of feature being visualized, by default 'mel_spectrogram'
    duration : Optional[float], optional
        Duration in seconds to set on x-axis, by default None
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Augmentation: {augmentation_name}', fontsize=16)
    
    # Plot original and augmented audio waveforms
    ax1 = fig.add_subplot(4, 1, 1)
    librosa.display.waveshow(original_audio, sr=sr, ax=ax1)
    ax1.set_title('Original Audio')
    ax1.set_ylabel("Amplitude")
    if duration is not None:
        ax1.set_xlim([0, duration])
        ax1.set_xlabel(f"Time (s) - Duration: {duration:.1f}s")
    else:
        ax1.set_xlabel("")
    
    ax2 = fig.add_subplot(4, 1, 2)
    librosa.display.waveshow(augmented_audio, sr=sr, ax=ax2)
    ax2.set_title(f'Augmented Audio ({augmentation_name})')
    ax2.set_ylabel("Amplitude")
    if duration is not None:
        ax2.set_xlim([0, duration])
        ax2.set_xlabel(f"Time (s) - Duration: {duration:.1f}s")
    else:
        ax2.set_xlabel("Time (s)")
    
    # Plot original and augmented features
    ax3 = fig.add_subplot(4, 1, 3)
    
    # Convert to dB scale if not already
    if feature_type.lower() != 'mfcc' and np.max(original_feature) > 1.0:
        original_feature_db = librosa.amplitude_to_db(original_feature, ref=np.max)
    else:
        original_feature_db = original_feature
        
    im1 = librosa.display.specshow(original_feature_db, sr=sr, hop_length=hop_length,
                                x_axis='time', y_axis='hz',
                                fmin=fmin, fmax=fmax, ax=ax3)
    ax3.set_title(f'Original {feature_type}')
    if duration is not None:
        ax3.set_xlim([0, duration])
    fig.colorbar(im1, ax=ax3, format='%+2.0f dB')
    
    ax4 = fig.add_subplot(4, 1, 4)
    
    # Convert to dB scale if not already
    if feature_type.lower() != 'mfcc' and np.max(augmented_feature) > 1.0:
        augmented_feature_db = librosa.amplitude_to_db(augmented_feature, ref=np.max)
    else:
        augmented_feature_db = augmented_feature
        
    im2 = librosa.display.specshow(augmented_feature_db, sr=sr, hop_length=hop_length,
                                x_axis='time', y_axis='hz',
                                fmin=fmin, fmax=fmax, ax=ax4)
    ax4.set_title(f'Augmented {feature_type} ({augmentation_name})')
    if duration is not None:
        ax4.set_xlim([0, duration])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        
    if show_figure:
        plt.show()
    else:
        plt.close() 