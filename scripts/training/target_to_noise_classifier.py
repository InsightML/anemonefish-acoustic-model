# %% [markdown]
# # Train Binary Spectrogram Classifier: Anemonefish vs. Noise
# 
# This notebook trains a binary classification model to distinguish between spectrograms of anemonefish calls and background noise. 
# 
# Training data consists of two directories, one is spectograms of anemonefish, and the other is noise.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import tempfile
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
import keras
import keras_tuner
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import yaml
from pathlib import Path
import sys
sys.path.append('/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/src')

from anemonefish_acoustics.utils.logger import get_logger
from anemonefish_acoustics.utils.utils import pretty_path
from anemonefish_acoustics.models.hypermodels import TargetToNoiseHyperModel
from anemonefish_acoustics.data.preprocessing import preprocess_audio_for_training

# Setup logging - always use get_logger to ensure proper handlers
logging = get_logger(name='binary_classifier', workspace_root='/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics')

# %%
# Check for GPU
if tf.config.list_physical_devices('GPU'):
    logging.info("TensorFlow is using the GPU!")
    # You can print more details if needed
    for gpu in tf.config.list_physical_devices('GPU'):
        logging.info(f"Name: {gpu.name}, Type: {gpu.device_type}")
else:
    logging.warning("TensorFlow is NOT using the GPU. Training will be on CPU.")

# %% [markdown]
# ## 2. Configuration


# %%
# --- Configuration ---
CONFIG_PATH = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/2_training_datasets/v2_biological/preprocessing_config_v2_biological.yaml'

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load configuration
logging.info(f"Loading configuration from: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Extract configuration values
WORKSPACE_BASE_PATH = Path(config['workspace_base_path'])
DATASET_VERSION = config['dataset_version']
DATA_SITE = config['raw_data_site']
CLASSES = config['classes']
ANNOTATION_VERSION = config['annotation_version']
RAW_DATA_DIR = os.path.join(WORKSPACE_BASE_PATH, 'data', '1_raw', DATA_SITE)
DATA_DIR = os.path.join(WORKSPACE_BASE_PATH, 'data', '2_training_datasets', DATASET_VERSION)

ANNOTATION_DIR = os.path.join(RAW_DATA_DIR, ANNOTATION_VERSION)
AUDIO_DIR = os.path.join(RAW_DATA_DIR, 'audio')

MODEL_INPUT_SIZE = [config['spectrogram']['height_pixels'], config['spectrogram']['width_pixels'], 1]
EPOCHS = config['epochs']
TUNER_EPOCHS = config['tuner_epochs']
MAX_TRIALS = config['max_trials']
EXECUTIONS_PER_TRIAL = config['executions_per_trial']
MODEL_SAVE_PATH = os.path.join(WORKSPACE_BASE_PATH, config['model_save_path'], config['project_name'])
LEARNING_RATE = config['learning_rate']
LOGS_DIR = os.path.join(WORKSPACE_BASE_PATH, config['logs_dir'], config['project_name'])
TUNER_LOGS_DIR = os.path.join(WORKSPACE_BASE_PATH, config['tuner_logs_dir'])
PROJECT_NAME = config['project_name']
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TUNER_LOGS_DIR, exist_ok=True)

# Preprocessing parameters
WINDOW_SIZE_SECONDS = config['audio_processing']['window_size_seconds']
SLIDE_SIZE_SECONDS = config['audio_processing']['slide_size_seconds']
N_FFT = config['spectrogram']['n_fft']
HOP_LENGTH = config['spectrogram']['hop_length']
SR_TARGET = config['spectrogram']['sr_target']

# Noise padding parameters
NOISE_PADDING_CONFIG = {
    "min_duration_s": config['noise_padding']['min_duration_seconds'],
    "max_duration_s": config['noise_padding']['max_duration_seconds'],
    "padding_ratio": config['noise_padding']['padding_ratio']
}

# Logs
logging.info(f"TensorBoard logs will be saved to: {pretty_path(LOGS_DIR)}")
logging.info(f"Tuner logs will be saved to: {pretty_path(TUNER_LOGS_DIR)}")
logging.info(f"Model checkpoints, config, and training results will be saved to: {pretty_path(MODEL_SAVE_PATH, num_dirs=2)}")


# %% [markdown]
# ## 3. Train Val split & Class weight
# 
# Here, we'll scan the specified directories for spectrogram images and assign labels based on their parent folder.
# - Images in `ANEMONEFISH_PATH` will be labeled as 'anemonefish' (1).
# - Images in `NOISE_PATH` will be labeled as 'noise' (0).
# ---
# 
# Plan:
# 1. using CLASSES and DATA_DIR identify spectogram directory and list image paths for each class
# 2. map X and Y dataset. Load X by loading all the images into an array. train_test_split

# %% [markdown]
# ### 3b: Raw spectograms

# %%
X, y, class_mappings = preprocess_audio_for_training(
    audio_dir=AUDIO_DIR,
    annotations_dir=ANNOTATION_DIR,
    window_duration_s=WINDOW_SIZE_SECONDS,
    slide_duration_s=SLIDE_SIZE_SECONDS,
    sr_target=SR_TARGET,
    n_fft=N_FFT,
    hop_length=None,
    logger=logging,
    classes=CLASSES,
    min_segment_len_seconds=0.1,
    noise_padding_params=NOISE_PADDING_CONFIG
)

# %%
# Convert one-hot encoded labels back to class indices
y_class_indices = np.argmax(y, axis=1)  # Converts [0,1,0] -> 1, [1,0,0] -> 0, etc.

# Calculate class weights for balanced training 
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_class_indices),
    y=y_class_indices)

# Convert array to dictionary for tf training
class_weights_dict = {i: class_weights_array[i] for i in range(len(class_weights_array))}

# Log the class weights
print("Class weights:")
for class_idx, weight in class_weights_dict.items():
    class_name = class_mappings[class_idx]
    print(f"  {class_name} (index {class_idx}): {weight:.4f}")

# %%
# Split X and Y into train, val, test
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=config['validation_size'], random_state=config['seed'], stratify=y)

logging.info(f"X_train: {X_train.shape}")
logging.info(f"X_val: {X_val.shape}")

# %% [markdown]
# ## 4. Create tf.data Datasets
# 
# Create optimized tf.data datasets using the shared preprocessing module and calculate class weights for balanced training.

# %%
train_dataset = (tf.data.Dataset.from_tensor_slices((X_train, Y_train))
                   .shuffle(buffer_size=len(X_train),
                           seed=config['seed'],
                           reshuffle_each_iteration=True)
                   .batch(config['batch_size'])
                   .cache()
                   .prefetch(tf.data.AUTOTUNE))


val_dataset = (tf.data.Dataset.from_tensor_slices((X_val, Y_val))
                 .batch(config['batch_size'])
                 .cache()
                 .prefetch(tf.data.AUTOTUNE))

# %%
# Test the dataset pipeline with a single batch
if train_dataset is not None:
    logging.info("Testing tf.data pipeline...")
    try:
        sample_batch = next(iter(train_dataset.take(1)))
        spectrograms, labels = sample_batch
        logging.info("✓ Successfully loaded batch:")
        logging.info(f"  - X shape: {spectrograms.shape}")
        logging.info(f"  - Y shape: {labels.shape}")
        logging.info(f"  - X value range: [{tf.reduce_min(spectrograms):.3f}, {tf.reduce_max(spectrograms):.3f}]")
        logging.info(f"  - X dtype: {spectrograms.dtype}")
        logging.info(f"  - Y dtype: {labels.dtype}")
    except Exception as e:
        logging.error(f"✗ Error testing pipeline: {e}")


# %% [markdown]
# ## 5. Define callbacks
# We'll use several Keras callbacks during training:
# - `ModelCheckpoint`: To save the best model based on validation loss.
# - `EarlyStopping`: To stop training if the validation loss doesn't improve for a certain number of epochs.
# - `ReduceLROnPlateau`: To reduce the learning rate if validation loss plateaus.
# - `TensorBoard`: To log training metrics and graphs for visualization with TensorBoard.

# %%
import datetime

# Create a unique directory for this training run's logs and checkpoints
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Get number of existing runs
existing_runs = [d for d in os.listdir(LOGS_DIR) if d.startswith('run_')]
logging.info(f"Existing runs: {existing_runs}")
next_run_number = len(existing_runs) + 1

run_log_dir = os.path.join(LOGS_DIR, f"run_{next_run_number}")
run_checkpoint_dir = os.path.join(MODEL_SAVE_PATH, f"run_{next_run_number}")

os.makedirs(run_log_dir, exist_ok=True)
os.makedirs(run_checkpoint_dir, exist_ok=True)

best_model_path = os.path.join(run_checkpoint_dir, "best_model.keras") # Using .keras format

# Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=0)

model_checkpoint_callback = ModelCheckpoint(
    filepath=best_model_path,
    save_best_only=True,
    monitor='val_loss', # Save the model with the best validation loss
    mode='min',         # The lower the validation loss, the better
    verbose=1
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=50, # Number of epochs with no improvement after which training will be stopped
    verbose=1,
    restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity.
)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2, # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=5,  # Number of epochs with no improvement after which learning rate will be reduced.
    min_lr=1e-6, # Lower bound on the learning rate.
    verbose=1
)
callbacks_list = [
    tensorboard_callback,
    model_checkpoint_callback,
    early_stopping_callback,
    reduce_lr_callback
]
callbacks_list_tuner = [
    tensorboard_callback,
    reduce_lr_callback
]

logging.info(f"TensorBoard logs will be saved to: {run_log_dir}")
logging.info(f"Model checkpoints will be saved to: {run_checkpoint_dir}")
logging.info(f"Best model will be saved as: {best_model_path}")

# %% [markdown]
# ## 6 train and tune model

# %% [markdown]
# ### 6.1 Start the search (tuner)

# %%
# Instantiate the hypermodel
hypermodel = TargetToNoiseHyperModel(input_shape=MODEL_INPUT_SIZE, num_classes=len(CLASSES))
logging.info(f"Hypermodel input shape: {MODEL_INPUT_SIZE},  X shape: {spectrograms.shape}")

# %%
# Instantiate the tuner
tuner = keras_tuner.RandomSearch(
    hypermodel=hypermodel,
    objective='val_loss',
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTIONS_PER_TRIAL,
    directory=TUNER_LOGS_DIR,
    project_name=PROJECT_NAME
)
tuner.search_space_summary(extended=True)

# %%
try:
    tuner.search(train_dataset, epochs=TUNER_EPOCHS, validation_data=val_dataset, callbacks=callbacks_list_tuner, class_weight=class_weights_dict)
except Exception as e:
    logging.error(f"Error during tuning.search: {e}")

# %% [markdown]
# ### 6.2 query the results

# %%
models = tuner.get_best_models(num_models=2)
model = models[0]
model.summary()
tuner.results_summary()

# %% [markdown]
# ### 6.3 retrain the model

# %%
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# Use the hypermodel instance to build the model with best hyperparameters
model = hypermodel.build(best_hps)

# Train with the full epoch count
history = model.fit(
    train_dataset, 
    epochs=EPOCHS, 
    validation_data=val_dataset, 
    callbacks=callbacks_list,
    class_weight=class_weights_dict
)

# %% [markdown]
# ## 7 Save model

# %%
# Save the trained model
logging.info("Saving the trained model...")

# Create model directory if it doesn't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Save the model in Keras format (.keras)
model_save_file = os.path.join(MODEL_SAVE_PATH, 'model.keras')
model.save(model_save_file)
logging.info(f"Model saved to: {model_save_file}")

# Also save model weights separately (optional)
weights_save_file = os.path.join(MODEL_SAVE_PATH, 'weights.weights.h5')
model.save_weights(weights_save_file)
logging.info(f"Model weights saved to: {weights_save_file}")

# Save training history
history_save_file = os.path.join(MODEL_SAVE_PATH, 'training_history.npy')
np.save(history_save_file, history.history)
logging.info(f"Training history saved to: {history_save_file}")

# Save model configuration for reference
model_config = {
    'input_shape': MODEL_INPUT_SIZE,
    'num_classes': len(CLASSES),
    'classes': CLASSES,
    'epochs_trained': len(history.history['loss']),
    'final_train_loss': history.history['loss'][-1],
    'final_val_loss': history.history['val_loss'][-1],
    'final_train_accuracy': history.history['accuracy'][-1],
    'final_val_accuracy': history.history['val_accuracy'][-1]
}

config_save_file = os.path.join(MODEL_SAVE_PATH, 'model_config.yaml')
with open(config_save_file, 'w') as f:
    yaml.dump(model_config, f, default_flow_style=False)
logging.info(f"Model configuration saved to: {config_save_file}")

logging.info("✅ Model saving completed!")


# %% [markdown]
# ## 11. Evaluate the Model
# 
# After training, we'll evaluate the model's performance on the unseen test set.
# We will:
# - Load the best weights saved during training (if `restore_best_weights=True` in `EarlyStopping`, this is already done).
# - Make predictions on the test set.
# - Calculate and display key metrics like accuracy, precision, recall, F1-score, and the confusion matrix.

# %%
# # If EarlyStopping with restore_best_weights=True was used, 
# # the model already has the best weights. Otherwise, load them:
# if os.path.exists(best_model_path):
#     logging.info(f"Loading best model weights from: {best_model_path}")
#     model.load_weights(best_model_path)
# else:
#     logging.warning("Best model checkpoint not found. Evaluating with current model weights.")

# if test_dataset is not None:
#     logging.info("Evaluating model on the test set using tf.data...")
    
#     # Use the test dataset directly for prediction and evaluation
#     y_pred_probs = model.predict(test_dataset, verbose=1)
#     y_pred_test = (y_pred_probs > 0.5).astype(int).flatten()
    
#     # Get true labels (we already have them from the data split)
#     y_true_test = test_labels
    
#     # Calculate metrics using the test dataset
#     test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset, verbose=0)
    
#     logging.info(f"Test Loss: {test_loss:.4f}")
#     logging.info(f"Test Accuracy: {test_accuracy:.4f}")
#     logging.info(f"Test Precision: {test_precision:.4f}")
#     logging.info(f"Test Recall: {test_recall:.4f}")

#     logging.info("Classification Report on Test Set:")
#     logging.info(f"\n{classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES)}")

#     logging.info("Confusion Matrix on Test Set:")
#     cm = confusion_matrix(y_true_test, y_pred_test)
#     logging.info(f"\n{cm}")

#     # Plotting the confusion matrix
#     fig, ax = plt.subplots(figsize=(6, 6))
#     cax = ax.matshow(cm, cmap=plt.cm.Blues)
#     fig.colorbar(cax)
#     ax.set_xlabel('Predicted Labels')
#     ax.set_ylabel('True Labels')
#     ax.set_title('Confusion Matrix')
#     ax.xaxis.set_ticklabels([''] + CLASS_NAMES) # Add empty string for 0-tick
#     ax.yaxis.set_ticklabels([''] + CLASS_NAMES) # Add empty string for 0-tick
    
#     # Annotate cells with counts
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, str(cm[i, j]), va='center', ha='center', color='black' if cm[i,j] < (cm.max()/2) else 'white')
            
#     plt.show()

# else:
#     logging.warning("Test dataset is not available. Skipping evaluation.")

# %% [markdown]
# ## 12. Visualize Training History
# 
# Plotting the training and validation accuracy and loss helps to understand the model's learning process and identify potential issues like overfitting.

# %%
if 'history' in locals() and history is not None:
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy') # Use .get() in case validation was skipped
    loss = history.history['loss']
    val_loss = history.history.get('val_loss') # Use .get()

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    if val_acc:
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
else:
    logging.warning("Training history not available. Skipping visualization.")

# %% [markdown]
# ## 13. Save the Final Model (Optional)
# 
# The `ModelCheckpoint` callback already saved the best performing model during training.
# This step is to explicitly save the model's final state (which might be different from the best if `restore_best_weights=False` or if you continued training after early stopping).

# %%
# import os
# import tf2onnx
# import tensorflow as tf # Required for tf.TensorSpec and if 'model' is tf.keras.Model

# # It's assumed that 'model', 'best_model_path', 'MODEL_SAVE_PATH', 
# # and 'current_time' are defined in previous cells of your notebook.

# # The best model is already saved by ModelCheckpoint (likely in Keras format)
# logging.info(f"The best performing Keras model (from ModelCheckpoint) was saved to: {best_model_path}")

# # Define the path for the ONNX model
# # This uses the directory from MODEL_SAVE_PATH and the current_time string, similar to original logic
# onnx_model_dir = os.path.dirname(run_checkpoint_dir)
# onnx_model_filename = f"model.onnx"
# onnx_model_save_path = os.path.join(onnx_model_dir, onnx_model_filename)

# logging.info(f"Preparing to save the final model in ONNX format to: {onnx_model_save_path}")

# try:
#     # Convert the Keras model to ONNX.
#     # 'model' should be your trained tf.keras.Model instance.
    
#     # For many common models, tf2onnx can infer the input signature.
#     # If conversion fails, you may need to explicitly provide the input_signature.
#     # ----- Example for explicitly defining input_signature -----
#     # # Replace (None, height, width, channels) with your model's actual input shape and dtype.
#     # # For a model with input shape (e.g., 128, 128, 1) for spectrograms:
#     # input_signature = [tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name="input_spectrogram")]
#     #
#     # # If your model has multiple inputs, provide a list of tf.TensorSpec objects.
#     # # You can also try to derive it dynamically from the model (might need adjustments):
#     # # if hasattr(model, 'inputs') and model.inputs:
#     # #     input_signature = [tf.TensorSpec.from_tensor(tensor) for tensor in model.inputs]
#     # # else:
#     # #     logging.info("Could not automatically determine input signature from model.inputs. You may need to define it manually.")
#     # #     input_signature = None # Fallback to tf2onnx inference
#     # ----- End of example -----

#     # For now, we'll let tf2onnx try to infer the input signature.
#     # If this fails, define 'input_signature' using the examples above.
#     input_signature = None 

#     logging.info("Starting Keras to ONNX conversion...")
#     # Ensure the 'model' variable holds your trained Keras model
#     model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
#         model=model,
#         input_signature=input_signature,
#         opset=13,  # Opset 13 is a common choice; adjust if needed for compatibility
#         output_path=onnx_model_save_path
#     )
#     logging.info(f"Successfully saved model in ONNX format to: {onnx_model_save_path}")

# except ImportError:
#     logging.error("The 'tf2onnx' library was not found.")
#     logging.error("Please install it, for example, by running: pip install tf2onnx")
# except AttributeError as ae:
#     if 'model' in str(ae):
#         logging.error("The 'model' variable is likely not defined or is not a Keras model.")
#         logging.error("Ensure 'model' is your trained Keras model instance before this cell.")
#     else:
#         logging.error(f"An AttributeError occurred: {ae}")
#         logging.error("This might be due to an issue with the model structure or tf2onnx.")
# except Exception as e:
#     logging.error(f"An error occurred during Keras to ONNX conversion: {e}")
#     logging.error("Tips for troubleshooting:")
#     logging.error("- Ensure 'tf2onnx' and its dependencies (like 'onnx') are installed and up to date (`pip install -U tf2onnx onnx`).")
#     logging.error("- If the error mentions input shapes, types, or names, you most likely need to define the 'input_signature' argument for `tf2onnx.convert.from_keras` explicitly.")
#     logging.error("  See the commented-out 'Example for explicitly defining input_signature' in the code above.")
#     logging.error("  Adjust the shape (e.g., `(None, 128, 128, 1)`), `dtype` (e.g., `tf.float32`), and `name` to match your model's input layer(s).")



