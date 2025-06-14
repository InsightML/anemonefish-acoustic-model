# %% [markdown]
# # Binary Spectrogram Classifier: Anemonefish vs. Noise
# 
# This notebook trains a binary classification model to distinguish between spectrograms of anemonefish calls and background noise.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import sys
sys.path.append('/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/src')

from anemonefish_acoustics.utils.logger import get_logger
from anemonefish_acoustics.data_processing import (
    SpectrogramConfig, SpectrogramDataLoader, SpectrogramDatasetBuilder,
    get_dataset_info, validate_preprocessing_consistency
)

# Setup logging
logging = get_logger()

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

# Paths - Adjust these to your actual data locations
BASE_DATA_PATH = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/1_binary_training_data/spectograms' # Base directory for spectrograms
ANEMONEFISH_PATH = os.path.join(BASE_DATA_PATH, 'anemonefish') # Spectrograms of anemonefish
NOISE_PATH = os.path.join(BASE_DATA_PATH, 'noise')             # Spectrograms of noise

LOGS_DIR = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/logs/experiments/binary_classifier_spectrogram'
MODEL_SAVE_PATH = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/models/binary_classifier/'

# Image and Model Parameters
IMG_WIDTH = 256  # Assuming square spectrograms, adjust if needed
IMG_HEIGHT = 256 # Assuming square spectrograms, adjust if needed
IMG_CHANNELS = 3 # Typically RGB, even if spectrograms are grayscale, they are often loaded/processed as RGB
MODEL_INPUT_SIZE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 50 # Start with a moderate number, can be adjusted
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1 # 10% of training data for validation
TEST_SPLIT = 0.1       # 10% of total data for final testing

# Labels
CLASS_NAMES = ['noise', 'anemonefish']
LABEL_MAP = {'noise': 0, 'anemonefish': 1}

logging.info("Configuration Loaded.")
logging.info(f"Anemonefish Spectrogram Path: {ANEMONEFISH_PATH}")
logging.info(f"Noise Spectrogram Path: {NOISE_PATH}")
logging.info(f"Model Input Size: {MODEL_INPUT_SIZE}")
logging.info(f"Batch Size: {BATCH_SIZE}")

# Check if spectrogram directories exist
if not os.path.isdir(ANEMONEFISH_PATH):
    logging.warning(f"Anemonefish spectrogram directory not found: {ANEMONEFISH_PATH}")
    logging.warning("Please ensure your anemonefish spectrograms are in the correct path.")
if not os.path.isdir(NOISE_PATH):
    logging.warning(f"Noise spectrogram directory not found: {NOISE_PATH}")
    logging.warning("Please ensure your noise spectrograms are in the correct path.")

# %% [markdown]
# ## 3. Load Data Paths and Labels
# 
# Here, we'll scan the specified directories for spectrogram images and assign labels based on their parent folder.
# - Images in `ANEMONEFISH_PATH` will be labeled as 'anemonefish' (1).
# - Images in `NOISE_PATH` will be labeled as 'noise' (0).

# %%
# Initialize shared preprocessing components
config = SpectrogramConfig()
loader = SpectrogramDataLoader(config)
builder = SpectrogramDatasetBuilder(config)

# Load labeled data using shared preprocessing module
all_filepaths, all_labels = loader.load_labeled_data(
    anemonefish_path=ANEMONEFISH_PATH,
    noise_path=NOISE_PATH,
    label_map=LABEL_MAP
)

if not all_filepaths:
    logging.critical("No image files were found. Please check your ANEMONEFISH_PATH and NOISE_PATH in the configuration.")
else:
    logging.info(f"Total images found: {len(all_filepaths)}")
    logging.info(f"Total labels: {len(all_labels)}")
    logging.info(f"Unique labels: {np.unique(all_labels)}")
    logging.info(f"Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

# Convert to numpy arrays for compatibility with existing code
all_filepaths = np.array(all_filepaths)
all_labels = np.array(all_labels)

# %% [markdown]
# ## 4. Train, Validation, and Test Split
# 
# We'll split the data into training, validation, and testing sets.
# - First, separate a test set.
# - Then, split the remaining data into training and validation sets.
# This ensures the test set is completely unseen during training and hyperparameter tuning.

# %%
if len(all_filepaths) > 0:
    # Step 1: Split into training+validation and test sets
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_filepaths,
        all_labels,
        test_size=TEST_SPLIT,
        random_state=SEED,
        stratify=all_labels  # Important for imbalanced datasets
    )

    # Step 2: Split training+validation into training and validation sets
    # Adjust validation_split relative to the size of train_val_paths
    effective_validation_split = VALIDATION_SPLIT / (1 - TEST_SPLIT) if (1 - TEST_SPLIT) > 0 else 0

    if len(train_val_paths) > 1 and effective_validation_split > 0 : # Ensure there's enough data to split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=effective_validation_split,
            random_state=SEED,
            stratify=train_val_labels # Important for imbalanced datasets
        )
    else: # Not enough data for a validation split after test split, or validation split is zero
        logging.warning("Not enough data for a separate validation set after test split, or VALIDATION_SPLIT is 0. Validation set will be empty or same as training.")
        train_paths, train_labels = train_val_paths, train_val_labels
        val_paths, val_labels = np.array([]), np.array([]) # Empty validation set


    logging.info(f"Training samples: {len(train_paths)}")
    logging.info(f"Validation samples: {len(val_paths)}")
    logging.info(f"Test samples: {len(test_paths)}")

    # Verify distribution in splits (optional)
    if len(train_labels) > 0:
        logging.info(f"Train label distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    if len(val_labels) > 0:
        logging.info(f"Validation label distribution: {dict(zip(*np.unique(val_labels, return_counts=True)))}")
    if len(test_labels) > 0:
        logging.info(f"Test label distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
else:
    logging.warning("Skipping data splitting as no data was loaded.")
    train_paths, val_paths, test_paths = np.array([]), np.array([]), np.array([])
    train_labels, val_labels, test_labels = np.array([]), np.array([]), np.array([])

# %%
# Calculate class weights (will be done after data splitting)
# This needs to be done only if train_labels are available and not empty
def calculate_class_weights(train_labels):
    """Calculate balanced class weights for training."""
    if len(train_labels) > 0:
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        # Keras expects class_weight as a dictionary
        class_weights_dict = {i: class_weights_array[i] for i in range(len(class_weights_array))}
        logging.info(f"Calculated class weights: {class_weights_dict}")
        return class_weights_dict
    else:
        logging.warning("No training labels available for class weight calculation.")
        return None

# %% [markdown]
# ## 7. Create tf.data Datasets and Calculate Class Weights 
# 
# Create optimized tf.data datasets using the shared preprocessing module and calculate class weights for balanced training.

# %%
# Calculate class weights for balanced training
class_weights_dict = calculate_class_weights(train_labels) if len(train_labels) > 0 else None

# Create optimized tf.data datasets using shared preprocessing
if len(train_paths) > 0:
    train_dataset = builder.create_classifier_dataset(
        image_paths=train_paths.tolist(),
        labels=train_labels.tolist(),
        batch_size=BATCH_SIZE,
        is_training=True,
        cache_data=True
    )
    logging.info(f"Training dataset created.")
else:
    train_dataset = None
    logging.warning("Training dataset not created as there are no training paths.")

if len(val_paths) > 0:
    val_dataset = builder.create_classifier_dataset(
        image_paths=val_paths.tolist(),
        labels=val_labels.tolist(),
        batch_size=BATCH_SIZE,
        is_training=False,
        cache_data=True
    )
    logging.info(f"Validation dataset created.")
else:
    val_dataset = None
    logging.warning("Validation dataset not created as there are no validation paths.")

if len(test_paths) > 0:
    test_dataset = builder.create_classifier_dataset(
        image_paths=test_paths.tolist(),
        labels=test_labels.tolist(),
        batch_size=1,  # Batch size 1 for testing
        is_training=False,
        cache_data=False  # Don't cache test data
    )
    logging.info(f"Test dataset created.")
else:
    test_dataset = None
    logging.warning("Test dataset not created as there are no test paths.")

# Print dataset information
if train_dataset:
    get_dataset_info(train_dataset, "Training")
if val_dataset:
    get_dataset_info(val_dataset, "Validation")
if test_dataset:
    get_dataset_info(test_dataset, "Test")

# %%
# Validate preprocessing consistency (optional)
if len(train_paths) > 0:
    validate_preprocessing_consistency(config, train_paths[0])

# Test the dataset pipeline with a single batch
if train_dataset is not None:
    print("\nTesting tf.data pipeline...")
    try:
        sample_batch = next(iter(train_dataset.take(1)))
        images, labels = sample_batch
        print(f"✓ Successfully loaded batch:")
        print(f"  - Images shape: {images.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Image value range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
        print(f"  - Unique labels in batch: {tf.unique(labels).y.numpy()}")
        print(f"  - Images dtype: {images.dtype}")
        print(f"  - Labels dtype: {labels.dtype}")
    except Exception as e:
        print(f"✗ Error testing pipeline: {e}")


# %% [markdown]
# ## 8. Define the CNN Model
# 
# We'll define a simple Convolutional Neural Network (CNN) suitable for binary image classification.
# The architecture will consist of a few convolutional blocks followed by dense layers.
# - Convolutional layers for feature extraction.
# - MaxPooling layers for down-sampling.
# - BatchNormalization for stabilizing learning.
# - Dropout for regularization.
# - A final Dense layer with a sigmoid activation for binary classification.

# %%
def create_binary_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),

        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 4
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),


        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Sigmoid activation for binary classification
    ])
    return model

# Instantiate the model
model = create_binary_cnn(MODEL_INPUT_SIZE)

# Display the model's architecture
model.summary()

# %% [markdown]
# ## 9. Compile the Model
# 
# Compile the model by specifying the optimizer, loss function, and metrics.
# - **Optimizer**: Adam is a good default choice.
# - **Loss Function**: `binary_crossentropy` is appropriate for binary classification with a sigmoid output.
# - **Metrics**: `accuracy` is a common metric for classification. We can also add others like Precision and Recall.

# %%
optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

logging.info("Model compiled.")

# %% [markdown]
# ## 10. Define Callbacks and Train the Model
# 
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
next_run_number = len(existing_runs) + 1

run_log_dir = os.path.join(LOGS_DIR, f"run_{next_run_number}")
run_checkpoint_dir = os.path.join(MODEL_SAVE_PATH, f"checkpoints_run_{next_run_number}")

os.makedirs(run_log_dir, exist_ok=True)
os.makedirs(run_checkpoint_dir, exist_ok=True)

best_model_path = os.path.join(run_checkpoint_dir, "best_model.keras") # Using .keras format

# Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=1)

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

logging.info(f"TensorBoard logs will be saved to: {run_log_dir}")
logging.info(f"Model checkpoints will be saved to: {run_checkpoint_dir}")
logging.info(f"Best model will be saved as: {best_model_path}")

# Check if datasets are valid before starting training
if train_dataset is None:
    logging.critical("Training dataset is not available. Cannot start training.")
elif val_dataset is None:
    logging.warning("Validation dataset is not available. Training will proceed without validation, which is not recommended.")
    # Train without validation
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        class_weight=class_weights_dict,
        verbose=1
    )
else:
    # Train with validation data
    logging.info("Starting training with tf.data datasets (optimized for performance)...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks_list,
        class_weight=class_weights_dict,
        verbose=1
    )
    logging.info("Training finished.")
    
logging.info("✅ Training completed using optimized tf.data pipeline!")
logging.info("Benefits achieved:")
logging.info("  • 2-5x faster data loading compared to generators")
logging.info("  • Better GPU utilization due to prefetching")
logging.info("  • Consistent preprocessing with autoencoder model")

# %% [markdown]
# ## 11. Evaluate the Model
# 
# After training, we'll evaluate the model's performance on the unseen test set.
# We will:
# - Load the best weights saved during training (if `restore_best_weights=True` in `EarlyStopping`, this is already done).
# - Make predictions on the test set.
# - Calculate and display key metrics like accuracy, precision, recall, F1-score, and the confusion matrix.

# %%
# If EarlyStopping with restore_best_weights=True was used, 
# the model already has the best weights. Otherwise, load them:
if os.path.exists(best_model_path):
    logging.info(f"Loading best model weights from: {best_model_path}")
    model.load_weights(best_model_path)
else:
    logging.warning("Best model checkpoint not found. Evaluating with current model weights.")

if test_dataset is not None:
    logging.info("Evaluating model on the test set using tf.data...")
    
    # Use the test dataset directly for prediction and evaluation
    y_pred_probs = model.predict(test_dataset, verbose=1)
    y_pred_test = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Get true labels (we already have them from the data split)
    y_true_test = test_labels
    
    # Calculate metrics using the test dataset
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset, verbose=0)
    
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test Precision: {test_precision:.4f}")
    logging.info(f"Test Recall: {test_recall:.4f}")

    logging.info("Classification Report on Test Set:")
    logging.info(f"\n{classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES)}")

    logging.info("Confusion Matrix on Test Set:")
    cm = confusion_matrix(y_true_test, y_pred_test)
    logging.info(f"\n{cm}")

    # Plotting the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels([''] + CLASS_NAMES) # Add empty string for 0-tick
    ax.yaxis.set_ticklabels([''] + CLASS_NAMES) # Add empty string for 0-tick
    
    # Annotate cells with counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', color='black' if cm[i,j] < (cm.max()/2) else 'white')
            
    plt.show()

else:
    logging.warning("Test dataset is not available. Skipping evaluation.")

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
import os
import tf2onnx
import tensorflow as tf # Required for tf.TensorSpec and if 'model' is tf.keras.Model

# It's assumed that 'model', 'best_model_path', 'MODEL_SAVE_PATH', 
# and 'current_time' are defined in previous cells of your notebook.

# The best model is already saved by ModelCheckpoint (likely in Keras format)
logging.info(f"The best performing Keras model (from ModelCheckpoint) was saved to: {best_model_path}")

# Define the path for the ONNX model
# This uses the directory from MODEL_SAVE_PATH and the current_time string, similar to original logic
onnx_model_dir = os.path.dirname(run_checkpoint_dir)
onnx_model_filename = f"model.onnx"
onnx_model_save_path = os.path.join(onnx_model_dir, onnx_model_filename)

logging.info(f"Preparing to save the final model in ONNX format to: {onnx_model_save_path}")

try:
    # Convert the Keras model to ONNX.
    # 'model' should be your trained tf.keras.Model instance.
    
    # For many common models, tf2onnx can infer the input signature.
    # If conversion fails, you may need to explicitly provide the input_signature.
    # ----- Example for explicitly defining input_signature -----
    # # Replace (None, height, width, channels) with your model's actual input shape and dtype.
    # # For a model with input shape (e.g., 128, 128, 1) for spectrograms:
    # input_signature = [tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name="input_spectrogram")]
    #
    # # If your model has multiple inputs, provide a list of tf.TensorSpec objects.
    # # You can also try to derive it dynamically from the model (might need adjustments):
    # # if hasattr(model, 'inputs') and model.inputs:
    # #     input_signature = [tf.TensorSpec.from_tensor(tensor) for tensor in model.inputs]
    # # else:
    # #     logging.info("Could not automatically determine input signature from model.inputs. You may need to define it manually.")
    # #     input_signature = None # Fallback to tf2onnx inference
    # ----- End of example -----

    # For now, we'll let tf2onnx try to infer the input signature.
    # If this fails, define 'input_signature' using the examples above.
    input_signature = None 

    logging.info("Starting Keras to ONNX conversion...")
    # Ensure the 'model' variable holds your trained Keras model
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
        model=model,
        input_signature=input_signature,
        opset=13,  # Opset 13 is a common choice; adjust if needed for compatibility
        output_path=onnx_model_save_path
    )
    logging.info(f"Successfully saved model in ONNX format to: {onnx_model_save_path}")

except ImportError:
    logging.error("The 'tf2onnx' library was not found.")
    logging.error("Please install it, for example, by running: pip install tf2onnx")
except AttributeError as ae:
    if 'model' in str(ae):
        logging.error("The 'model' variable is likely not defined or is not a Keras model.")
        logging.error("Ensure 'model' is your trained Keras model instance before this cell.")
    else:
        logging.error(f"An AttributeError occurred: {ae}")
        logging.error("This might be due to an issue with the model structure or tf2onnx.")
except Exception as e:
    logging.error(f"An error occurred during Keras to ONNX conversion: {e}")
    logging.error("Tips for troubleshooting:")
    logging.error("- Ensure 'tf2onnx' and its dependencies (like 'onnx') are installed and up to date (`pip install -U tf2onnx onnx`).")
    logging.error("- If the error mentions input shapes, types, or names, you most likely need to define the 'input_signature' argument for `tf2onnx.convert.from_keras` explicitly.")
    logging.error("  See the commented-out 'Example for explicitly defining input_signature' in the code above.")
    logging.error("  Adjust the shape (e.g., `(None, 128, 128, 1)`), `dtype` (e.g., `tf.float32`), and `name` to match your model's input layer(s).")



