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
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from PIL import Image
import cv2 # OpenCV for image processing, used by albumentations
import albumentations as A
from glob import glob

import sys
sys.path.append('/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/src')

from anemonefish_acoustics.utils.logger import get_logger

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
ANEMONEFISH_SPECS_PATH = os.path.join(BASE_DATA_PATH, 'anemonefish') # Spectrograms of anemonefish
NOISE_SPECS_PATH = os.path.join(BASE_DATA_PATH, 'noise')             # Spectrograms of noise

LOGS_DIR = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/logs/experiments/binary_classifier_spectrogram'
MODEL_SAVE_PATH = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/models/binary_classifier/'

# Image and Model Parameters
IMG_WIDTH = 256  # Assuming square spectrograms, adjust if needed
IMG_HEIGHT = 256 # Assuming square spectrograms, adjust if needed
IMG_CHANNELS = 3 # Typically RGB, even if spectrograms are grayscale, they are often loaded/processed as RGB
MODEL_INPUT_SIZE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50 # Start with a moderate number, can be adjusted
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1 # 10% of training data for validation
TEST_SPLIT = 0.1       # 10% of total data for final testing

# Labels
CLASS_NAMES = ['noise', 'anemonefish']
LABEL_MAP = {'noise': 0, 'anemonefish': 1}

logging.info("Configuration Loaded.")
logging.info(f"Anemonefish Spectrogram Path: {ANEMONEFISH_SPECS_PATH}")
logging.info(f"Noise Spectrogram Path: {NOISE_SPECS_PATH}")
logging.info(f"Model Input Size: {MODEL_INPUT_SIZE}")
logging.info(f"Batch Size: {BATCH_SIZE}")

# Check if spectrogram directories exist
if not os.path.isdir(ANEMONEFISH_SPECS_PATH):
    logging.warning(f"Anemonefish spectrogram directory not found: {ANEMONEFISH_SPECS_PATH}")
    logging.warning("Please ensure your anemonefish spectrograms are in the correct path.")
if not os.path.isdir(NOISE_SPECS_PATH):
    logging.warning(f"Noise spectrogram directory not found: {NOISE_SPECS_PATH}")
    logging.warning("Please ensure your noise spectrograms are in the correct path.")

# %% [markdown]
# ## 3. Load Data Paths and Labels
# 
# Here, we'll scan the specified directories for spectrogram images and assign labels based on their parent folder.
# - Images in `ANEMONEFISH_SPECS_PATH` will be labeled as 'anemonefish' (1).
# - Images in `NOISE_SPECS_PATH` will be labeled as 'noise' (0).

# %%
def load_filepaths_and_labels(base_path, class_name, label_map):
    """Loads image file paths and assigns labels."""
    filepaths = []
    labels = []
    class_dir = os.path.join(base_path, class_name)
    
    if not os.path.isdir(class_dir):
        logging.warning(f"Directory not found for class '{class_name}': {class_dir}")
        return filepaths, labels
        
    for filename in os.listdir(class_dir):
        if not filename.startswith('.') and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepaths.append(os.path.join(class_dir, filename))
            labels.append(label_map[class_name])
    logging.info(f"Found {len(filepaths)} images for class '{class_name}' in {class_dir}")
    return filepaths, labels

# Load anemonefish spectrograms
anemonefish_files, anemonefish_labels = load_filepaths_and_labels(BASE_DATA_PATH, 'anemonefish', LABEL_MAP)

# Load noise spectrograms
noise_files, noise_labels = load_filepaths_and_labels(BASE_DATA_PATH, 'noise', LABEL_MAP)

# Combine data
all_filepaths = anemonefish_files + noise_files
all_labels = anemonefish_labels + noise_labels

if not all_filepaths:
    logging.critical("No image files were found. Please check your ANEMONEFISH_SPECS_PATH and NOISE_SPECS_PATH in the configuration.")
    # You might want to stop execution here if no data is found.
    # For now, we'll proceed, but the generator and training will fail.
else:
    logging.info(f"Total images found: {len(all_filepaths)}")
    logging.info(f"Total labels: {len(all_labels)}")
    logging.info(f"Unique labels: {np.unique(all_labels)}")
    logging.info(f"Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")  # Shows counts per class

# Convert to numpy arrays
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
from sklearn.utils import class_weight
import numpy as np

# Calculate class weights
# This needs to be done only if train_labels are available and not empty
if 'train_labels' in globals() and len(train_labels) > 0:
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    # Keras expects class_weight as a dictionary
    class_weights_dict = {i : class_weights_array[i] for i in range(len(class_weights_array))}
    logging.info(f"Calculated class weights: {class_weights_dict}")
else:
    class_weights_dict = None
    logging.warning("Skipping class weight calculation as train_labels are not available or empty.")
    logging.warning("If training proceeds, it will be without class weights.")


# %% [markdown]
# ## 5. Data Augmentation and Preprocessing Pipeline
# 
# We'll use `albumentations` for preprocessing. For now, this will primarily involve resizing and normalization.
# We define separate pipelines for training (which could include augmentation later) and validation/testing (which only does necessary preprocessing).

# %%
# For binary classification, we don't need keypoint_params like in your example.

# Training Augmentations / Preprocessing
# Initially, this will just be resizing and normalization.
# We can add more augmentations like RandomBrightnessContrast, ShiftScaleRotate later if needed.
train_transform = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, interpolation=cv2.INTER_AREA),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0) # Typical ImageNet stats
    # ToTensorV2() # Albumentations can also convert to tensor, but Keras Sequence usually handles numpy
])

# Validation/Test Preprocessing (no random augmentations)
val_test_transform = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, interpolation=cv2.INTER_AREA),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0)
    # ToTensorV2()
])

logging.info("Augmentation/Preprocessing pipelines defined.")
logging.info(f"Images will be resized to: ({IMG_HEIGHT}, {IMG_WIDTH}) and normalized.")

# %% [markdown]
# ## 6. Data Generator (Keras Sequence)
# 
# We'll create a custom Keras `Sequence` to load and preprocess images on-the-fly. This is memory-efficient, especially for large datasets.
# The generator will take file paths and labels, load images, apply the defined transformations, and yield batches of (image, label) pairs.

# %%
class SpectrogramDataGenerator(Sequence):
    def __init__(self,
                 image_paths,
                 labels,
                 batch_size,
                 input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                 transform=None,
                 shuffle=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.input_size = input_size
        self.transform = transform
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs (paths) and corresponding labels
        batch_image_paths = self.image_paths[batch_indexes]
        batch_labels = self.labels[batch_indexes]

        # Generate data
        X = np.empty((self.batch_size, *self.input_size), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=np.int64) # For binary classification, labels are single integers

        for i, img_path in enumerate(batch_image_paths):
            try:
                # Load image using PIL (handles various formats, ensures 3 channels if needed)
                img = Image.open(img_path).convert('RGB') # Convert to RGB
                img_array = np.array(img)

                if self.transform:
                    augmented = self.transform(image=img_array)
                    img_array_processed = augmented['image']
                else:
                    # Basic resize if no albumentations transform (should not happen with our setup)
                    img_array_processed = cv2.resize(img_array, (self.input_size[1], self.input_size[0]))
                    img_array_processed = img_array_processed / 255.0 # Basic normalization if not using albumentations

                X[i,] = img_array_processed
                y[i] = batch_labels[i]
                
            except FileNotFoundError:
                logging.error(f"Image file not found at {img_path}. Skipping.")
                # Potentially fill with zeros or a placeholder, or skip and adjust batch size
                # For simplicity, we'll just have this sample missing if an error occurs.
                # A more robust solution would handle this more gracefully.
                X[i,] = np.zeros(self.input_size, dtype=np.float32)
                y[i] = 0 # Or some default label
            except Exception as e:
                logging.error(f"Error processing image {img_path}: {e}. Skipping.")
                X[i,] = np.zeros(self.input_size, dtype=np.float32)
                y[i] = 0


        # For binary_crossentropy, labels should be (batch_size,) and model output (batch_size, 1) with sigmoid
        # Or labels can be one-hot encoded (batch_size, num_classes) for categorical_crossentropy
        # Here, we are using simple integer labels for binary classification with from_logits=False or direct sigmoid.
        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

logging.info("SpectrogramDataGenerator class defined.")

# %% [markdown]
# ## 7. Create Data Generators
# 
# Instantiate the data generators for training, validation, and test sets.

# %%
if len(train_paths) > 0 :
    train_generator = SpectrogramDataGenerator(
        image_paths=train_paths,
        labels=train_labels,
        batch_size=BATCH_SIZE,
        input_size=MODEL_INPUT_SIZE,
        transform=train_transform,
        shuffle=True
    )
    logging.info(f"Train generator created with {len(train_generator)} batches.")
else:
    train_generator = None
    logging.warning("Train generator not created as there are no training paths.")

if len(val_paths) > 0:
    validation_generator = SpectrogramDataGenerator(
        image_paths=val_paths,
        labels=val_labels,
        batch_size=BATCH_SIZE, # Can use a different batch size for validation if desired
        input_size=MODEL_INPUT_SIZE,
        transform=val_test_transform,
        shuffle=False # No need to shuffle validation data
    )
    logging.info(f"Validation generator created with {len(validation_generator)} batches.")
else:
    validation_generator = None
    logging.warning("Validation generator not created as there are no validation paths.")

if len(test_paths) > 0:
    test_generator = SpectrogramDataGenerator(
        image_paths=test_paths,
        labels=test_labels,
        batch_size=1, # Typically batch size 1 for testing
        input_size=MODEL_INPUT_SIZE,
        transform=val_test_transform,
        shuffle=False # No need to shuffle test data
    )
    logging.info(f"Test generator created with {len(test_generator)} batches (samples).")
else:
    test_generator = None
    logging.warning("Test generator not created as there are no test paths.")

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
logging.info(model.summary())

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

# Check if generators are valid before starting training
if train_generator is None:
    logging.critical("Training generator is not available. Cannot start training.")
elif validation_generator is None:
    logging.warning("Validation generator is not available. Training will proceed without validation, which is not recommended.")
    # Optionally, you could decide to not train, or train with a subset of training data as validation.
    # For now, we'll allow training without validation if the user explicitly set it up this way.
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=callbacks_list, # Some callbacks might depend on validation data (e.g. ModelCheckpoint on val_loss)
        class_weight=class_weights_dict,
        verbose=1
    )
else:
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks_list,
        class_weight=class_weights_dict,
        verbose=1
    )
    logging.info("Training finished.")
    # Load the best weights saved by ModelCheckpoint (if EarlyStopping didn't already restore them)
    # model.load_weights(best_model_path) # Redundant if restore_best_weights=True in EarlyStopping

# %% [markdown]
# ## 11. Evaluate the Model
# 
# After training, we'll evaluate the model's performance on the unseen test set.
# We will:
# - Load the best weights saved during training (if `restore_best_weights=True` in `EarlyStopping`, this is already done).
# - Make predictions on the test set.
# - Calculate and display key metrics like accuracy, precision, recall, F1-score, and the confusion matrix.

# %%
best_model_path = "/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/models/binary_classifier/checkpoints_run_3/best_model.keras"

# %%
# If EarlyStopping with restore_best_weights=True was used, 
# the model already has the best weights. Otherwise, load them:
if os.path.exists(best_model_path):
    logging.info(f"Loading best model weights from: {best_model_path}")
    model.load_weights(best_model_path)
else:
    logging.warning("Best model checkpoint not found. Evaluating with current model weights.")

if test_generator is not None and len(test_generator) > 0:
    logging.info("Evaluating model on the test set...")
    
    # Get true labels from the test generator
    # Note: test_generator batch_size is 1, so test_labels are directly usable.
    # If batch_size was > 1, you'd need to iterate through the generator to collect all labels.
    y_true_test = test_labels 
    
    # Make predictions
    # The predict method of the model expects the data directly, 
    # and our test_generator yields (images, labels)
    # We need to collect all images from the test_generator first.
    
    num_test_samples = len(test_paths)
    X_test = np.empty((num_test_samples, *MODEL_INPUT_SIZE), dtype=np.float32)
    
    for i in range(num_test_samples): # test_generator has batch_size 1
        img_batch, _ = test_generator[i] # Get the i-th batch (which is a single image)
        X_test[i] = img_batch[0] # img_batch is (1, H, W, C), so take the first element

    y_pred_probs = model.predict(X_test)
    y_pred_test = (y_pred_probs > 0.5).astype(int).flatten() # Convert probabilities to binary predictions (0 or 1)

    # Calculate metrics
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_true_test, verbose=0)
    
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
    logging.warning("Test generator is not available or empty. Skipping evaluation.")

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