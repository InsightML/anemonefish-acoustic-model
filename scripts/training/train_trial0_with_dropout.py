# Training script for Trial 0 parameters + Dropout 0.5
# This script trains the model for 300 epochs for AWS deployment
# Based on Trial 0's best parameters from hyperparameter tuning

import os
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import yaml
from pathlib import Path
import sys
import datetime

sys.path.append('/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/src')

from anemonefish_acoustics.utils.logger import get_logger
from anemonefish_acoustics.utils.utils import pretty_path

# Setup logging
logging = get_logger(name='trial0_training', workspace_root='/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics')

logging.info("="*80)
logging.info("TRAINING RUN: Trial 0 Parameters + Dropout 0.5")
logging.info("="*80)

# Check for GPU
if tf.config.list_physical_devices('GPU'):
    logging.info("✓ TensorFlow is using the GPU!")
    for gpu in tf.config.list_physical_devices('GPU'):
        logging.info(f"  Name: {gpu.name}, Type: {gpu.device_type}")
else:
    logging.warning("⚠ TensorFlow is NOT using the GPU. Training will be on CPU.")

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
CLASSES = config['classes']
DATA_DIR = os.path.join(WORKSPACE_BASE_PATH, 'data', '2_training_datasets', DATASET_VERSION)

MODEL_INPUT_SIZE = [config['spectrogram']['height_pixels'], config['spectrogram']['width_pixels'], 3]

EPOCHS = config['epochs']
logging.info(f"Training for {EPOCHS} epochs")

LEARNING_RATE = config['learning_rate']
PROJECT_NAME = config['project_name']

# Create unique run directory
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"trial0_dropout_300ep_{current_time}"

MODEL_SAVE_PATH = os.path.join(WORKSPACE_BASE_PATH, 'models', PROJECT_NAME, run_name)
LOGS_DIR = os.path.join(WORKSPACE_BASE_PATH, 'logs', 'experiments', PROJECT_NAME, run_name)

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.info(f"TensorBoard logs: {pretty_path(LOGS_DIR)}")
logging.info(f"Model save path: {pretty_path(MODEL_SAVE_PATH, num_dirs=2)}")

# --- Load Data ---
logging.info("="*80)
logging.info("Loading training data...")
logging.info("="*80)

X_paths = []
Y_labels = []
Y_labels_for_weights = []

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

try:
    items = os.listdir(DATA_DIR)
except OSError as e:
    logging.error(f"Could not read directory {DATA_DIR}: {e}")
    items = []

directories = [item for item in items 
              if not item.startswith('.') and 
              os.path.isdir(os.path.join(DATA_DIR, item))]

for directory in directories:
    if directory in CLASSES:
        class_index = CLASSES.index(directory)
        class_dir_path = os.path.join(DATA_DIR, directory)
        
        one_hot_label = [0] * len(CLASSES)
        one_hot_label[class_index] = 1
        
        try:
            files = os.listdir(class_dir_path)
            image_files = [f for f in files 
                          if not f.startswith('.') and 
                          not f.startswith('_') and
                          os.path.splitext(f.lower())[1] in image_extensions]
            
            for image_file in image_files:
                full_path = os.path.join(class_dir_path, image_file)
                X_paths.append(full_path)
                Y_labels.append(one_hot_label)
                Y_labels_for_weights.append(class_index)
                
            logging.info(f"✓ Class '{directory}': {len(image_files)} images (index {class_index})")
            
        except OSError as e:
            logging.warning(f"Could not read class directory {class_dir_path}: {e}")
            continue

logging.info(f"Total files loaded: {len(X_paths)}")

# Calculate class weights
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(Y_labels_for_weights),
    y=np.array(Y_labels_for_weights))

class_weights_dict = {i: class_weights_array[i] for i in range(len(class_weights_array))}
logging.info(f"Class weights: {class_weights_dict}")

# Split data
X_train_paths, X_val_paths, Y_train, Y_val = train_test_split(
    X_paths, Y_labels, 
    test_size=config['validation_size'], 
    random_state=SEED, 
    stratify=Y_labels
)

logging.info(f"Training samples: {len(X_train_paths)}")
logging.info(f"Validation samples: {len(X_val_paths)}")

# --- Create TF Datasets ---
def load_and_preprocess_image(path, label):
    """Load and preprocess a single image."""
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [256, 256, 3])
    return image, label

def create_tf_dataset(X, Y):
    tf_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    tf_dataset = tf_dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return tf_dataset

train_dataset = (create_tf_dataset(X_train_paths, Y_train)
                   .shuffle(buffer_size=len(X_train_paths),
                           seed=SEED,
                           reshuffle_each_iteration=True)
                   .batch(config['batch_size'])
                   .cache()
                   .prefetch(tf.data.AUTOTUNE))

val_dataset = (create_tf_dataset(X_val_paths, Y_val)
                 .batch(config['batch_size'])
                 .cache()
                 .prefetch(tf.data.AUTOTUNE))

# Test pipeline
logging.info("Testing data pipeline...")
try:
    sample_batch = next(iter(train_dataset.take(1)))
    images, labels = sample_batch
    logging.info(f"✓ Batch shape: {images.shape}, Labels shape: {labels.shape}")
    logging.info(f"✓ Value range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
except Exception as e:
    logging.error(f"✗ Pipeline error: {e}")
    raise

# --- Build Model with Trial 0 Parameters + Dropout ---
logging.info("="*80)
logging.info("Building model with Trial 0 parameters + Dropout 0.5")
logging.info("="*80)

# TRIAL 0 PARAMETERS:
ACTIVATION = 'relu'
CNN_DEPTH = 5  # 5 conv layers per block
USE_DROPOUT = True  # ADDED
DROPOUT_RATE = 0.5  # ADDED
LEARNING_RATE_TRIAL0 = 0.0012124065560207727  # From Trial 0

NUM_CNN_BLOCKS = 4
FILTERS = [32, 64, 128, 256]

logging.info("Model Parameters:")
logging.info(f"  Activation: {ACTIVATION}")
logging.info(f"  CNN Depth: {CNN_DEPTH} layers per block")
logging.info(f"  CNN Blocks: {NUM_CNN_BLOCKS}")
logging.info(f"  Filters: {FILTERS}")
logging.info(f"  Use Dropout: {USE_DROPOUT}")
logging.info(f"  Dropout Rate: {DROPOUT_RATE}")
logging.info(f"  Learning Rate: {LEARNING_RATE_TRIAL0}")
logging.info(f"  Input Shape: {MODEL_INPUT_SIZE}")
logging.info(f"  Output Classes: {len(CLASSES)}")

model = Sequential()
model.add(Input(shape=MODEL_INPUT_SIZE))

for cnn_block in range(NUM_CNN_BLOCKS):
    for i in range(CNN_DEPTH):
        model.add(Conv2D(FILTERS[cnn_block], (3, 3), padding='same', activation=ACTIVATION))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation=ACTIVATION))
model.add(BatchNormalization())

if USE_DROPOUT:
    model.add(Dropout(DROPOUT_RATE))
    logging.info(f"✓ Dropout layer added with rate: {DROPOUT_RATE}")

model.add(Dense(len(CLASSES), activation='softmax'))

# Compile model
optimizer = Adam(learning_rate=LEARNING_RATE_TRIAL0)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary(print_fn=logging.info)

# --- Setup Callbacks ---
best_model_path = os.path.join(MODEL_SAVE_PATH, "best_model.keras")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=LOGS_DIR, 
    histogram_freq=1
)

model_checkpoint_callback = ModelCheckpoint(
    filepath=best_model_path,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=100,
    verbose=1,
    restore_best_weights=True
)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks_list = [
    tensorboard_callback,
    model_checkpoint_callback,
    early_stopping_callback,
    reduce_lr_callback
]

logging.info(f"Best model will be saved to: {best_model_path}")

# --- Train Model ---
logging.info("="*80)
logging.info(f"Starting training for {EPOCHS} epochs...")
logging.info("="*80)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks_list,
    class_weight=class_weights_dict
)

logging.info("="*80)
logging.info("Training completed!")
logging.info("="*80)

# --- Save Model and Artifacts ---
logging.info("="*80)
logging.info("Saving model and training artifacts...")
logging.info("="*80)

# Save final model in Keras format
final_model_path = os.path.join(MODEL_SAVE_PATH, 'final_model.keras')
model.save(final_model_path)
logging.info(f"✓ Final model saved: {final_model_path}")

# Save weights
weights_path = os.path.join(MODEL_SAVE_PATH, 'weights.weights.h5')
model.save_weights(weights_path)
logging.info(f"✓ Weights saved: {weights_path}")

# --- Save Model in SageMaker Deployment Format ---
logging.info("-"*80)
logging.info("Saving model for AWS SageMaker deployment...")
logging.info("-"*80)

# Create SageMaker deployment directory structure
sagemaker_deployment_dir = os.path.join(MODEL_SAVE_PATH, 'sagemaker_deployment')
savedmodel_dir = os.path.join(sagemaker_deployment_dir, '1')
os.makedirs(savedmodel_dir, exist_ok=True)

try:
    # Build model completely before saving
    logging.info("Building model with dummy input for SavedModel export...")
    dummy_input = tf.random.normal((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[2]), dtype=tf.float32)
    _ = model(dummy_input, training=False)
    
    # Create serving function with explicit signature
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[2]], dtype=tf.float32)])
    def serving_function(input_tensor):
        """Clean serving function for TensorFlow Serving"""
        return model(input_tensor, training=False)
    
    # Test serving function
    logging.info("Testing serving function...")
    test_output = serving_function(dummy_input)
    logging.info(f"✓ Serving function output shape: {test_output.shape}")
    
    # Save as SavedModel format for TensorFlow Serving
    logging.info(f"Saving SavedModel to: {savedmodel_dir}")
    tf.saved_model.save(
        model,
        savedmodel_dir,
        signatures={
            'serving_default': serving_function,
            'predict': serving_function
        }
    )
    logging.info(f"✓ SavedModel saved successfully")
    
    # Verify the SavedModel
    logging.info("Verifying SavedModel...")
    loaded_model = tf.saved_model.load(savedmodel_dir)
    logging.info(f"✓ Available signatures: {list(loaded_model.signatures.keys())}")
    
    # Test inference
    serving_fn = loaded_model.signatures['serving_default']
    test_result = serving_fn(dummy_input)
    
    # Extract output tensor
    if isinstance(test_result, dict):
        output_key = list(test_result.keys())[0]
        output_tensor = test_result[output_key]
    else:
        output_tensor = test_result
    
    logging.info(f"✓ Verification successful! Output shape: {output_tensor.shape}")
    logging.info(f"✓ Output dtype: {output_tensor.dtype}")
    
    # Check for NaN or Inf values
    has_nan = tf.reduce_any(tf.math.is_nan(output_tensor))
    has_inf = tf.reduce_any(tf.math.is_inf(output_tensor))
    
    if has_nan or has_inf:
        logging.warning(f"⚠ Model output contains NaN: {has_nan}, Inf: {has_inf}")
    else:
        logging.info("✓ Model output is clean (no NaN or Inf values)")
    
    logging.info("-"*80)
    logging.info("📦 SageMaker Deployment Package Ready!")
    logging.info(f"SavedModel location: {savedmodel_dir}")
    logging.info("Next steps:")
    logging.info("  1. Add inference.py and requirements.txt to sagemaker_deployment/code/")
    logging.info("  2. Create model.tar.gz:")
    logging.info(f"     cd {sagemaker_deployment_dir}")
    logging.info("     tar -czf model.tar.gz 1/ code/")
    logging.info("  3. Upload model.tar.gz to S3 for SageMaker deployment")
    logging.info("-"*80)
    
except Exception as e:
    logging.error(f"❌ Failed to save SavedModel: {e}")
    logging.error("The Keras model was saved successfully, but SavedModel export failed.")
    logging.error("You can try exporting manually later if needed.")

# Save training history
history_path = os.path.join(MODEL_SAVE_PATH, 'training_history.npy')
np.save(history_path, history.history)
logging.info(f"✓ Training history saved: {history_path}")

# Save configuration
model_config = {
    'run_name': run_name,
    'timestamp': current_time,
    'trial_source': 'Trial 0 with modifications',
    'input_shape': MODEL_INPUT_SIZE,
    'num_classes': len(CLASSES),
    'classes': CLASSES,
    'hyperparameters': {
        'activation': ACTIVATION,
        'cnn_depth': CNN_DEPTH,
        'cnn_blocks': NUM_CNN_BLOCKS,
        'filters': FILTERS,
        'use_dropout': USE_DROPOUT,
        'dropout_rate': DROPOUT_RATE,
        'learning_rate': LEARNING_RATE_TRIAL0,
        'batch_size': config['batch_size'],
        'epochs_configured': EPOCHS,
    },
    'training_results': {
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
    },
    'class_weights': {str(k): float(v) for k, v in class_weights_dict.items()},
    'paths': {
        'model_save_path': MODEL_SAVE_PATH,
        'best_model_keras': best_model_path,
        'final_model_keras': final_model_path,
        'weights': weights_path,
        'sagemaker_deployment': sagemaker_deployment_dir,
        'savedmodel': savedmodel_dir,
        'logs_dir': LOGS_DIR,
    }
}

config_path = os.path.join(MODEL_SAVE_PATH, 'model_config.yaml')
with open(config_path, 'w') as f:
    yaml.dump(model_config, f, default_flow_style=False)
logging.info(f"✓ Model config saved: {config_path}")

# --- Plot Training History ---
logging.info("Generating training history plots...")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(14, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(MODEL_SAVE_PATH, 'training_history.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
logging.info(f"✓ Training plot saved: {plot_path}")

# --- Final Summary ---
logging.info("="*80)
logging.info("TRAINING SUMMARY")
logging.info("="*80)
logging.info(f"Run Name: {run_name}")
logging.info(f"Epochs Trained: {len(history.history['loss'])}")
logging.info(f"Best Val Loss: {min(history.history['val_loss']):.4f}")
logging.info(f"Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
logging.info(f"Final Train Loss: {history.history['loss'][-1]:.4f}")
logging.info(f"Final Train Accuracy: {history.history['accuracy'][-1]:.4f}")
logging.info(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}")
logging.info(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
logging.info("="*80)
logging.info("✅ ALL DONE! Model ready for AWS deployment.")
logging.info("="*80)
logging.info(f"Model files location: {MODEL_SAVE_PATH}")
logging.info(f"Best model: {best_model_path}")
logging.info("="*80)

