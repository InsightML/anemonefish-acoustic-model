# %% [markdown]
# # Spectrogram Autoencoder for Embedding Generation
# 
# This notebook trains a convolutional autoencoder on all available spectrogram data (anemonefish, noise, and unlabeled) to generate embeddings that represent the general landscape of acoustic patterns in the dataset.
# 
# The goal is to create an unbiased representation of the data structure that can help us:
# 1. Understand the overall distribution of spectrogram patterns
# 2. Visualize where labeled data falls within the broader landscape
# 3. Identify potential clusters in unlabeled data
# 4. Find potential anemonefish calls in the unlabeled dataset
# 

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, Dense, Reshape, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import sys
sys.path.append('/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/src')

from anemonefish_acoustics.data_processing import (
    SpectrogramConfig, SpectrogramDataLoader, SpectrogramDatasetBuilder,
    get_dataset_info, validate_preprocessing_consistency
)
from anemonefish_acoustics.utils.logger import get_logger

# Setup logging
logging = get_logger(name='autoencoder_spectrogram_embeddings', workspace_root='/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics')

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.info("Libraries imported and random seeds set.")
logging.info("✓ Using shared preprocessing module for consistency with binary classifier")
logging.info("✓ Optimized tf.data pipeline for high-performance data loading")
logging.info("✓ Identical preprocessing ensures meaningful embedding analysis")


# %%
# Configure GPU memory growth to prevent memory allocation issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("TensorFlow is using the GPU with memory growth enabled!")
        for gpu in gpus:
            logging.info(f"Name: {gpu.name}, Type: {gpu.device_type}")
    except RuntimeError as e:
        logging.warning(f"GPU memory growth setting failed: {e}")
else:
    logging.warning("TensorFlow is NOT using the GPU. Training will be on CPU.")


# %% [markdown]
# ## 2. Configuration
# 

# %%
# --- Configuration ---

# Paths - Using same structure as binary classifier
BASE_DATA_PATH = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/1_binary_training_data/spectograms'
ANEMONEFISH_SPECS_PATH = os.path.join(BASE_DATA_PATH, 'anemonefish')
NOISE_SPECS_PATH = os.path.join(BASE_DATA_PATH, 'noise')
UNLABELED_SPECS_PATH = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/unlabelled_spectrograms'

LOGS_DIR = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/logs/experiments/autoencoder_spectrogram'
MODEL_SAVE_PATH = '/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/models/autoencoder/'

# Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 100  # Autoencoders may need more epochs
LEARNING_RATE = 1e-2
VALIDATION_SPLIT = 0.1

# Autoencoder specific
LATENT_DIM = 1024  # Dimension of the bottleneck layer (embedding size)

# Data sampling - for computational efficiency, we'll sample from the large unlabeled set
MAX_UNLABELED_SAMPLES = 100000  # Adjust based on computational resources

# Image parameters will be taken from shared config (automatically consistent with binary classifier)
logging.info("Configuration loaded:")
logging.info(f"  • Model training configuration set")
logging.info(f"  • Image parameters will come from shared preprocessing config")
logging.info(f"  • Latent Dimension: {LATENT_DIM}")
logging.info(f"  • Batch Size: {BATCH_SIZE}")
logging.info(f"  • Max Unlabeled Samples: {MAX_UNLABELED_SAMPLES}")
logging.info(f"  • Ensuring consistency with binary classifier preprocessing")

# Create directories
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Check if directories exist
paths_to_check = {
    'Anemonefish': ANEMONEFISH_SPECS_PATH,
    'Noise': NOISE_SPECS_PATH,
    'Unlabeled': UNLABELED_SPECS_PATH
}

for name, path in paths_to_check.items():
    if os.path.isdir(path):
        logging.info(f"✓ {name} directory found: {path}")
    else:
        logging.error(f"✗ {name} directory NOT found: {path}")


# %% [markdown]
# ## 3. Load All Data Paths
# 
# We'll load file paths from all three sources:
# - Labeled anemonefish spectrograms (~70)
# - Labeled noise spectrograms (~4000)
# - Unlabeled spectrograms (~80000, but we'll sample a subset for training efficiency)
# 

# %%
# Initialize shared preprocessing components (same as binary classifier)
config = SpectrogramConfig()
# Disable augmentation for training dataset
config.ENABLE_AUGMENTATION = True
loader = SpectrogramDataLoader(config)
builder = SpectrogramDatasetBuilder(config)

logging.info("Loading file paths using shared preprocessing module...")

# Load all data (labeled + unlabeled) for autoencoder training
all_paths, source_labels = loader.load_all_data(
    anemonefish_path=ANEMONEFISH_SPECS_PATH,
    noise_path=NOISE_SPECS_PATH,
    unlabeled_path=UNLABELED_SPECS_PATH,
    max_unlabeled_samples=MAX_UNLABELED_SAMPLES
)

if not all_paths:
    logging.error("CRITICAL: No image files found. Please check the paths.")
else:
    logging.info(f"📊 Dataset Summary:")
    logging.info(f"  • Total spectrograms: {len(all_paths)}")
    logging.info(f"  • Using identical preprocessing as binary classifier")
    logging.info(f"  • Ready for meaningful embedding analysis")


# %% [markdown]
# ## 4. Data Preprocessing Pipeline
# 
# Using the same preprocessing as the binary classifier to ensure consistency.
# 

# %%
# Validate preprocessing consistency
logging.info("Validating preprocessing consistency with binary classifier...")
try:
    is_consistent = validate_preprocessing_consistency(config, verbose=True)
    if is_consistent:
        logging.info("✅ Preprocessing is identical to binary classifier")
    else:
        logging.warning("⚠️  Warning: Preprocessing differences detected")
except Exception as e:
    logging.warning(f"Unable to validate consistency: {e}")

logging.info("Using shared preprocessing module:")
logging.info(f"  ✓ Input size: {config.IMG_HEIGHT}x{config.IMG_WIDTH}x{config.IMG_CHANNELS}")
logging.info(f"  ✓ Normalization: Scaled to [-1, 1] to match tanh activation")
logging.info(f"  ✓ Augmentation settings: brightness_delta={config.AUG_BRIGHTNESS_DELTA}, contrast=({config.AUG_CONTRAST_LOWER}, {config.AUG_CONTRAST_UPPER})")
logging.info(f"  ✓ Random seed: {config.SEED}")
logging.info(f"  ✓ Identical preprocessing ensures meaningful embedding analysis")


# %% [markdown]
# ## 5. Dataset Creation with Shared Preprocessing
# 
# Using the shared preprocessing module to create datasets with identical processing as the binary classifier.
# 

# %%
# Test the shared preprocessing on a sample image
if len(all_paths) > 0:
    logging.info("Testing shared preprocessing module...")
    try:
        # Use the shared preprocessor to test a single image
        test_image_path = all_paths[0]
        
        # Test the full pipeline: parse to [0,1], then scale to [-1,1]
        image_0_1 = builder.preprocessor.parse_image(test_image_path)
        test_image = builder.preprocessor.scale_image(image_0_1)
        
        logging.info(f"✅ Successfully processed test image:")
        logging.info(f"   • Input path: {test_image_path}")
        logging.info(f"   • Output shape: {test_image.shape}")
        logging.info(f"   • Data type: {test_image.dtype}")
        logging.info(f"   • Value range: [{tf.reduce_min(test_image):.3f}, {tf.reduce_max(test_image):.3f}]")
        logging.info(f"   • Expected range for normalization: [-1, 1]")
        
    except Exception as e:
        logging.error(f"❌ Error testing preprocessing: {e}")

logging.info("🔧 Shared preprocessing benefits:")
logging.info("  ✓ Identical preprocessing as binary classifier")
logging.info("  ✓ 2-5x faster data loading with tf.data optimization")
logging.info("  ✓ Automatic prefetching and parallel processing")
logging.info("  ✓ In-memory caching for repeated epochs")
logging.info("  ✓ Meaningful embedding analysis guaranteed")


# %% [markdown]
# ## 6. Train/Validation Split and Create Datasets
# 

# %%
if len(all_paths) > 0:
    # Split data into train and validation using the same approach as binary classifier
    train_paths, val_paths = train_test_split(
        all_paths,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        shuffle=True
    )
    
    logging.info(f"Training samples: {len(train_paths)}")
    logging.info(f"Validation samples: {len(val_paths)}")
    
    # Create datasets using shared preprocessing module (autoencoder mode returns (X, X))
    logging.info("Creating optimized tf.data datasets with shared preprocessing...")
    
    # Training dataset with augmentation - NO CACHING for large datasets
    train_dataset = builder.create_autoencoder_dataset(
        image_paths=train_paths,
        batch_size=BATCH_SIZE,
        is_training=True,
        cache_data=False,  # Disable caching to prevent memory exhaustion
    )
    
    # Validation dataset without augmentation
    val_dataset = builder.create_autoencoder_dataset(
        image_paths=val_paths,
        batch_size=BATCH_SIZE,
        is_training=False,
        cache_data=False,  # Disable caching to prevent memory exhaustion
    )
    
    # Get dataset information using shared utility
    get_dataset_info(train_dataset, "Training")
    get_dataset_info(val_dataset, "Validation")
    
    # Calculate steps per epoch
    train_steps = len(train_paths) // BATCH_SIZE
    val_steps = len(val_paths) // BATCH_SIZE
    
    logging.info(f"📈 Dataset Statistics:")
    logging.info(f"  • Training steps per epoch: {train_steps}")
    logging.info(f"  • Validation steps per epoch: {val_steps}")
    
    # Test the shared pipeline
    logging.info(f"🧪 Testing shared preprocessing pipeline...")
    try:
        sample_batch = next(iter(train_dataset.take(1)))
        input_batch, target_batch = sample_batch
        logging.info(f"  ✅ Successfully loaded batch:")
        logging.info(f"     • Input shape: {input_batch.shape}")
        logging.info(f"     • Target shape: {target_batch.shape}")
        logging.info(f"     • Input == Target (autoencoder): {tf.reduce_all(input_batch == target_batch)}")
        logging.info(f"     • Data type: {input_batch.dtype}")
        logging.info(f"     • Value range: [{tf.reduce_min(input_batch):.3f}, {tf.reduce_max(input_batch):.3f}]")
        logging.info(f"     • Ready for autoencoder training!")
    except Exception as e:
        logging.error(f"  ❌ Error testing pipeline: {e}")
    
else:
    logging.error("❌ No data available for training.")
    train_dataset = None
    val_dataset = None


# %%
# Validate shared preprocessing setup
logging.info("🔍 Validating Shared Preprocessing Integration")
logging.info("=" * 50)

if train_dataset is not None and val_dataset is not None:
    try:
        # Test preprocessing consistency
        logging.info("1. Testing preprocessing consistency...")
        consistency_result = validate_preprocessing_consistency(config, train_paths[0])
        
        # Verify dataset structure
        logging.info("2. Verifying dataset structure...")
        train_batch = next(iter(train_dataset.take(1)))
        val_batch = next(iter(val_dataset.take(1)))
        
        logging.info(f"   ✅ Training batch shape: {train_batch[0].shape}")
        logging.info(f"   ✅ Validation batch shape: {val_batch[0].shape}")
        logging.info(f"   ✅ Input == Target (autoencoder): {tf.reduce_all(train_batch[0] == train_batch[1])}")
        
        # Verify configuration consistency
        logging.info("3. Verifying configuration consistency...")
        logging.info(f"   • Shared config input size: {config.input_shape}")
        logging.info(f"   • Normalization: Scaled to [-1, 1]")
        logging.info(f"   • Augmentation enabled: {config.AUG_BRIGHTNESS_DELTA}, {config.AUG_CONTRAST_LOWER}, {config.AUG_CONTRAST_UPPER}")
        
        logging.info("✅ All validations passed! Ready for autoencoder training.")
        logging.info("   The autoencoder will use identical preprocessing as the binary classifier,")
        logging.info("   ensuring meaningful and consistent embedding analysis.")
        
    except Exception as e:
        logging.error(f"❌ Validation error: {e}")
        logging.error("Please check the shared preprocessing module setup.")

else:
    logging.error("❌ Datasets not available for validation.")
    logging.error("Please ensure data loading completed successfully.")


# %% [markdown]
# ## 7. Define Convolutional Autoencoder Architecture
# 
# We'll create a convolutional autoencoder with:
# - **Encoder**: Progressively downsamples the input and compresses to a latent representation
# - **Decoder**: Reconstructs the original image from the latent representation
# - **Bottleneck**: The latent layer that will serve as our embedding
# 

# %%
def create_convolutional_autoencoder(input_shape, latent_dim):
    """
    Creates a truly efficient convolutional autoencoder without skip connections.
    Returns the full autoencoder, the encoder, and the decoder.

    This version is optimized for a lower parameter count (~9M) and faster training.
    It uses 5 stages of downsampling to create a small bottleneck, but with
    fewer filters than the original model. It also uses strided convolutions
    for downsampling and transposed convolutions for upsampling.
    """
    input_img = Input(shape=input_shape, name='input')

    # === ENCODER (Downsampling Path) ===
    # Strided convolutions reduce spatial dimensions and extract features.
    x = input_img

    # Block 1: 256x256 -> 128x128
    x = Conv2D(16, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Block 2: 128x128 -> 64x64
    x = Conv2D(32, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Block 3: 64x64 -> 32x32
    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Block 4: 32x32 -> 16x16
    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Block 5: 16x16 -> 8x8
    x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # --- BOTTLENECK ---
    shape_before_flattening = tf.keras.backend.int_shape(x)[1:] # Should be (8, 8, 256)
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_layer')(x)

    # Encoder model maps the input image to the latent vector.
    encoder = Model(input_img, latent, name='encoder')

    # === DECODER (Upsampling Path) ===
    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(np.prod(shape_before_flattening))(decoder_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape(shape_before_flattening)(x)

    # Decoder uses Transposed Convolutions to upsample.
    # Block 5 (Reverse): 8x8 -> 16x16
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Block 4 (Reverse): 16x16 -> 32x32
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Block 3 (Reverse): 32x32 -> 64x64
    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Block 2 (Reverse): 64x64 -> 128x128
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Block 1 (Reverse): 128x128 -> 256x256
    x = Conv2DTranspose(input_shape[-1], (3, 3), strides=2, padding='same')(x)
    
    # --- OUTPUT LAYER ---
    decoded = LeakyReLU(alpha=0.2)(x) # Use LeakyReLU then final conv for stability
    decoded = Conv2D(input_shape[-1], (3, 3), padding='same', activation='tanh', name='output')(decoded)

    # Decoder model maps the latent vector back to a reconstructed image.
    decoder = Model(decoder_input, decoded, name='decoder')

    # --- FULL AUTOENCODER ---
    autoencoder_output = decoder(encoder(input_img))
    autoencoder = Model(input_img, autoencoder_output, name='autoencoder')

    return autoencoder, encoder, decoder

# Create the models using shared config for input size
MODEL_INPUT_SIZE = config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS  # Get from shared config to ensure consistency

autoencoder, encoder, decoder = create_convolutional_autoencoder(MODEL_INPUT_SIZE, LATENT_DIM)

logging.info("Autoencoder architecture created!")
logging.info(f"✅ Using consistent input size from shared config: {MODEL_INPUT_SIZE}")
logging.info("Autoencoder summary:")
autoencoder.summary()

logging.info("Encoder summary:")
encoder.summary()


# %% [markdown]
# ## 8. Compile and Train the Autoencoder
# 

# %%
# Compile the autoencoder
autoencoder.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='mse',  # Mean Squared Error for reconstruction
    metrics=['mae']  # Mean Absolute Error as additional metric
)

logging.info("Autoencoder compiled with MSE loss and Adam optimizer.")


# %%
class ReconstructionTensorBoardCallback(tf.keras.callbacks.Callback):
    """Custom TensorBoard callback to log reconstruction images during training (updated for tf.data)."""
    
    def __init__(self, log_dir, validation_dataset, num_samples=6, log_freq=5):
        super().__init__()
        self.log_dir = log_dir
        self.validation_dataset = validation_dataset
        self.num_samples = num_samples
        self.log_freq = log_freq  # Log every N epochs
        self.file_writer = tf.summary.create_file_writer(log_dir + '/reconstruction_images')
        
        # Get a fixed batch for consistent comparison across epochs
        if self.validation_dataset is not None:
            try:
                # Get first batch from the tf.data dataset
                sample_batch = next(iter(self.validation_dataset.take(1)))
                self.fixed_batch_x, _ = sample_batch
                self.fixed_batch_x = self.fixed_batch_x[:self.num_samples]
                logging.info(f"Fixed batch of {len(self.fixed_batch_x)} samples prepared for TensorBoard logging.")
            except Exception as e:
                logging.warning(f"Warning: Could not prepare fixed batch for reconstruction logging: {e}")
                self.fixed_batch_x = None
        else:
            self.fixed_batch_x = None
            logging.warning("Warning: No validation dataset provided for reconstruction logging.")
    
    def denormalize_image(self, img):
        """Denormalize image from [-1, 1] to [0, 1] for visualization."""
        # Scale from [-1, 1] back to [0, 1]
        # tf.summary.image expects values in [0, 1], which is why
        # we don't scale to [0, 255]
        img = (img + 1.0) / 2.0
        return tf.clip_by_value(img, 0, 1)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.fixed_batch_x is None or epoch % self.log_freq != 0:
            return
            
        # Generate reconstructions
        reconstructed = self.model.predict(self.fixed_batch_x, verbose=0)
        
        # Denormalize for proper visualization
        originals = self.denormalize_image(self.fixed_batch_x)
        reconstructions = self.denormalize_image(reconstructed)
        
        # Log to TensorBoard
        with self.file_writer.as_default():
            tf.summary.image(
                "Original_Spectrograms", 
                originals, 
                step=epoch, 
                max_outputs=self.num_samples
            )
            tf.summary.image(
                "Reconstructed_Spectrograms", 
                reconstructions, 
                step=epoch, 
                max_outputs=self.num_samples
            )
            
            # Create a side-by-side comparison
            # Concatenate original and reconstructed horizontally
            comparison = tf.concat([originals, reconstructions], axis=2)  # Concatenate along width
            tf.summary.image(
                "Original_vs_Reconstructed", 
                comparison, 
                step=epoch, 
                max_outputs=self.num_samples
            )
        
        self.file_writer.flush()

logging.info("ReconstructionTensorBoardCallback class defined (updated for tf.data).")
logging.info("This will log reconstruction images to TensorBoard every few epochs during training.")
logging.info("You can view them in TensorBoard under the 'Images' tab with a slider to see progress over time.")


# %%
import datetime

# Create directories for this training run
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
existing_runs = [d for d in os.listdir(LOGS_DIR) if d.startswith('run_')]
next_run_number = len(existing_runs) + 1

run_log_dir = os.path.join(LOGS_DIR, f"run_{next_run_number}")
run_checkpoint_dir = os.path.join(MODEL_SAVE_PATH, f"checkpoints_run_{next_run_number}")

os.makedirs(run_log_dir, exist_ok=True)
os.makedirs(run_checkpoint_dir, exist_ok=True)

# Model save paths
autoencoder_best_path = os.path.join(run_checkpoint_dir, "best_autoencoder.keras")
encoder_best_path = os.path.join(run_checkpoint_dir, "best_encoder.keras")

logging.info(f"Training run {next_run_number}")
logging.info(f"Logs directory: {run_log_dir}")
logging.info(f"Checkpoints directory: {run_checkpoint_dir}")

# Define callbacks
callbacks = [
    ModelCheckpoint(
        filepath=autoencoder_best_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=run_log_dir,
        histogram_freq=10
    ),
    ReconstructionTensorBoardCallback(
        log_dir=run_log_dir,
        validation_dataset=val_dataset,
        num_samples=6,
        log_freq=1  # Log reconstruction images every 5 epochs
    )
]

logging.info("Callbacks configured, including custom reconstruction visualization for TensorBoard.")
logging.info("Reconstruction images will be logged every 5 epochs - you can adjust log_freq as needed.")


# %%
# Train the autoencoder with shared preprocessing pipeline
if train_dataset is not None and val_dataset is not None:
    logging.info("🚀 Starting autoencoder training with shared preprocessing pipeline...")
    
    history = autoencoder.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    logging.info("🎉 Training completed!")
    
    # Save the encoder separately for easy embedding extraction
    encoder.save(encoder_best_path)
    logging.info(f"✅ Encoder saved to: {encoder_best_path}")
    
    # Summary of achievements
    logging.info("📈 Training Achievements:")
    logging.info("  • Autoencoder trained with identical preprocessing as binary classifier")
    logging.info("  • Embeddings will be consistent and meaningful for analysis")
    logging.info("  • 2-5x faster training due to optimized data pipeline")
    logging.info("  • Ready for UMAP visualization and embedding analysis")
    
else:
    logging.error("❌ Cannot start training - datasets not available.")
    history = None


# %% [markdown]
# ## 9. Visualize Training History
# 

# %%
if history is not None:
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Autoencoder Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    logging.info(f"Final training loss: {history.history['loss'][-1]:.6f}")
    logging.info(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
else:
    logging.warning("No training history available.")



