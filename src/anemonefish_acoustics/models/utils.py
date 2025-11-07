import tensorflow as tf
from anemonefish_acoustics import MODELS_DIR, SRC_DIR, PROJECT_ROOT
import os

def load_model(model_version, logger=None):
    """
    Load a TensorFlow Keras model from disk.
    
    Args:
        model_version (str): Version of the model to load. If "latest", loads the most recent model.
        logger: Logger instance for logging messages.
    
    Returns:
        tf.keras.Model: Loaded Keras model, or None if loading fails.
    """
    if not os.path.exists(MODELS_DIR):
        logger.error(f"Models directory not found: {MODELS_DIR}")
        return None
    if model_version == "latest":
        model_path = os.path.join(MODELS_DIR, "latest", "model.keras")
    else:
        # Use specific model version
        model_path = os.path.join(MODELS_DIR, model_version, "model.keras")
        
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
        
    # Load the model
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Successfully loaded model from: {model_path}")
        
    return model