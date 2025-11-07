from fastapi import FastAPI
import json
import base64
import soundfile as sf
import io

from src.anemonefish_acoustics.data import preprocess_audio_for_inference, postprocess_prediction
from src.anemonefish_acoustics.models.utils import load_model
from src.anemonefish_acoustics.utils.logger import get_logger

logger = get_logger(__name__)


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict(body: dict):
    """
    Predict the class of the audio file using the trained model.
    """
    # Extract base64 encoded audio data from the request body
    audio_b64 = body.get("audio_data")
    if not audio_b64:
        logger.error("No audio_data field found in request body")
        return {"error": "No audio_data field found in request body"}
    
    try:
        # Decode base64 string to bytes
        audio_bytes = base64.b64decode(audio_b64)
        
        # Create a BytesIO object from the decoded bytes
        audio_buffer = io.BytesIO(audio_bytes)
    except Exception as e:
        logger.error(f"Failed to decode audio: {str(e)}")
        return {"error": f"Failed to decode audio: {str(e)}"}

    # Preprocess audio data
    logger.info(f"Preprocessing audio data...")
    try:
        spectrograms = preprocess_audio_for_inference(audio_buffer, window_duration_s=0.4, slide_duration_s=0.2, sr_target=8000, n_fft=1024, hop_length=None, fmax=2000, logger=logger)
    except Exception as e:
        logger.exception(f"Failed to preprocess audio: {str(e)}")
        return {"error": f"Failed to preprocess audio: {str(e)}"}
    
    # TODO: Process audio_data with your trained model
    print(f"Model prediction....")
    model = load_model("latest", logger=logger)
    if model is None:
        logger.error("Failed to load model")
        return {"error": "Failed to load model"}
    
    prediction = model.predict(spectrograms, batch_size=32, verbose=1)
    
    # TODO: postprocess prediction for output
    print(f"Postprocessing prediction for output...")

    post_processed_prediction = postprocess_prediction(prediction, logger=logger)

    return {"prediction": post_processed_prediction}
    