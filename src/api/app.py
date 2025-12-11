from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
import soundfile as sf
import io
import os
import jwt

from anemonefish_acoustics.data import preprocess_audio_for_inference, postprocess_prediction
from anemonefish_acoustics.models.utils import load_model
from anemonefish_acoustics.utils.logger import get_logger

logger = get_logger(__name__)


app = FastAPI()

# Security Scheme
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies the Supabase JWT token from the Authorization header.
    Requires SUPABASE_JWT_SECRET environment variable to be set.
    """
    token = credentials.credentials
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    
    if not secret:
        logger.error("SUPABASE_JWT_SECRET environment variable not set")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server authentication configuration error"
        )

    try:
        # Supabase uses HS256 algorithm
        # The 'aud' claim is usually 'authenticated' for logged in users
        payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token attempt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Allow the frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict", dependencies=[Depends(verify_token)])
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
        logger.info(f"Spectrograms shape: {spectrograms.shape}")
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

    return post_processed_prediction
    