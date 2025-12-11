"""
Script to test the /predict endpoint by loading an audio file and sending it as base64.
"""

import base64
import requests
import json

# Path to the audio file
AUDIO_FILE_PATH = "/Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/data/1_raw/papua_new_guines_2023/audio/20230210_000001_LL_B55_M_R_with labels.wav"

# API endpoint (update this to your actual endpoint URL)
API_URL = "https://qqccznuahn.eu-west-2.awsapprunner.com/predict"


def load_audio_to_b64(audio_path: str) -> str:
    """
    Load an audio file and encode it to base64 string.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Base64 encoded string of the audio file
    """
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    
    # Encode to base64
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    return audio_b64


def create_request_body(audio_b64: str) -> dict:
    """
    Create the request body dictionary for the /predict endpoint.
    
    Args:
        audio_b64: Base64 encoded audio string
        
    Returns:
        Dictionary with audio_data field
    """
    return {
        "audio_data": audio_b64
    }


def main():
    print(f"Loading audio file: {AUDIO_FILE_PATH}")
    
    # Load and encode audio
    audio_b64 = load_audio_to_b64(AUDIO_FILE_PATH)
    print(f"Audio encoded to base64 (length: {len(audio_b64)} characters)")
    
    # Create request body
    body = create_request_body(audio_b64)
    print(f"Request body created with 'audio_data' field")
    
    # Optional: Send request to API
    print(f"\nSending request to {API_URL}...")
    try:
        response = requests.post(API_URL, json=body)
        print(f"Response status: {response.status_code}")
        print(f"Response body: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error sending request: {str(e)}")
        print("\nIf the server is not running, you can still use the body dict:")
        print(f"Body keys: {list(body.keys())}")
        print(f"Audio data length: {len(body['audio_data'])}")


if __name__ == "__main__":
    main()

