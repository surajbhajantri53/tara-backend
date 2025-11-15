from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import asyncio
import uuid
import os
import speech_recognition as sr
import pyttsx3
import ssl
import urllib3

# ===============================
# CONFIG
# ===============================
#HF_MODEL_ID = "Surajsb/STS"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
   # <-- PUT TOKEN HERE

AUDIO_DIR = "./audio_outputs"
os.makedirs(AUDIO_DIR, exist_ok=True)

VR_API_KEY = "VR_KEY_12345_TARA_ROBOT"
USER_API_KEY = "USER_KEY_67890_WEB_ACCESS"

app = Flask(__name__)
CORS(app)

recognizer = sr.Recognizer()

# Disable SSL warnings
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context


# ===============================
# HUGGINGFACE API INFERENCE
# ===============================
def hf_infer(question: str):
    """
    Improved HF inference with better error handling
    """
    try:
        # Use direct API endpoint instead of router
        url = r"https://api-inference.huggingface.co/models/Surajsb/STS"
        
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {"inputs": question}
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Check status codes first
        if response.status_code == 401:
            return "Authentication failed. Check your HF_API_TOKEN."
        
        if response.status_code == 503:
            return "Model is loading. Please try again in 20 seconds."
        
        if response.status_code == 500:
            return "Hugging Face server error. Try again later."
        
        # Try to parse JSON
        try:
            data = response.json()
        except:
            # If JSON parsing fails, return the raw text for debugging
            print(f"Raw response: {response.text[:500]}")
            print(f"Status code: {response.status_code}")
            return f"Error: Invalid response from API"
        
        # Handle list response
        if isinstance(data, list) and len(data) > 0:
            if "generated_text" in data[0]:
                return data[0]["generated_text"]
            return str(data[0])
        
        # Handle dict response
        if isinstance(data, dict):
            if "generated_text" in data:
                return data["generated_text"]
            # Check for error in response
            if "error" in data:
                return f"API Error: {data['error']}"
            return str(data)
        
        return str(data)
    
    except requests.exceptions.Timeout:
        return "Request timed out. Model may be too slow."
    except requests.exceptions.ConnectionError:
        return "Connection error. Check your internet connection."
    except Exception as e:
        print(f"Exception in hf_infer: {str(e)}")
        return f"Unexpected error: {str(e)}"


# ===============================
# TEXT → SPEECH
# ===============================
async def tts_edge(text, voice, filename):
    import edge_tts

    voices = {
        "male": "en-US-GuyNeural",
        "male_indian": "en-IN-PrabhatNeural"
    }

    try:
        communicate = edge_tts.Communicate(text, voices.get(voice, "en-IN-PrabhatNeural"))
        await communicate.save(filename)
        return filename
    except:
        return None


def tts_pyttsx3(text, filename):
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename


def generate_tts(text, voice="male_indian"):
    filename = f"{AUDIO_DIR}/{uuid.uuid4()}.mp3"
    try:
        result = asyncio.run(tts_edge(text, voice, filename))
        if result:
            return filename
    except:
        pass

    # fallback
    wav = filename.replace(".mp3", ".wav")
    return tts_pyttsx3(text, wav)


# ===============================
# SPEECH → TEXT
# ===============================
def speech_to_text(file_path):
    try:
        with sr.AudioFile(file_path) as src:
            recognizer.adjust_for_ambient_noise(src)
            audio = recognizer.record(src)
        return recognizer.recognize_google(audio)
    except:
        return "Could not recognize speech."


# ===============================
# HELPERS
# ===============================
def check_api(key, required):
    return key == required


# ===============================
# ROUTES
# ===============================

@app.route("/")
def index():
    return {"status": "online", "mode": "huggingface-api"}


# ---------- USER FRONTEND ----------
@app.route("/user/ask_with_voice", methods=["POST"])
def user_ask():
    if not check_api(request.headers.get("X-API-Key"), USER_API_KEY):
        return jsonify({"error": "Invalid API key"}), 401

    data = request.get_json()
    question = data.get("question", "")
    voice = data.get("voice", "male_indian")

    answer = hf_infer(question)
    audio = generate_tts(answer, voice)

    return {
        "question": question,
        "answer": answer,
        "audio_url": f"/audio/{os.path.basename(audio)}"
    }


# ---------- VR TEXT ----------
@app.route("/VR/text_to_voice", methods=["POST"])
def vr_text_to_voice():
    if not check_api(request.headers.get("X-API-Key"), VR_API_KEY):
        return jsonify({"error": "Invalid API key"}), 401

    data = request.get_json()
    q = data["question"]
    voice = data.get("voice", "male_indian")

    answer = hf_infer(q)
    audio = generate_tts(answer, voice)

    return send_file(audio, mimetype="audio/mpeg")


# ---------- VR VOICE ----------
@app.route("/VR/voice_to_voice", methods=["POST"])
def vr_voice_to_voice():
    if not check_api(request.headers.get("X-API-Key"), VR_API_KEY):
        return jsonify({"error": "Invalid API key"}), 401

    audio_file = request.files["audio"]
    temp_path = f"{AUDIO_DIR}/temp_{uuid.uuid4()}.wav"
    audio_file.save(temp_path)

    question = speech_to_text(temp_path)
    os.remove(temp_path)

    answer = hf_infer(question)
    audio = generate_tts(answer, "male_indian")

    return send_file(audio, mimetype="audio/mpeg")


# ---------- AR JSON ----------
@app.route("/AR/text_and_audio", methods=["POST"])
def ar_text_and_audio():
    if not check_api(request.headers.get("X-API-Key"), VR_API_KEY):
        return {"error": "Invalid API key"}, 401

    data = request.get_json()
    q = data["question"]
    voice = data.get("voice", "male_indian")

    answer = hf_infer(q)
    audio = generate_tts(answer, voice)

    return {
        "answer": answer,
        "audio_url": f"/audio/{os.path.basename(audio)}"
    }


@app.route("/audio/<name>")
def serve_audio(name):
    path = f"{AUDIO_DIR}/{name}"
    return send_file(path)


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006)
