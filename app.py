from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import speech_recognition as sr
import asyncio
import os
import uuid
import warnings
import ssl
import urllib3
import base64
import requests
import pyttsx3


###############################
# INIT
###############################
app = Flask(__name__)
CORS(app)

warnings.filterwarnings("ignore")

HF_MODEL_ID = "Surajsb/STS"      # <-- your HuggingFace model repo
AUDIO_DIR = "./audio_outputs"
os.makedirs(AUDIO_DIR, exist_ok=True)

VR_API_KEY = "VR_KEY_12345_TARA_ROBOT"
USER_API_KEY = "USER_KEY_67890_WEB_ACCESS"


###############################
# SSL FIX (optional)
###############################
_original_ssl_context = ssl.create_default_context
def _insecure_ssl_context(*args, **kwargs):
    ctx = _original_ssl_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

ssl.create_default_context = _insecure_ssl_context
urllib3.disable_warnings()


###############################
# LOAD MODEL FROM HUGGINGFACE
###############################
print("\n=================================")
print("   LOADING MODEL FROM HF")
print("=================================")

try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    tokenizer = T5Tokenizer.from_pretrained(HF_MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(HF_MODEL_ID).to(device)
    model.eval()
    print("Model loaded successfully.")

except Exception as e:
    print("❌ Could not load model:", e)
    model = None
    tokenizer = None
    device = None


recognizer = sr.Recognizer()


###############################
# TEXT → SPEECH
###############################
async def text_to_speech_edge(text, voice, filename):
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


def text_to_speech_pyttsx3(text, filename):
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename


def generate_tts(text, voice="male_indian"):
    fname = f"{AUDIO_DIR}/{uuid.uuid4()}.mp3"
    try:
        result = asyncio.run(text_to_speech_edge(text, voice, fname))
        if result:
            return fname
    except:
        pass

    fname = fname.replace(".mp3", ".wav")
    return text_to_speech_pyttsx3(text, fname)


###############################
# MODEL INFERENCE
###############################
def generate_answer(question):
    if not model:
        return f"[Mock Answer] {question}"

    try:
        input_text = f"question: {question}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        output_ids = model.generate(
            input_ids,
            max_length=200,
            num_beams=4,
            early_stopping=True
        )

        ans = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return ans

    except Exception as e:
        return f"Error generating answer: {e}"


###############################
# SPEECH → TEXT
###############################
def speech_to_text(file_path):
    try:
        with sr.AudioFile(file_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except:
        return "Could not understand audio"


###############################
# HELPERS
###############################
def validate(api, required):
    return api == required


###############################
# ROUTES
###############################

@app.route("/")
def home():
    return {"status": "OK", "model": HF_MODEL_ID}


@app.route("/user/ask_with_voice", methods=["POST"])
def user_ask_with_voice():
    if not validate(request.headers.get("X-API-Key"), USER_API_KEY):
        return jsonify({"error": "Invalid API KEY"}), 401

    data = request.get_json()
    question = data.get("question", "")
    voice = data.get("voice", "male_indian")

    answer = generate_answer(question)
    audio_path = generate_tts(answer, voice)

    return jsonify({
        "status": "success",
        "question": question,
        "answer": answer,
        "audio_url": f"/audio/{os.path.basename(audio_path)}"
    })


@app.route("/VR/text_to_voice", methods=["POST"])
def vr_text_to_voice():
    if not validate(request.headers.get("X-API-Key"), VR_API_KEY):
        return jsonify({"error": "Invalid VR API KEY"}), 401

    data = request.get_json()
    question = data.get("question")
    voice = data.get("voice", "male_indian")

    answer = generate_answer(question)
    audio_path = generate_tts(answer, voice)

    return send_file(audio_path, mimetype="audio/mpeg")


@app.route("/VR/voice_to_voice", methods=["POST"])
def vr_voice_to_voice():
    if not validate(request.headers.get("X-API-Key"), VR_API_KEY):
        return jsonify({"error": "Invalid VR API KEY"}), 401

    audio = request.files["audio"]
    temp = f"{AUDIO_DIR}/temp_{uuid.uuid4()}.wav"
    audio.save(temp)

    question = speech_to_text(temp)
    os.remove(temp)

    answer = generate_answer(question)
    audio_path = generate_tts(answer)

    return send_file(audio_path, mimetype="audio/mpeg")


@app.route("/AR/text_and_audio", methods=["POST"])
def ar_text_and_audio():
    if not validate(request.headers.get("X-API-Key"), VR_API_KEY):
        return jsonify({"error": "Invalid API KEY"}), 401

    data = request.get_json()
    question = data["question"]
    voice = data.get("voice", "male_indian")

    answer = generate_answer(question)
    audio_path = generate_tts(answer, voice)

    return jsonify({
        "answer": answer,
        "audio_url": f"/audio/{os.path.basename(audio_path)}"
    })


@app.route("/audio/<file>")
def audio_file(file):
    path = f"{AUDIO_DIR}/{file}"
    return send_file(path)


###############################
# RUN SERVER
###############################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=True)
