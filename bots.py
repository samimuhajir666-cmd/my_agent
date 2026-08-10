import io
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

# --- AUDIO EXTRACTION LIBRARIES FOR NOISE CANCELLATION ---
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr

load_dotenv()

# --- STT MODEL KEY INITIALIZATION ---
STT_MODEL_KEY = os.getenv("GROQ_API_KEY")

if not STT_MODEL_KEY:
    try:
        if "GROQ_API_KEY" in st.secrets:
            STT_MODEL_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

if not STT_MODEL_KEY:
    STT_MODEL_KEY = "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"

if not STT_MODEL_KEY:
    st.error("GROQ_API_KEY not found in .env or Streamlit Secrets.")
    st.stop()

try:
    client = Groq(api_key=STT_MODEL_KEY)
except Exception as e:
    st.error(f"Could not initialize Groq client: {e}")
    st.stop()

# ==========================================
# 🎯 MODEL, PROMPT & NOISE CONFIGURATION
# ==========================================
STT_MODEL = "whisper-large-v3"

SYSTEM_PROMPT = (
    "Transcribe the audio exactly as heard using ONLY the English/Roman alphabet script. "
    "Do NOT translate or answer. If the speaker speaks Urdu, transcribe it strictly in "
    "Roman Urdu (e.g., 'kya haal hai', 'main theek hoon'). Accurately preserve all spoken numbers, "
    "digits, mathematical terms, and technical keywords (e.g., 'hello 1, 2, 3', 'plus', 'equal')."
)

# --- BACKGROUND NOISE DETECTION ENGINE ---
def remove_background_noise(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)
        
        # Convert stereo channel to mono if necessary
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)
            
        # Spectral gating drops ambient background hums, keyboard clicks, and fan noises
        cleaned_audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.95)
        
        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, cleaned_audio_data.astype(np.int16))
        output_buffer.seek(0)
        
        return output_buffer.read()
    except Exception:
        # Pass raw audio seamlessly if buffer array processing hits any exception frames
        return audio_bytes

st.set_page_config(
    page_title="SPEECH TO TEXT - AI Assistant",
    page_icon="👾",
    layout="centered"
)

# --- LOAD HTML & CSS FILES SAFELY ---
def load_css(file_path="style.css"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_html(file_path="index.html"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)

load_css("style.css")
load_html("index.html")

# --- SESSION STATE ---
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

st.subheader("🎤 Voice Input")

audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="listener_mic"
)

# --- AUDIO PROCESSING LOGIC ---
if audio_output:
    audio_bytes = audio_output.get("bytes")
    if not audio_bytes:
        st.error("No audio data received.")
        st.stop()
        
    # 🌟 NEW ADDITION: Cleans up background disruption artifacts before hitting cloud APIs
    with st.spinner("⏳ Filter out background noise..."):
        audio_bytes = remove_background_noise(audio_bytes)
        
    with st.spinner("⚡ Processing speech with Groq AI Agent..."):
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "recording.wav"
            
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=STT_MODEL,
                prompt=SYSTEM_PROMPT,
                response_format="json",
                temperature=0.0
            )
            
            text_from_voice = transcription.text.strip()
            if text_from_voice:
                st.session_state.last_transcription = text_from_voice
                st.success("✅ Transcription complete!")
            else:
                st.warning("Could not detect any speech clearly.")
        except Exception as e:
            st.error(f"Transcription error: {e}")

# --- DISPLAY OUTPUT ---
if st.session_state.last_transcription:
    st.markdown("### 📝 Transcribed Text")
    st.markdown(
        f"""
        <div class="output-card">
            <div class="output-title">Result:</div>
            <div class="output-text">{st.session_state.last_transcription}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🛑 Stop & Process Text", use_container_width=True):
        if st.session_state.last_transcription:
            st.success("Text saved/locked in session state.")
        else:
            st.warning("No recorded text available.")

with col2:
    if st.button("🗑️ Clear Text", use_container_width=True):
        st.session_state.last_transcription = ""
        st.rerun()
