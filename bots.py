import io
import os
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

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
# 🎯 MODEL & CLEAN PROMPT HINT CONFIGURATION
# ==========================================
STT_MODEL = "whisper-large-v3-turbo"

# --- CHANGED HERE: Pure vocabulary hint to guide script & prevent prompt leak ---
SYSTEM_PROMPT = "Roman Urdu, English, numbers 1 2 3, plus, minus, equal, kya haal hai, main theek hoon."

# --- BACKGROUND NOISE DETECTION & SILENCE CHECK ---
def process_audio_buffer(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)
            
        # Loudness (RMS) energy calculation for silence & noise check
        rms_energy = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
        
        # --- CHANGED HERE: Silence/Noise energy threshold check ---
        if rms_energy < 15.0:
            return None  # Signal that audio is pure silence or low background hiss
            
        # Clean background noise so Whisper gets clear voice input
        cleaned_audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.75)
        
        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, cleaned_audio_data.astype(np.int16))
        output_buffer.seek(0)
        
        return output_buffer.read()
    except Exception:
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
            try:
                st.html(f.read())
            except Exception:
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
        
    with st.spinner("⏳ Analyzing sound levels and filtering noise..."):
        processed_bytes = process_audio_buffer(audio_bytes)
        
    # --- CHANGED HERE: Clean warning handling when noise or silence is detected ---
    if processed_bytes is None:
        st.warning("⚠️ Noise or silence detected. Please speak loudly and clearly into the mic.")
    else:
        with st.spinner("⚡ Processing speech with Groq AI Agent..."):
            try:
                audio_file = io.BytesIO(processed_bytes)
                audio_file.name = "recording.wav"
                
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model=STT_MODEL,
                    prompt=SYSTEM_PROMPT,
                    response_format="json",
                    temperature=0.0
                )
                
                text_from_voice = transcription.text.strip()
                
                # Filter out Whisper hallucinated phrases on bad input
                hallucination_phrases = [
                    "thanks for watching", "thank you", "subtitles by", "amara.org",
                    "roman urdu", "transcribe audio"
                ]
                
                if any(phrase in text_from_voice.lower() for phrase in hallucination_phrases) and len(text_from_voice) < 30:
                    st.warning("⚠️ Voice was not clear. Please try speaking again.")
                elif len(text_from_voice) < 2:
                    st.warning("⚠️ Voice was too low or unclear.")
                elif text_from_voice:
                    st.session_state.last_transcription = text_from_voice
                    st.success("✅ Transcription complete!")
                else:
                    st.warning("Could not detect any speech clearly.")
                    
            except Exception as e:
                if "500" in str(e):
                    st.error("🚨 Groq Cloud Server overloaded (Error 500). Please wait 5 seconds and try again.")
                else:
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
