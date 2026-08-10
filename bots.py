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
    "You are a professional AI transcription agent. "
    "Transcribe the audio exactly as heard using ONLY the English/Roman alphabet script. "
    "Do NOT translate the language or answer the content. If the speaker speaks Urdu, "
    "transcribe it strictly in Roman Urdu (e.g., 'kya haal hai', 'main theek hoon'). "
    "Accurately preserve all spoken numbers, digits, mathematical terms, and technical keywords "
    "(e.g., 'hello 1, 2, 3', 'plus', 'equal'). "
    "If the audio contains only static, noise, silence, or unintelligible speech that you cannot "
    "understand clearly, output exactly this phrase and nothing else: 'I could not understand the audio clearly, please say clearly and try again.'"
)


# --- BACKGROUND NOISE DETECTION & SILENCE CHECK ---
def process_audio_buffer(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)
            
        # 1. FIXED: Calculate Root Mean Square (RMS) loudness to detect silence
        # This checks the physical volume level of the recording
        rms_energy = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
        
        # If the energy level is below 15.0, it means it is pure silence or just minor background hiss
        if rms_energy < 15.0:
            return None  # Signal that the audio is silent
            
        # 2. Apply Noise Reduction if the audio actually contains speech
        cleaned_audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.95)
        
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

# --- FIXED: Used st.html instead of st.markdown to render pure layout blocks ---
def load_html(file_path="index.html"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            # Purani line 'st.markdown' ko hata kar ye native line lagayein:
            st.html(f.read())


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
        # Process audio data arrays
        processed_bytes = process_audio_buffer(audio_bytes)
        
    # FIXED: If the function returned None, it means the audio was completely silent
    if processed_bytes is None:
        st.warning("⚠️ No speech detected. Please speak into the microphone.")
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
                
                # Double-check protection against generic Whisper hallucinations
                hallucination_phrases = ["thanks for watching", "thank you", "subtitles by", "amara.org"]
                if any(phrase in text_from_voice.lower() for phrase in hallucination_phrases) and len(text_from_voice) < 25:
                    st.warning("⚠️ No clear speech detected.")
                elif text_from_voice:
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
