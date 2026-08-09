import io
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from st_audiorec import st_audiorec

# --- 1. CONFIGURATION (MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="SPEECH TO TEXT - AI Assistant",
    page_icon="🌙",
    layout="centered"
)

# --- 2. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# Look into your .env file or hosting environment variables first
STT_MODEL_KEY = os.getenv("GROQ_API_KEY")

# Look into Streamlit/Hosting Secrets safely without crashing if .env isn't loaded
if not STT_MODEL_KEY:
    try:
        if "GROQ_API_KEY" in st.secrets:
            STT_MODEL_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# Hardcoded fallback token backup string validation (Kept exactly as requested)
if not STT_MODEL_KEY:
    STT_MODEL_KEY = "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"

if not STT_MODEL_KEY:
    st.error(
        "STT Model API key not found. "
        "Add GROQ_API_KEY to your .env file or platform environment configs."
    )
    st.stop()

try:
    # Initialize the Speech-to-Text Client using the active model key
    client = Groq(api_key=STT_MODEL_KEY)
except Exception as e:
    st.error(f"Could not initialize STT Model client: {e}")
    st.stop()

# Optimized low-latency conversational speech engine
STT_MODEL = "whisper-large-v3-turbo"

# --- 3. LOAD EXTERNAL UI FILES SAFELY ---
def load_css(file_path="style.css"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_html(file_path="index.html"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)

# Load your custom styling structure
load_css("style.css")
load_html("index.html")

# --- 4. APP STATE & AUDIOMODULE LOOP ---
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

st.subheader("🎤 Voice Input")

# Native component with professional embedded wave visualizers and recording controls
audio_bytes = st_audiorec()

if audio_bytes:
    with st.spinner("⚡ Processing audio and clearing background noise..."):
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "recording.wav"
            
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=STT_MODEL,
                prompt=(
                    "Environment Note: This audio contains background noise, room echo, hums, and microphone crackle. "
                    "Ignore all background sounds, static hiss, breathing artifacts, and ambient noise completely. "
                    "Transcribe ONLY the explicit spoken human words. "
                    "Language Profile: The user speaks fluently in English, Urdu, or mixed Roman Urdu. "
                    "Do not try to translate or fix the vocabulary—transcribe exactly what was said literally."
                    "do not ans the what user ask just listen clearly and cinvert user,s voice into text"
                ),
                response_format="json",
                temperature=0.0
            )
            
            text_from_voice = getattr(transcription, 'text', '')
            if isinstance(text_from_voice, str):
                text_from_voice = text_from_voice.strip()
            
            if text_from_voice:
                st.session_state.last_transcription = text_from_voice
                st.success("✅ Transcription complete")
            else:
                st.warning("I couldn't detect clear speech over the room background noise.")
        except Exception as e:
            st.error(f"Speech-to-text error: {e}")

# --- 5. UI OUTPUT RENDER DISPLAY ---
if st.session_state.last_transcription:
    st.markdown("### 📝 Transcribed Text")
    st.markdown(
        f'<div class="output-box">{st.session_state.last_transcription}</div>', 
        unsafe_allow_html=True
    )

st.divider()

# Action Layout Interface Control Row
col1, col2 = st.columns(2)

with col1:
    if st.button("🛑 Stop & Process Text", use_container_width=True):
        if st.session_state.last_transcription:
            st.success("Current text processing locked successfully.")
        else:
            st.warning("No live recording stream active to commit.")

with col2:
    if st.button("🗑️ Clear Text", use_container_width=True):
        st.session_state.last_transcription = ""
        st.experimental_rerun()
