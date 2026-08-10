import io
import os
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
# 🎯 MODEL & PROMPT CONFIGURATION (UPDATED)
# ==========================================

# Model set to fast Groq Whisper Turbo (Whisper V3 ka upgraded fast model)
STT_MODEL = "whisper-large-v3-turbo"

# --- CHANGED HERE: Enforced strict Roman script rules without translation ---
SYSTEM_PROMPT = (
    "Listen carefully to the audio and write down EXACTLY what you hear using ONLY Roman/English letters. "
    "Do NOT translate the language. If spoken in Urdu, write it strictly in Roman Urdu (e.g., 'kya haal hai', 'main theek hoon'). "
    "Always output text strictly in the English alphabet (Roman script)."
    "understand user,s math if user say hello 1 ,2 ,3 or  like that understand well ."
    "convert speech into text just in in english and roman language  ,"
    "no metter user say any language you just ans user in roman language"
)

st.set_page_config(
    page_title="SPEECH TO TEXT - AI Assistant",
    page_icon="🌙",
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

# Audio Mic Input
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
        
    with st.spinner("⚡ Processing speech with Groq AI Agent..."):
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "recording.wav"
            
            # Transcription Request with updated Model and Agent Prompt
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=STT_MODEL,
                prompt=SYSTEM_PROMPT,  # Professional Agent Prompt
                # CHANGED HERE: Removed language="ur" / language="en" so Whisper doesn't force translation
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

# Control Buttons
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
