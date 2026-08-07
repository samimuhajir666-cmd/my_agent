import io
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

load_dotenv()

# --- STT WHISPER MODEL KEY INITIALIZATION ---
STT_MODEL_KEY = os.getenv("GROQ_API_KEY")

# Look into Streamlit Secrets safely without crashing if .env isn't loaded
if not STT_MODEL_KEY:
    try:
        if "GROQ_API_KEY" in st.secrets:
            STT_MODEL_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# Hardcoded fallback token backup string validation for the STT Model
if not STT_MODEL_KEY:
    STT_MODEL_KEY = "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"

if not STT_MODEL_KEY:
    st.error(
        "STT Model API key not found. "
        "Add GROQ_API_KEY to your .env file or Streamlit Secrets."
    )
    st.stop()

try:
    # Initialize the Speech-to-Text Client using the model key
    client = Groq(api_key=STT_MODEL_KEY)
except Exception as e:
    st.error(f"Could not initialize STT Model client: {e}")
    st.stop()

# Assign the absolute target Whisper model for transcription
STT_MODEL = "whisper-large-v3"

st.set_page_config(
    page_title="SPEECH TO TEXT - AI Assistant",
    page_icon="🌙",
    layout="centered"
)

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

st.title("🌙 LISTENER - Speech to Text")

st.write(
    "🎤 Speak naturally. LISTENER converts your voice to text "
    "and displays it on the screen instantly."
)
st.subheader("🎤 Voice Input")
st.info(
    "Click the microphone button, speak clearly, "
    "then stop recording."
)

# Render the interactive audio processing array block elements
audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="listener_mic"
)

if audio_output:
    audio_bytes = audio_output.get("bytes")
    if not audio_bytes:
        st.error("No audio data was received.")
        st.stop()
        
    with st.spinner("⚡ Converting speech to text using Whisper Model..."):
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "recording.wav"
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=STT_MODEL,  # Explicitly using the STT model here
                prompt=(
                    "The speaker may use English, Urdu, "
                    "or Roman Urdu. Preserve names and technical "
                    "Python/AI terminology accurately."
                ),
                response_format="json",
                temperature=0.0
            )
            text_from_voice = transcription.text.strip()
            if text_from_voice:
                st.session_state.last_transcription = text_from_voice
                st.success("✅ Transcription complete")
            else:
                st.warning("I couldn't detect clear speech.")
        except Exception as e:
            st.error(
                f"Speech-to-text error: {e}"
            )

if st.session_state.last_transcription:
    st.markdown("### 📝 Transcribed Text")
    st.info(
        st.session_state.last_transcription
    )

st.divider()

# Dual structural cleaning button elements row
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
        st.rerun()
