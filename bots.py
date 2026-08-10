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
    st.error("GROQ_API_KEY not found. Please set it in .env or Streamlit Secrets.")
    st.stop()

try:
    client = Groq(api_key=STT_MODEL_KEY)
except Exception as e:
    st.error(f"Could not initialize Groq client: {e}")
    st.stop()

# ==========================================
# MODEL & PROMPT CONFIGURATION
# ==========================================
STT_MODEL = "whisper-large-v3-turbo"

# NOTE: Groq/Whisper's "prompt" field is NOT an instruction field.
# Whisper does not "follow rules" written in the prompt -- it treats the
# prompt as if it were the PRECEDING TRANSCRIPT, and continues in the same
# style/script/vocabulary. If you write instructions ("transcribe numbers",
# "don't solve math"), Whisper can literally transcribe those words back to
# you as if they were spoken, especially on short/quiet clips.
#
# The fix: give it a natural SAMPLE of the exact style you want (Roman Urdu +
# English mixed, numbers written out, casual tone) instead of commands.
SYSTEM_PROMPT = (
    "Yeh transcription hai. Aap kaisay hain? Mein theek hoon, shukriya. "
    "Aaj mausam acha hai. Ek, do, teen, char, panch. Meeting kal subah das "
    "bajay hogi. Please send karo, thank you, no problem."
)

# Safety check so this never silently breaks again in the future
if len(SYSTEM_PROMPT) > 896:
    st.error(
        f"SYSTEM_PROMPT is {len(SYSTEM_PROMPT)} characters, exceeds Groq's 896 character "
        "limit for the transcription prompt field. Shorten it before running."
    )
    st.stop()

# --- AUDIO PROCESSING FOR NOISE REDUCTION ---
MIN_RMS_ENERGY = 60.0       # below this = treated as silence/background noise
MIN_DURATION_SECONDS = 0.6  # below this = too short, Whisper tends to hallucinate


def process_audio_buffer(audio_bytes):
    """
    Returns cleaned audio bytes, or None if the clip is silence/noise/too short
    (i.e. not worth sending to the API).
    """
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)

        duration_seconds = len(audio_data) / float(sample_rate)
        if duration_seconds < MIN_DURATION_SECONDS:
            return None

        rms_energy = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))

        # Silence / pure background noise check
        if rms_energy < MIN_RMS_ENERGY:
            return None

        cleaned_audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.75)

        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, cleaned_audio_data.astype(np.int16))
        output_buffer.seek(0)

        return output_buffer.read()
    except Exception:
        return audio_bytes


st.set_page_config(
    page_title="SPEECH TO TEXT",
    page_icon="🎤",
    layout="centered"
)


# --- LOAD HTML & CSS SAFELY ---
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

    with st.spinner("⏳ Processing sound..."):
        processed_bytes = process_audio_buffer(audio_bytes)

    if processed_bytes is None:
        st.warning("⚠️ Noise, silence, or clip too short. Please speak clearly into the mic.")
    else:
        with st.spinner("⚡ Transcribing speech..."):
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

                if text_from_voice and len(text_from_voice) > 1:
                    st.session_state.last_transcription = text_from_voice
                    st.success("✅ Complete!")
                else:
                    st.warning("⚠️ Could not detect clear speech.")

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
    if st.button("🛑 Lock Text", use_container_width=True):
        if st.session_state.last_transcription:
            st.success("Text saved.")
        else:
            st.warning("No recorded text available.")

with col2:
    if st.button("🗑️ Clear Text", use_container_width=True):
        st.session_state.last_transcription = ""
        st.rerun()
