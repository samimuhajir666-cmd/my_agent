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

# ============================
# 🔑 API KEY INITIALIZATION
# ============================
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
    st.error("GROQ_API_KEY not found. Please set it in .env or Streamlit Secrets.")
    st.stop()

try:
    client = Groq(api_key=STT_MODEL_KEY)
except Exception as e:
    st.error(f"Could not initialize Groq client: {e}")
    st.stop()

# ============================
# 🎯 MODEL & PROMPT
# ============================
STT_MODEL = "whisper-large-v3-turbo"

# Whisper prompt: only for spelling/context guidance (max 244 chars)
SYSTEM_PROMPT = SYSTEM_PROMPT = """Roman Urdu,English,Urdu,numbers,digits, plus, minus, equal, multiply,divide, kya, kaise, hain, main, theek, hoon, han, nahi, yes, no, ok, time, date, price, amount, payment, order, help, support,ticket, error, code, terminal, POS, card, payment, failed, declined, approved, transaction, receipt, invoice, refund, void, settle,pre-auth, capture, reversal, tip, discount, tax, subtotal, total, change, cash, credit, debit, network, timeout, connection, API, gateway, processor,host, port, status, code, message, response, request, payload, JSON, kya hua, kaise hai, theek hai, main hoon, han bhai, nahi bhai, chalo, ruko, suno, dekho,batao, karo, one, two, three, four, five, six, seven, eight, nine, ten, hundred, thousand, lakh, crore, January, February, March, April, May, June, July, August,September,October,November,December,Monday,Tuesday,Wednesday,Thursday,Friday,Saturday, Sunday"""
# ============================
# 🎚️ AUDIO PROCESSING
# ============================
MIN_RMS_ENERGY = 60.0
MIN_DURATION_SECONDS = 0.7

def process_audio_buffer(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)

        duration_seconds = len(audio_data) / float(sample_rate)
        if duration_seconds < MIN_DURATION_SECONDS:
            return None

        rms_energy = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        if rms_energy < MIN_RMS_ENERGY:
            return None

        cleaned_audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.65)

        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, cleaned_audio_data.astype(np.int16))
        output_buffer.seek(0)

        return output_buffer.read()
    except Exception:
        return audio_bytes

# ============================
# 🖥️ STREAMLIT UI
# ============================
st.set_page_config(
    page_title="Speech to Text",
    page_icon="🎤",
    layout="centered"
)

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

# ============================
# 🧠 TRANSCRIPTION LOGIC
# ============================
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
                    temperature=0.0,
                    language="en"
                    # 👈 Ensures correct language detection
                )

                text_from_voice = transcription.text.strip()

                if text_from_voice and len(text_from_voice) > 1:
                    st.session_state.last_transcription = text_from_voice
                    st.success("✅ Complete!")
                else:
                    st.warning("⚠️ Could not detect clear speech.")

            except Exception as e:
                st.error(f"Transcription error: {e}")

# ============================
# 📝 DISPLAY OUTPUT
# ============================
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
