import html
import io
import os
import re
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

# ============================
# 🖥️ STREAMLIT PAGE CONFIG
# ============================
st.set_page_config(
    page_title="High-Precision Speech to Text",
    page_icon="🎤",
    layout="centered",
)
load_dotenv()

# ============================
# 🔑 API KEYS CONFIGURATION
# ============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set GROQ_API_KEY in .env file or Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ============================
# 🧹 TEXT CLEANUP FUNCTION
# ============================
def clean_roman_script(text):
    if not text:
        return ""
    # Remove diacritics and strange accents
    text = re.sub(r"[’'‘`\^\~]", "", text)
    
    replacements = {
        "iN": "in", "aN": "an", "uN": "un", "eN": "en",
        "N": "n", "gii": "gi", "uu": "u", "aa": "a",
        "DD": "d", "TT": "t", "RR": "r", "khh": "kh"
    }
    for word, repl in replacements.items():
        text = text.replace(word, repl)
        
    return re.sub(r"\s+", " ", text).strip()

# ============================
# 🎚️ AUDIO NORMALIZER (SOFT VOICE BOOST)
# ============================
def normalize_and_prepare_audio(audio_bytes):
    """Halki/dheemi awaz ko auto-boost karta hai taake STT misread na kare."""
    audio_file = io.BytesIO(audio_bytes)
    sample_rate, audio_data = wav.read(audio_file)

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    audio_data = audio_data.astype(np.float32)
    max_val = np.max(np.abs(audio_data))

    # Dynamic Audio Gain Adjustment (If user speaks softly)
    if max_val > 0:
        target_peak = 28000.0  # Max amplitude peak
        gain = target_peak / max_val
        gain = min(gain, 10.0) # Limit extreme boost to prevent noise distortion
        audio_data = audio_data * gain

    audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)

    output_buffer = io.BytesIO()
    wav.write(output_buffer, sample_rate, audio_data)
    output_buffer.seek(0)
    output_buffer.name = "input_speech.wav"
    return output_buffer

# ============================
# 🤖 URDU TO ROMAN URDU LLM CONVERTER
# ============================
def convert_urdu_to_roman(text):
    """Accurately converts Urdu script / mixed text to clean Roman Urdu."""
    if not text or len(text.strip()) == 0:
        return ""

    # Known Whisper silence hallucinations list
    hallucinations = [
        "satsang with mooji", "ignore background", "subtitles by",
        "amara.org", "thank you for watching", "mb1", "subscribe"
    ]
    if any(h in text.lower() for h in hallucinations):
        return ""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an exact phonetic transliterator from Urdu/Hindi to Roman Urdu (Latin Script). "
                        "CRITICAL RULES:\n"
                        "1. Convert the input text into clean, natural Roman Urdu/English.\n"
                        "2. DO NOT translate the meaning to English. Keep the exact words spoken (e.g. 'Jahil ke bacho ko' -> 'Jahil ke bacho ko').\n"
                        "3. Do NOT add explanations, notes, or quotes. Output ONLY the converted text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Fallback to raw text if LLM call fails
        return text

# ============================
# 🎙️ GROQ WHISPER TRANSCRIBE
# ============================
def transcribe_with_whisper(audio_buffer):
    """Groq Whisper Engine + LLM Roman Urdu Normalizer"""
    try:
        # Step 1: Capture native Urdu speech accurately with Urdu language setting
        transcription = client.audio.transcriptions.create(
            file=(audio_buffer.name, audio_buffer.read(), "audio/wav"),
            model="whisper-large-v3",
            language="ur",  # Crucial: Use native Urdu language for 100% phonetic audio capture
            temperature=0.0,
        )
        raw_urdu_text = transcription.text.strip()

        if not raw_urdu_text:
            return ""

        # Step 2: Convert native Urdu script into clean Roman Urdu
        roman_text = convert_urdu_to_roman(raw_urdu_text)
        return clean_roman_script(roman_text)

    except Exception as e:
        raise RuntimeError(f"Whisper Transcription Error: {e}")

# ============================
# 🧠 SESSION STATE
# ============================
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

# ============================
# 🖥️ UI INTERFACE
# ============================
st.title("🎤 High-Precision Speech to Text")
st.caption("Powered by Groq Whisper-Large-v3 & Llama-3.3 • Auto Soft-Voice Boosting")

st.info("Press Start, speak (even softly), and press Stop.")

st.subheader("🎤 Voice Input")
audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="whisper_mic",
)

if audio_output:
    audio_bytes = audio_output.get("bytes")
    if audio_bytes:
        with st.spinner("⚡ Processing & Transcribing with High Precision..."):
            try:
                # 1. Boost soft audio
                boosted_audio = normalize_and_prepare_audio(audio_bytes)
                # 2. Transcribe & Convert
                result_text = transcribe_with_whisper(boosted_audio)

                if result_text:
                    st.session_state.last_transcription = result_text
                    st.success("✅ Transcription Complete!")
                else:
                    st.warning("⚠️ No speech detected.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================
# 📝 DISPLAY OUTPUT
# ============================
st.divider()
st.subheader("📝 Transcribed Text")
if st.session_state.last_transcription:
    safe_text = html.escape(st.session_state.last_transcription)
    st.markdown(
        f"""
        <div style="
            padding: 18px; 
            border-radius: 10px; 
            background-color: #ffffff; 
            border: 1px solid #dcdcdc; 
            box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
        ">
            <div style="font-weight: bold; font-size: 14px; color: #555555; margin-bottom: 8px;">Result:</div>
            <div style="font-size: 18px; color: #111111; line-height: 1.5; font-weight: 500;">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Your transcription will appear here.")

st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🛑 Save Text", use_container_width=True):
        st.success("Text locked in session.")
with col2:
    if st.button("🗑️ Clear Text", use_container_width=True):
        st.session_state.last_transcription = ""
        st.rerun()
