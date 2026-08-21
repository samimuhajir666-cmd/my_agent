import html
import io
import os
import re
import numpy as np
import requests
import scipy.io.wavfile as wav
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

# ============================
# 🖥️ STREAMLIT PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Speech to Text (Roman Urdu)",
    page_icon="🎤",
    layout="centered",
)
load_dotenv()

# ============================
# 🔑 API KEYS CONFIGURATION
# ============================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    try:
        DEEPGRAM_API_KEY = st.secrets.get("DEEPGRAM_API_KEY")
    except Exception:
        DEEPGRAM_API_KEY = None

if not DEEPGRAM_API_KEY:
    st.error("DEEPGRAM_API_KEY nahi mili. .env file ya Streamlit Secrets mein add karein.")
    st.stop()

# ============================
# 🎙️ DEEPGRAM CONFIGURATION
# ============================
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"

# Roman Urdu Keywords taake Deepgram phonetics fast aur sahi pakde
ROMAN_KEYTERMS = [
    "aap", "kaise", "hain", "main", "theek", "hun", "kya", "kar", "rahe", "ho",
    "shukriya", "haan", "nahi", "bhai", "sahi", "ho", "gaya", "karo", "bolo",
    "suno", "raha", "rahi", "chal", "rha", "rhi", "mera", "meri", "sir"
]

# Common English sound-alike misspellings ko Roman Urdu mein auto-fix karne ka map
PHONETIC_FIXES = {
    r"\bup\b": "aap",
    r"\bkese\b": "kaise",
    r"\bthik\b": "theek",
    r"\bhen\b": "hain",
    r"\bme\b": "main",
    r"\bkia\b": "kya",
    r"\bhoon\b": "hun",
    r"\bnhn\b": "nahi",
    r"\brha\b": "raha",
    r"\brhi\b": "rahi",
}

# ============================
# 🧹 TEXT CLEANUP & FIXES
# ============================
def clean_and_fix_transcript(text):
    if not text:
        return ""
    
    # 1. Phonetic dictionary fixes
    for wrong_pattern, correct_word in PHONETIC_FIXES.items():
        text = re.sub(wrong_pattern, correct_word, text, flags=re.IGNORECASE)

    # 2. Urdu Script Remove (In case koi Urdu char reh jaye)
    text = re.sub(r"[\u0600-\u06FF\u0900-\u097F]", "", text)

    # 3. Extra spaces cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text.capitalize()

# ============================
# 🎙️ DEEPGRAM TRANSCRIBE ENGINE
# ============================
def transcribe_audio(audio_bytes):
    # Setup fast model with language English for Latin character transcription
    params = [
        ("model", DEEPGRAM_MODEL),
        ("language", "en"),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("utterances", "true"),
    ]

    # Keyword boosting
    for term in ROMAN_KEYTERMS:
        params.append(("keyterm", term))

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }

    try:
        response = requests.post(
            DEEPGRAM_API_URL,
            params=params,
            headers=headers,
            data=audio_bytes,
            timeout=25,
        )
    except Exception as e:
        raise RuntimeError(f"Connection Error: {e}")

    if response.status_code != 200:
        raise RuntimeError(f"Deepgram Error ({response.status_code}): {response.text}")

    data = response.json()
    results = data.get("results", {})
    channels = results.get("channels", [])

    if not channels:
        return "", 0.0

    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return "", 0.0

    raw_transcript = alternatives[0].get("transcript", "").strip()
    confidence = float(alternatives[0].get("confidence", 0.0))

    final_transcript = clean_and_fix_transcript(raw_transcript)
    return final_transcript, confidence

# ============================
# 🧠 SESSION STATE
# ============================
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

# ============================
# 🖥️ UI INTERFACE
# ============================
st.title("🎤 Speech to Text (Roman Urdu)")
st.caption("Powered by Deepgram Nova-3")

st.write("Niche button par click karke bolein aur Stop karein:")

# Microphone Recorder
audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="mic_input"
)

# Audio Processing
if audio_output and "bytes" in audio_output:
    audio_bytes = audio_output["bytes"]
    
    if len(audio_bytes) > 0:
        with st.spinner("⚡ Transcribing audio..."):
            try:
                roman_text, confidence = transcribe_audio(audio_bytes)
                
                if roman_text:
                    st.session_state.last_transcription = roman_text
                    st.session_state.last_confidence = confidence
                    st.success("✅ Done!")
                else:
                    st.warning("⚠️ Koi awaaz detect nahi hui. Meharbani karke thoda clear aur mic ke paas bolein.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Display Result
st.divider()
st.subheader("📝 Transcribed Text (Roman Script)")

if st.session_state.last_transcription:
    safe_text = html.escape(st.session_state.last_transcription)
    st.markdown(
        f"""
        <div style="padding: 18px; border-radius: 10px; background-color: #1e1e2e; border: 1px solid #45475a; margin-top: 10px;">
            <div style="font-weight: bold; color: #89b4fa; margin-bottom: 8px; font-size: 1.1em;">Result:</div>
            <div style="font-size: 1.3em; color: #cdd6f4; font-weight: 500;">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.last_confidence is not None:
        st.caption(f"📊 Confidence Score: {st.session_state.last_confidence * 100:.1f}%")
else:
    st.info("Aap ki boli hui awaaz yahan show hogi.")

st.divider()
if st.button("🗑️ Clear Output", use_container_width=True):
    st.session_state.last_transcription = ""
    st.session_state.last_confidence = None
    st.rerun()
