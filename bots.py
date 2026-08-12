import io
import os
import re
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode

load_dotenv()

# ============================
# 🔑 API KEY INITIALIZATION
# ============================
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

if not API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it in .env or Streamlit Secrets.")
    st.stop()

try:
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    st.error(f"Could not initialize OpenAI client: {e}")
    st.stop()

# ============================
# 🎯 MODEL SETTINGS
# ============================
STT_MODEL = "gpt-4o-transcribe-diarize"

# NO enrollment, no fixed "admin" voice file. Every recording is diarized
# fresh -- the model labels whoever it hears as Speaker A, Speaker B, etc.
# We show all of it, clearly separated by speaker, and the person reading
# it decides what's relevant. This avoids the fragility of assuming there's
# always exactly one fixed "admin" voice across every use of the app.

MIN_DURATION_SECONDS = 0.8


def force_roman_script(text):
    """Any non-Latin character gets converted to its closest Roman-letter
    form. gpt-4o-transcribe-diarize doesn't support a prompt field to bias
    script, so this safety net matters more here than before."""
    if not text:
        return text
    has_non_ascii = bool(re.search(r'[^\x00-\x7F]', text))
    if not has_non_ascii:
        return text
    return unidecode(text)


def get_audio_duration(audio_bytes):
    sample_rate, audio_data = wav.read(io.BytesIO(audio_bytes))
    return len(audio_data) / float(sample_rate)


def group_by_speaker(segments):
    """
    Turns a flat list of diarized segments into readable, speaker-labeled
    blocks -- consecutive segments from the same speaker are merged so the
    output reads like a real conversation transcript, not a choppy list.
    Returns a list of (speaker_label, combined_text) in chronological order.
    """
    blocks = []
    for seg in segments:
        speaker = getattr(seg, "speaker", None) or "Unknown"
        text = getattr(seg, "text", None)
        if not text:
            continue
        clean_text = force_roman_script(text.strip())
        if blocks and blocks[-1][0] == speaker:
            blocks[-1] = (speaker, blocks[-1][1] + " " + clean_text)
        else:
            blocks.append((speaker, clean_text))
    return blocks


# ============================
# 🖥️ STREAMLIT UI
# ============================
st.set_page_config(page_title="Speech to Text", page_icon="🎤", layout="centered")

if "last_transcription_blocks" not in st.session_state:
    st.session_state.last_transcription_blocks = []

st.title("🎤 Voice Input")
st.caption("Har awaz sunta hai, speaker ke hisaab se label karta hai — tum decide karo kaunsi zaroori hai.")

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

    try:
        duration = get_audio_duration(audio_bytes)
    except Exception:
        st.error("Could not read audio file.")
        st.stop()

    if duration < MIN_DURATION_SECONDS:
        st.warning("⚠️ Recording bohot chhoti hai, phir se try karein.")
    else:
        with st.spinner("⚡ Transcribing + identifying speakers..."):
            try:
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "recording.wav"

                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model=STT_MODEL,
                    response_format="diarized_json",
                    chunking_strategy="auto",
                )

                segments = getattr(transcription, "segments", None) or []
                blocks = group_by_speaker(segments)

                if blocks:
                    st.session_state.last_transcription_blocks = blocks
                    st.success(f"✅ Complete! {len(set(b[0] for b in blocks))} speaker(s) detected.")
                else:
                    st.warning("⚠️ Could not detect clear speech.")

            except Exception as e:
                st.error(f"Transcription error: {e}")

# ============================
# 📝 DISPLAY OUTPUT -- speaker-labeled, like a real transcript
# ============================
SPEAKER_COLORS = ["#ff4b4b", "#4b9eff", "#4bff88", "#ffb84b", "#c94bff"]

if st.session_state.last_transcription_blocks:
    st.markdown("### 📝 Transcript")

    speaker_order = []
    for speaker, _ in st.session_state.last_transcription_blocks:
        if speaker not in speaker_order:
            speaker_order.append(speaker)
    color_map = {sp: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, sp in enumerate(speaker_order)}

    for speaker, text in st.session_state.last_transcription_blocks:
        color = color_map.get(speaker, "#888888")
        st.markdown(
            f"""
            <div style="background-color:#1e2530; padding:12px; border-radius:8px;
                        border-left:4px solid {color}; margin-bottom:8px;">
                <div style="color:{color}; font-weight:bold; font-size:13px; margin-bottom:4px;">
                    {speaker}
                </div>
                <div>{text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # Plain combined text, useful for copy/lock
    full_text = "\n".join(f"{speaker}: {text}" for speaker, text in st.session_state.last_transcription_blocks)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🛑 Lock Text", use_container_width=True):
            st.success("Text saved.")
    with col2:
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state.last_transcription_blocks = []
            st.rerun()
