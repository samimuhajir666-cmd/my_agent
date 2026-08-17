import io
import os
import re
import html
import time
import wave
import threading
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

st.set_page_config(page_title="Speech to Text", page_icon="🎤", layout="centered")

load_dotenv()

# ============================
# 🔑 API KEY
# ============================
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    try:
        ASSEMBLYAI_API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY")
    except Exception:
        ASSEMBLYAI_API_KEY = None

if not ASSEMBLYAI_API_KEY:
    st.error("ASSEMBLYAI_API_KEY not found. Add it to .env or Streamlit Secrets.")
    st.stop()

# ============================
# 🔤 HELPERS
# ============================
def force_roman_script(text):
    if not text:
        return text
    if bool(re.search(r'[^\x00-\x7F]', text)):
        return unidecode(text)
    return text

# ============================
# 🎚️ AUDIO VALIDATION
# ============================
MIN_RMS_ENERGY = 35.0
MIN_DURATION_SECONDS = 0.45
MAX_DURATION_SECONDS = 120
VAD_FRAME_MS = 30
MIN_SPEECH_SECONDS = 0.20
NOISE_FLOOR_PERCENTILE = 10
SPEECH_ABOVE_NOISE_FACTOR = 2.0


def contains_real_speech(audio_data, sample_rate):
    if audio_data is None or len(audio_data) == 0:
        return False
    frame_len = max(1, int(sample_rate * VAD_FRAME_MS / 1000))
    energies = [
        float(np.sqrt(np.mean(audio_data[s:s + frame_len].astype(np.float64) ** 2)))
        for s in range(0, len(audio_data), frame_len)
        if len(audio_data[s:s + frame_len]) > 0
    ]
    if not energies:
        return False
    noise_floor = np.percentile(energies, NOISE_FLOOR_PERCENTILE)
    threshold = max(noise_floor * SPEECH_ABOVE_NOISE_FACTOR, MIN_RMS_ENERGY)
    speech_secs = sum(1 for e in energies if e > threshold) * VAD_FRAME_MS / 1000.0
    return speech_secs >= MIN_SPEECH_SECONDS


def process_audio(audio_bytes, debug=False):
    try:
        sample_rate, audio_data = wav.read(io.BytesIO(audio_bytes))

        if sample_rate <= 0:
            raise ValueError("Invalid sample rate.")

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = audio_data.astype(np.float64)
        duration = len(audio_data) / float(sample_rate)

        if duration < MIN_DURATION_SECONDS:
            return None
        if not contains_real_speech(audio_data, sample_rate):
            return None
        if duration > MAX_DURATION_SECONDS:
            audio_data = audio_data[:int(MAX_DURATION_SECONDS * sample_rate)]
            duration = len(audio_data) / float(sample_rate)

        audio_out = np.clip(audio_data, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        wav.write(buf, sample_rate, audio_out)
        buf.seek(0)

        return {
            "processed_bytes": buf.read(),
            "sample_rate": int(sample_rate),
            "duration": float(duration),
        }
    except Exception as e:
        if debug:
            st.exception(e)
        return None

# ============================
# 🎙️ TRANSCRIBE
# ============================
def transcribe(processed_bytes, sample_rate, mic_type, debug=False):
    collected_turns = []
    done_event = threading.Event()

    def on_turn(client: StreamingClient, event: TurnEvent):
        if event.transcript and event.end_of_turn:
            collected_turns.append(event.transcript.strip())

    def on_terminated(client: StreamingClient, event: TerminationEvent):
        done_event.set()

    def on_error(client: StreamingClient, error: StreamingError):
        if debug:
            st.warning(f"AssemblyAI error: {error}")
        done_event.set()

    client = StreamingClient(
        StreamingClientOptions(api_key=ASSEMBLYAI_API_KEY, terminate_timeout=10.0)
    )
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    client.connect(
        StreamingParameters(
            sample_rate=sample_rate,
            speech_model="universal-3-5-pro",
            format_turns=True,
            language_codes=["en", "ur"],
            prompt="Conversation in English and Roman Urdu between two people.",
            keyterms_prompt=[
                "Python", "Streamlit", "API", "AI", "machine learning",
                "NumPy", "SciPy", "AssemblyAI", "function", "variable",
                "class", "dictionary", "integer", "string", "Flask",
                "FastAPI", "JavaScript", "HTML", "CSS",
            ],
            voice_focus=mic_type,
            voice_focus_threshold=0.7,
            end_of_turn_confidence_threshold=0.6,
        )
    )

    CHUNK_DURATION = 0.1
    with wave.open(io.BytesIO(processed_bytes), "rb") as wf:
        frames_per_chunk = int(wf.getframerate() * CHUNK_DURATION)
        start_time = time.monotonic()
        chunks_sent = 0
        while True:
            frames = wf.readframes(frames_per_chunk)
            if not frames:
                break
            client.send_audio(frames)
            chunks_sent += 1
            sleep_for = start_time + (chunks_sent * CHUNK_DURATION) - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    client.disconnect(terminate=True)
    done_event.wait(timeout=15)

    return force_roman_script(" ".join(collected_turns).strip())

# ============================
# 🧠 SESSION STATE
# ============================
if "transcription" not in st.session_state:
    st.session_state.transcription = ""

# ============================
# 🧩 UI
# ============================
st.title("🎤 Speech to Text")
st.caption("AssemblyAI Universal-3.5 Pro • Human Ears Mode • English + Roman Urdu")

st.subheader("👂 Human Ears Mode")
mic_type = st.radio(
    "Microphone type",
    options=["far-field", "near-field"],
    index=0,
    horizontal=True,
    help="far-field = laptop/room mic.   near-field = headset/phone held close.",
)

debug_mode = st.checkbox("🐞 Debug mode", value=False)

st.divider()
st.subheader("🎤 Voice Input")
st.write("Press Start, speak, then press Stop.")

audio_output = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="mic",
)

# ============================
# 🧠 TRANSCRIPTION
# ============================
if audio_output:
    audio_bytes = audio_output.get("bytes")

    if not audio_bytes:
        st.error("No audio received.")
        st.stop()

    with st.spinner("Checking audio..."):
        result = process_audio(audio_bytes, debug=debug_mode)

    if result is None:
        st.warning("Recording too quiet or too short. Speak closer to the mic.")
    else:
        st.audio(audio_bytes, format="audio/wav")

        with st.spinner("Transcribing..."):
            try:
                text = transcribe(
                    result["processed_bytes"],
                    result["sample_rate"],
                    mic_type=mic_type,
                    debug=debug_mode,
                )
                if text:
                    st.session_state.transcription = text
                    st.success("✅ Done!")
                else:
                    st.warning("No speech detected. Try switching mic type.")
            except Exception as e:
                st.error(f"Error: {e}")
                if debug_mode:
                    st.exception(e)

# ============================
# 📝 OUTPUT
# ============================
st.divider()
st.subheader("📝 Transcribed Text")

if st.session_state.transcription:
    st.markdown(
        f"""
        <div style="background:#1e1e2e;padding:20px;border-radius:10px;
                    border-left:4px solid #7c3aed;">
            <p style="color:#cdd6f4;font-size:1.1rem;margin:0;">
                {html.escape(st.session_state.transcription)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Your transcription will appear here.")

st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.button("🛑 Lock Text", use_container_width=True):
        if st.session_state.transcription:
            st.success("Text locked.")
        else:
            st.warning("Nothing to lock.")

with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.transcription = ""
        st.rerun()
