import html
import io
import os
import re
import numpy as np
import requests
import scipy.io.wavfile as wav
import scipy.signal as signal
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

# ============================
# 🖥️ STREAMLIT PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Speech to Text",
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
    st.error("DEEPGRAM_API_KEY not found. Put it in .env or Streamlit Secrets.")
    st.stop()


# ============================
# 🎙️ DEEPGRAM CONFIGURATION
# ============================
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_LANGUAGE = "ur"  # Explicitly Urdu for accurate speech recognition
DEEPGRAM_TIMEOUT = 60
DEEPGRAM_CONFIDENCE_THRESHOLD = 0.35

DEEPGRAM_KEYTERMS = [
    "Python",
    "Streamlit",
    "Jupyter",
    "Matplotlib",
    "Plotly",
    "NumPy",
    "SciPy",
    "Deepgram",
    "AI",
    "machine learning",
    "deep learning",
    "API",
    "API key",
    "variable",
    "function",
    "class",
    "list",
    "dictionary",
    "tuple",
    "integer",
    "string",
    "float",
    "Flask",
    "FastAPI",
    "JavaScript",
    "HTML",
    "CSS",
]


# ============================
# 🧹 CLEAN SCRIPT FUNCTION
# ============================
def clean_text(text):
    """Basic cleanup for extra spaces and unwanted symbols."""
    if not text:
        return ""
    text = re.sub(r"[’'‘`\^\~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================
# 😮 NON-SPEECH SOUND EVENT DETECTION
# ============================
EVENT_LABEL_MAP = {
    "cough": "[coughing]",
    "laughter": "[laughing]",
    "baby laughter": "[laughing]",
    "giggle": "[laughing]",
    "snicker": "[laughing]",
    "belly laugh": "[laughing]",
    "chuckle, chortle": "[laughing]",
    "crying, sobbing": "[crying]",
    "baby cry, infant cry": "[crying]",
    "whimper": "[crying]",
    "screaming": "[screaming]",
    "clapping": "[clapping]",
    "applause": "[clapping]",
    "sneeze": "[sneezing]",
    "sigh": "[sighing]",
    "sniff": "[sniffing]",
    "throat clearing": "[clearing throat]",
    "whistling": "[whistling]",
    "breathing": "[breathing]",
    "wheeze": "[breathing]",
    "gasp": "[breathing]",
}
EVENT_CONFIDENCE_THRESHOLD = 0.20
BREATHING_CONFIDENCE_THRESHOLD = 0.12
BREATHING_LABELS = {"breathing", "wheeze", "gasp"}
EVENT_WINDOW_SECONDS = 1.5
PANNS_SAMPLE_RATE = 32000


@st.cache_resource(show_spinner=False)
def load_sound_event_model():
    try:
        from panns_inference import AudioTagging

        return AudioTagging(checkpoint_path=None, device="cpu")
    except Exception:
        return None


def detect_sound_events(audio_data, sample_rate):
    try:
        model = load_sound_event_model()
        if model is None:
            return []
    except Exception:
        return []

    try:
        import librosa
        from panns_inference import labels as audioset_labels

        audio_float = audio_data.astype(np.float32) / 32768.0
        if sample_rate != PANNS_SAMPLE_RATE:
            audio_float = librosa.resample(
                audio_float, orig_sr=sample_rate, target_sr=PANNS_SAMPLE_RATE
            )

        window_len = int(EVENT_WINDOW_SECONDS * PANNS_SAMPLE_RATE)
        total_len = len(audio_float)

        raw_events = []
        for start_sample in range(0, total_len, window_len):
            end_sample = min(start_sample + window_len, total_len)
            chunk = audio_float[start_sample:end_sample]
            if len(chunk) < PANNS_SAMPLE_RATE * 0.3:
                continue

            clipwise_output, _ = model.inference(chunk[None, :])
            probs = clipwise_output[0]

            for idx, prob in enumerate(probs):
                label_name = audioset_labels[idx].strip().lower()
                if label_name not in EVENT_LABEL_MAP:
                    continue

                threshold = (
                    BREATHING_CONFIDENCE_THRESHOLD
                    if label_name in BREATHING_LABELS
                    else EVENT_CONFIDENCE_THRESHOLD
                )
                if prob < threshold:
                    continue

                start_sec = start_sample / PANNS_SAMPLE_RATE
                end_sec = end_sample / PANNS_SAMPLE_RATE
                raw_events.append(
                    (start_sec, end_sec, EVENT_LABEL_MAP[label_name])
                )

        raw_events.sort(key=lambda e: e[0])
        merged = []
        for start_sec, end_sec, tag in raw_events:
            if (
                merged
                and merged[-1][2] == tag
                and start_sec <= merged[-1][1] + 0.1
            ):
                merged[-1] = (merged[-1][0], end_sec, tag)
            else:
                merged.append((start_sec, end_sec, tag))

        return merged
    except Exception:
        return []


# ============================
# 🎙️ DEEPGRAM TRANSCRIBE (ONLY DEEPGRAM)
# ============================
def transcribe_with_deepgram(processed_bytes, debug=False):
    params = [
        ("model", DEEPGRAM_MODEL),
        ("language", DEEPGRAM_LANGUAGE),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("utterances", "true"),
        ("numerals", "true"),
    ]

    for term in DEEPGRAM_KEYTERMS:
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
            data=processed_bytes,
            timeout=DEEPGRAM_TIMEOUT,
        )
    except requests.RequestException as e:
        if debug:
            st.exception(e)
        raise RuntimeError(f"Could not reach Deepgram: {e}") from e

    if response.status_code != 200:
        detail = response.text[:1200]
        raise RuntimeError(
            f"Deepgram API error {response.status_code}: {detail}"
        )

    try:
        data = response.json()
    except Exception as e:
        raise RuntimeError("Deepgram returned invalid JSON.") from e

    results = data.get("results", {})
    channels = results.get("channels", [])

    if not channels:
        return {"text": "", "confidence": 0.0, "raw": data}

    alternatives = channels[0].get("alternatives", [])

    if not alternatives:
        return {"text": "", "confidence": 0.0, "raw": data}

    alternative = alternatives[0]
    transcript = (alternative.get("transcript") or "").strip()
    confidence = float(alternative.get("confidence", 0.0) or 0.0)

    if not transcript:
        utterances = results.get("utterances") or []
        transcript = " ".join(
            (u.get("transcript") or "").strip()
            for u in utterances
            if (u.get("transcript") or "").strip()
        ).strip()

        if utterances:
            confidences = [
                float(u.get("confidence", 0.0) or 0.0)
                for u in utterances
                if u.get("transcript")
            ]
            if confidences:
                confidence = float(np.mean(confidences))

    return {
        "text": clean_text(transcript),
        "confidence": confidence,
        "raw": data,
    }


# ============================
# 🎚️ AUDIO PROCESSING CONFIGS
# ============================
MIN_RMS_ENERGY = 10.0
MIN_DURATION_SECONDS = 0.20
MAX_DURATION_SECONDS = 180
VAD_FRAME_MS = 30
MIN_SPEECH_SECONDS = 0.10
NOISE_FLOOR_PERCENTILE = 10
SPEECH_ABOVE_NOISE_FACTOR = 1.1
SPEECH_LOW_HZ = 50
SPEECH_HIGH_HZ = 8000


def bandpass_filter(
    audio_data, sample_rate, low_hz=SPEECH_LOW_HZ, high_hz=SPEECH_HIGH_HZ
):
    if len(audio_data) < 100:
        return audio_data
    nyquist = 0.5 * sample_rate
    low = max(0.001, low_hz / nyquist)
    high = min(0.99, high_hz / nyquist)

    if low >= high:
        return audio_data

    try:
        b, a = signal.butter(3, [low, high], btype="band")
        return signal.filtfilt(b, a, audio_data.astype(np.float64))
    except Exception:
        return audio_data


def normalize_audio(audio_data, target_peak=0.90):
    max_val = np.max(np.abs(audio_data))
    if max_val < 1e-8:
        return audio_data

    scale = target_peak * 32767.0 / max_val
    return audio_data * scale


def frame_energies(audio_data, sample_rate, frame_ms=VAD_FRAME_MS):
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    energies = []

    for start in range(0, len(audio_data), frame_len):
        chunk = audio_data[start : start + frame_len]
        if len(chunk) == 0:
            continue
        energies.append(
            float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        )

    return energies


def contains_real_speech(audio_data, sample_rate):
    if audio_data is None or len(audio_data) == 0:
        return False

    energies = frame_energies(audio_data, sample_rate)
    if not energies:
        return False

    noise_floor = np.percentile(energies, NOISE_FLOOR_PERCENTILE)
    dynamic_threshold = max(
        noise_floor * SPEECH_ABOVE_NOISE_FACTOR, MIN_RMS_ENERGY
    )

    speech_frame_count = sum(
        1 for energy in energies if energy > dynamic_threshold
    )
    speech_seconds = speech_frame_count * VAD_FRAME_MS / 1000.0

    return speech_seconds >= MIN_SPEECH_SECONDS


def process_audio_buffer(
    audio_bytes,
    enhance_audio=False,
    suppress_background=False,
    background_sensitivity=0.35,
    debug=False,
):
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)

        if sample_rate <= 0:
            raise ValueError("Invalid sample rate.")

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = audio_data.astype(np.float64)
        duration = len(audio_data) / float(sample_rate)

        if duration < MIN_DURATION_SECONDS:
            return None

        if duration > MAX_DURATION_SECONDS:
            audio_data = audio_data[: int(MAX_DURATION_SECONDS * sample_rate)]
            duration = len(audio_data) / float(sample_rate)

        if not contains_real_speech(audio_data, sample_rate):
            return None

        raw_mono_audio = audio_data.copy()
        processed_audio = audio_data.copy()

        if enhance_audio:
            processed_audio = bandpass_filter(processed_audio, sample_rate)
            try:
                import noisereduce as nr

                processed_audio = nr.reduce_noise(
                    y=processed_audio,
                    sr=sample_rate,
                    stationary=False,
                    prop_decrease=0.20,
                )
            except Exception:
                pass
            processed_audio = normalize_audio(processed_audio)

        if not contains_real_speech(processed_audio, sample_rate):
            processed_audio = raw_mono_audio.copy()

        processed_audio = np.clip(processed_audio, -32768, 32767).astype(
            np.int16
        )

        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, processed_audio)
        output_buffer.seek(0)

        return {
            "processed_bytes": output_buffer.read(),
            "raw_audio": raw_mono_audio,
            "sample_rate": int(sample_rate),
            "duration": float(duration),
        }

    except Exception as e:
        if debug:
            st.exception(e)
        return None


# ============================
# 🖥️ STREAMLIT UI HELPERS
# ============================
def load_css(file_path="style.css"):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                st.markdown(
                    f"<style>{f.read()}</style>", unsafe_allow_html=True
                )
        except Exception:
            pass


def load_html(file_path="index.html"):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                try:
                    st.html(content)
                except Exception:
                    st.markdown(content, unsafe_allow_html=True)
        except Exception:
            pass


# ============================
# 🧠 SESSION STATE
# ============================
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None
if "last_audio_duration" not in st.session_state:
    st.session_state.last_audio_duration = None
if "last_sample_rate" not in st.session_state:
    st.session_state.last_sample_rate = None

# ============================
# 🧩 PAGE CONTENT
# ============================
load_css("style.css")
load_html("index.html")
st.title("🎤 SPEECH TO TEXT")
st.caption("Deepgram Nova-3 • Direct Speech Processing")

st.info(
    "Record your voice or play audio. Keep Light audio enhancement OFF for songs/music."
)

# ============================
# 🎚️ AUDIO CONTROLS
# ============================
st.subheader("🎚️ Audio Controls")
col_a, col_b = st.columns(2)
with col_a:
    enhance_audio = st.checkbox(
        "✨ Light audio enhancement",
        value=False,
        help="Keep OFF for songs or speech with music.",
    )
with col_b:
    suppress_background = st.checkbox(
        "🔇 Reduce background noise",
        value=False,
    )

bg_sensitivity = 0.35

# ============================
# 🐞 DEBUG & SOUND EVENTS
# ============================
debug_mode = st.checkbox("🐞 Show real technical errors", value=False)
detect_events = st.checkbox(
    "😮 Detect sound events (laughter/breathing)", value=False
)

# ============================
# 🎤 MICROPHONE INPUT
# ============================
st.subheader("🎤 Voice Input")
st.write("Press Start, speak or play audio, then press Stop.")

audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="listener_mic",
)

# ============================
# 🧠 TRANSCRIPTION LOGIC
# ============================
if audio_output:
    audio_bytes = audio_output.get("bytes")
    if not audio_bytes:
        st.error("No audio data received.")
        st.stop()

    with st.spinner("⏳ Processing audio..."):
        result = process_audio_buffer(
            audio_bytes,
            enhance_audio=enhance_audio,
            suppress_background=suppress_background,
            background_sensitivity=bg_sensitivity,
            debug=debug_mode,
        )

    if result is None:
        st.warning(
            "⚠️ Audio too quiet or no clear speech detected. Try speaking closer to the mic."
        )
    else:
        processed_bytes = result["processed_bytes"]
        raw_audio = result["raw_audio"]
        sample_rate = result["sample_rate"]
        duration = result["duration"]

        st.session_state.last_audio_bytes = processed_bytes
        st.session_state.last_audio_duration = duration
        st.session_state.last_sample_rate = sample_rate

        with st.expander("🔧 Audio details"):
            st.write(f"Duration: {duration:.2f} seconds")
            st.write(f"Sample rate: {sample_rate} Hz")

        with st.expander("🔊 Listen to recording"):
            st.audio(audio_bytes, format="audio/wav")

        events = []
        if detect_events:
            with st.spinner("😮 Detecting non-speech sounds..."):
                events = detect_sound_events(raw_audio, sample_rate)

        with st.spinner("⚡ Transcribing with Deepgram Nova-3..."):
            try:
                transcription_result = transcribe_with_deepgram(
                    processed_bytes, debug=debug_mode
                )
                text_from_voice = transcription_result["text"].strip()
                confidence = float(transcription_result["confidence"])

                if events:
                    event_tags = " ".join(event[2] for event in events)
                    if event_tags:
                        text_from_voice = f"{text_from_voice} {event_tags}".strip()

                if text_from_voice and confidence >= 0.35:
                    st.session_state.last_transcription = text_from_voice
                    st.session_state.last_confidence = confidence
                    st.success("✅ Complete!")
                elif text_from_voice and confidence < 0.35:
                    st.warning("⚠️ Low confidence transcription. Please speak clearly.")
                    st.session_state.last_transcription = text_from_voice
                    st.session_state.last_confidence = confidence
                else:
                    st.warning("⚠️ No clear speech detected.")

            except Exception as e:
                st.error(f"❌ Transcription error: {e}")
                if debug_mode:
                    st.exception(e)

# ============================
# 📝 DISPLAY OUTPUT
# ============================
st.divider()
st.subheader("📝 Transcribed Text")
if st.session_state.last_transcription:
    safe_text = html.escape(st.session_state.last_transcription)
    st.markdown(
        f"""
        <div style="padding: 18px; border-radius: 10px; background-color: #1e1e2e; border: 1px solid #45475a; margin-top: 10px;">
            <div style="font-weight: bold; color: #89b4fa; margin-bottom: 8px; font-size: 1.1em;">Result:</div>
            <div style="font-size: 1.2em; color: #cdd6f4; font-weight: 500; line-height: 1.5;">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_confidence is not None:
        st.caption(
            f"Deepgram confidence score: {st.session_state.last_confidence:.2f}"
        )
else:
    st.info("Your transcription will appear here.")

# ============================
# 🛠️ CONTROLS
# ============================
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🛑 Lock Text", use_container_width=True):
        if st.session_state.last_transcription:
            st.success("Text saved in current session.")
        else:
            st.warning("No text available to save.")
with col2:
    if st.button("🗑️ Clear Text", use_container_width=True):
        st.session_state.last_transcription = ""
        st.session_state.last_confidence = None
        st.session_state.last_audio_bytes = None
        st.rerun()
