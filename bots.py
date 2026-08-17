import io
import os
import re
import html
import requests
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import noisereduce as nr
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode

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
# 🔑 DEEPGRAM API KEY
# ============================
# Use .env or Streamlit Secrets.
# Do NOT paste your API key directly into this file.

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    try:
        DEEPGRAM_API_KEY = st.secrets.get("DEEPGRAM_API_KEY")
    except Exception:
        DEEPGRAM_API_KEY = None

if not DEEPGRAM_API_KEY:
    st.error(
        "DEEPGRAM_API_KEY not found. "
        "Put it in .env or Streamlit Secrets."
    )
    st.stop()

# ============================
# 🎙️ DEEPGRAM CONFIGURATION
# ============================
# REST is used instead of the Python SDK. This avoids SDK-version
# conflicts such as: BaseModel.init() positional-argument errors.

DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_LANGUAGE = "multi"
DEEPGRAM_TIMEOUT = 60
DEEPGRAM_CONFIDENCE_THRESHOLD = 0.35

DEEPGRAM_KEYTERMS = [
    "Python", "Streamlit", "Jupyter", "Matplotlib", "Plotly",
    "NumPy", "SciPy", "Deepgram", "AI", "machine learning",
    "deep learning", "API", "API key", "variable", "function",
    "class", "list", "dictionary", "tuple", "integer", "string",
    "float", "Flask", "FastAPI", "JavaScript", "HTML", "CSS",
]

SYSTEM_PROMPT = (
    "Transcribe the speaker faithfully. The speaker may switch between "
    "English and Roman Urdu. Keep the original spoken words and meaning. "
    "Do not answer questions. Do not summarize. Do not invent missing words. "
    "Keep technical terms such as Python, Streamlit, Jupyter, Matplotlib, "
    "Plotly, NumPy, API, AI and machine learning."
)

def force_roman_script(text):
    if not text:
        return text
    has_non_ascii = bool(re.search(r'[^\x00-\x7F]', text))
    if not has_non_ascii:
        return text
    return unidecode(text)

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
    "loudness": "[louding]",
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
                raw_events.append((start_sec, end_sec, EVENT_LABEL_MAP[label_name]))

        raw_events.sort(key=lambda e: e[0])
        merged = []
        for start_sec, end_sec, tag in raw_events:
            if merged and merged[-1][2] == tag and start_sec <= merged[-1][1] + 0.1:
                merged[-1] = (merged[-1][0], end_sec, tag)
            else:
                merged.append((start_sec, end_sec, tag))

        return merged
    except Exception:
        return []

# ============================
# 🎙️ DEEPGRAM TRANSCRIBE
# ============================

def transcribe_with_deepgram(processed_bytes, debug=False):
    """
    Send WAV bytes directly to Deepgram's REST API.

    This intentionally does NOT use the Deepgram Python SDK. Different
    installed SDK versions caused the previous BaseModel positional-
    argument error. The REST endpoint is stable and accepts WAV bytes.
    """

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

    # Deepgram can return utterances. We use them only as a fallback;
    # filtering them by confidence was one reason words were being dropped.
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
        "text": force_roman_script(transcript),
        "confidence": confidence,
        "raw": data,
    }

# ============================
# 🎚️ AUDIO PROCESSING & CONFIGS
# ============================

# These settings are intentionally conservative. The goal is to reject
# only actual silence, not accidentally delete quiet words.

MIN_RMS_ENERGY = 35.0
MIN_DURATION_SECONDS = 0.45
MAX_DURATION_SECONDS = 120
VAD_FRAME_MS = 30
MIN_SPEECH_SECONDS = 0.20
NOISE_FLOOR_PERCENTILE = 10
SPEECH_ABOVE_NOISE_FACTOR = 2.0
SPEECH_LOW_HZ = 70
SPEECH_HIGH_HZ = 7600

DOMINANT_FRAME_MS = 100
DOMINANT_WINDOW_SECONDS = 2.0
DOMINANT_MIN_GAIN = 0.65

def bandpass_filter(
    audio_data,
    sample_rate,
    low_hz=SPEECH_LOW_HZ,
    high_hz=SPEECH_HIGH_HZ,
):
    """Gentle speech filter. If filtering fails, keep original audio."""
    if len(audio_data) < 100:
        return audio_data

    nyquist = 0.5 * sample_rate
    low = max(0.001, low_hz / nyquist)
    high = min(0.99, high_hz / nyquist)

    if low >= high:
        return audio_data

    try:
        b, a = signal.butter(3, [low, high], btype="band")
        return signal.filtfilt(
            b,
            a,
            audio_data.astype(np.float64),
        )
    except Exception:
        return audio_data

def attenuate_background_speakers(
    audio_data,
    sample_rate,
    sensitivity,
):
    """
    Optional and deliberately gentle.

    This does NOT know who is speaking. It only reduces some quieter
    frames relative to the local loudest frame. It can hurt accuracy if
    another person is louder than the main speaker, so it is OFF by default.
    """
    try:
        frame_len = max(
            1,
            int(sample_rate * DOMINANT_FRAME_MS / 1000),
        )
        n_frames = int(np.ceil(len(audio_data) / frame_len))

        if n_frames <= 1:
            return audio_data

        frame_energy = np.zeros(n_frames, dtype=np.float64)

        for i in range(n_frames):
            chunk = audio_data[
                i * frame_len : (i + 1) * frame_len
            ]
            if len(chunk) > 0:
                frame_energy[i] = np.sqrt(
                    np.mean(chunk.astype(np.float64) ** 2)
                )

        window_frames = max(
            1,
            int(
                DOMINANT_WINDOW_SECONDS
                * 1000
                / DOMINANT_FRAME_MS
            ),
        )

        local_peak = np.zeros(
            n_frames,
            dtype=np.float64,
        )

        for i in range(n_frames):
            start = max(
                0,
                i - window_frames // 2,
            )
            end = min(
                n_frames,
                i + window_frames // 2 + 1,
            )
            local_peak[i] = np.max(
                frame_energy[start:end]
            )

        sensitivity = float(
            np.clip(sensitivity, 0.05, 0.80)
        )

        quieter = frame_energy < (
            sensitivity * local_peak
        )

        gain = np.ones(
            n_frames,
            dtype=np.float64,
        )
        gain[quieter] = DOMINANT_MIN_GAIN

        if len(gain) > 5:
            smoothing_window = signal.get_window(
                "hamming",
                5,
            )
            smoothing_window /= np.sum(
                smoothing_window
            )
            gain = signal.convolve(
                gain,
                smoothing_window,
                mode="same",
            )
            gain = np.clip(
                gain,
                DOMINANT_MIN_GAIN,
                1.0,
            )

        output = audio_data.astype(
            np.float64
        ).copy()

        for i in range(n_frames):
            start = i * frame_len
            end = min(
                len(audio_data),
                start + frame_len,
            )
            output[start:end] *= gain[i]

        return output

    except Exception:
        return audio_data

def normalize_audio(
    audio_data,
    target_peak=0.90,
):
    max_val = np.max(
        np.abs(audio_data)
    )

    if max_val < 1e-8:
        return audio_data

    scale = (
        target_peak
        * 32767.0
        / max_val
    )

    return audio_data * scale

def frame_energies(
    audio_data,
    sample_rate,
    frame_ms=VAD_FRAME_MS,
):
    frame_len = max(
        1,
        int(sample_rate * frame_ms / 1000),
    )

    energies = []

    for start in range(
        0,
        len(audio_data),
        frame_len,
    ):
        chunk = audio_data[
            start : start + frame_len
        ]

        if len(chunk) == 0:
            continue

        energies.append(
            float(
                np.sqrt(
                    np.mean(
                        chunk.astype(np.float64) ** 2
                    )
                )
            )
        )

    return energies

def contains_real_speech(
    audio_data,
    sample_rate,
):
    """
    Lightweight silence detector.

    It is intentionally permissive so quiet words are not removed.
    It is used only to reject recordings with essentially no audio.
    """
    if audio_data is None or len(audio_data) == 0:
        return False

    energies = frame_energies(
        audio_data,
        sample_rate,
    )

    if not energies:
        return False

    noise_floor = np.percentile(
        energies,
        NOISE_FLOOR_PERCENTILE,
    )

    dynamic_threshold = max(
        noise_floor * SPEECH_ABOVE_NOISE_FACTOR,
        MIN_RMS_ENERGY,
    )

    speech_frame_count = sum(
        1
        for energy in energies
        if energy > dynamic_threshold
    )

    speech_seconds = (
        speech_frame_count
        * VAD_FRAME_MS
        / 1000.0
    )

    return speech_seconds >= MIN_SPEECH_SECONDS

def process_audio_buffer(
    audio_bytes,
    enhance_audio=False,
    suppress_background=False,
    background_sensitivity=0.35,
    debug=False,
):
    """
    Process one recording.

    Returns a dictionary containing processed bytes, raw mono audio,
    sample rate and duration. Returns None only for invalid/silent audio.
    """
    try:
        audio_file = io.BytesIO(
            audio_bytes
        )

        sample_rate, audio_data = wav.read(
            audio_file
        )

        if sample_rate <= 0:
            raise ValueError(
                "Invalid sample rate."
            )

        if len(audio_data.shape) > 1:
            audio_data = np.mean(
                audio_data,
                axis=1,
            )

        audio_data = audio_data.astype(
            np.float64
        )

        duration = len(audio_data) / float(
            sample_rate
        )

        if duration < MIN_DURATION_SECONDS:
            return None

        if duration > MAX_DURATION_SECONDS:
            audio_data = audio_data[
                : int(
                    MAX_DURATION_SECONDS
                    * sample_rate
                )
            ]
            duration = len(audio_data) / float(
                sample_rate
            )

        # Reject only recordings that are effectively silent.
        if not contains_real_speech(
            audio_data,
            sample_rate,
        ):
            return None

        raw_mono_audio = audio_data.copy()
        processed_audio = audio_data.copy()

        # Optional background suppression.
        if suppress_background:
            processed_audio = attenuate_background_speakers(
                processed_audio,
                sample_rate,
                background_sensitivity,
            )

        # Optional enhancement. OFF is recommended initially because
        # aggressive filters can make STT worse rather than better.
        if enhance_audio:
            processed_audio = bandpass_filter(
                processed_audio,
                sample_rate,
            )

            try:
                processed_audio = nr.reduce_noise(
                    y=processed_audio,
                    sr=sample_rate,
                    stationary=False,
                    prop_decrease=0.20,
                )
            except Exception:
                pass

            processed_audio = normalize_audio(
                processed_audio
            )

        # Never send a damaged/empty result if processing went wrong.
        if not contains_real_speech(
            processed_audio,
            sample_rate,
        ):
            processed_audio = raw_mono_audio.copy()

        processed_audio = np.clip(
            processed_audio,
            -32768,
            32767,
        ).astype(np.int16)

        output_buffer = io.BytesIO()

        wav.write(
            output_buffer,
            sample_rate,
            processed_audio,
        )

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
            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as f:
                st.markdown(
                    f"<style>{f.read()}</style>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

def load_html(file_path="index.html"):
    if os.path.exists(file_path):
        try:
            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as f:
                content = f.read()

            try:
                st.html(content)
            except Exception:
                st.markdown(
                    content,
                    unsafe_allow_html=True,
                )
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
st.caption(
    "Deepgram Nova-3 • English + Roman Urdu • Technical vocabulary"
)
st.info(
    "Record your voice, stop the recording, and the app will transcribe it. "
    "First test with audio enhancement OFF."
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
        help=(
            "Keep OFF for your first tests. "
            "Heavy processing can remove speech details."
        ),
    )

with col_b:
    suppress_background = st.checkbox(
        "🔇 Reduce background speakers",
        value=False,
        help=(
            "Keep OFF unless another person is talking nearby. "
            "This is not speaker identification."
        ),
    )

if suppress_background:
    bg_sensitivity = st.slider(
        "Background suppression strength",
        min_value=0.05,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help=(
            "If your own words disappear, lower this value or turn "
            "background suppression OFF."
        ),
    )
else:
    bg_sensitivity = 0.35

# ============================
# 🐞 DEBUG
# ============================

debug_mode = st.checkbox(
    "🐞 Show real technical errors",
    value=False,
)

# ============================
# 😮 OPTIONAL SOUND EVENTS
# ============================

detect_events = st.checkbox(
    "😮 Detect coughing/laughing/breathing",
    value=False,
    help=(
        "Optional feature. It can require PANNs/librosa and may download "
        "a large model. It is OFF by default so it cannot interfere with STT."
    ),
)

# ============================
# 🎤 MICROPHONE
# ============================

st.subheader("🎤 Voice Input")
st.write(
    "Press Start, speak normally, then press Stop."
)

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

    # ----------------------------------------
    # CHECK / PROCESS AUDIO
    # ----------------------------------------
    with st.spinner("⏳ Checking audio..."):
        result = process_audio_buffer(
            audio_bytes,
            enhance_audio=enhance_audio,
            suppress_background=suppress_background,
            background_sensitivity=bg_sensitivity,
            debug=debug_mode,
        )

    if result is None:
        st.warning(
            "⚠️ The recording was too quiet, too short, or contained "
            "no detectable speech. Speak closer to the microphone and try again."
        )
    else:
        processed_bytes = result["processed_bytes"]
        raw_audio = result["raw_audio"]
        sample_rate = result["sample_rate"]
        duration = result["duration"]

        st.session_state.last_audio_bytes = processed_bytes
        st.session_state.last_audio_duration = duration
        st.session_state.last_sample_rate = sample_rate

        # ----------------------------------------
        # AUDIO INFORMATION
        # ----------------------------------------
        with st.expander("🔧 Audio information"):
            st.write(f"Duration: {duration:.2f} seconds")
            st.write(f"Sample rate: {sample_rate} Hz")
            st.write(f"Original size: {len(audio_bytes):,} bytes")
            st.write(f"Processed size: {len(processed_bytes):,} bytes")
            st.write(
                "Enhancement: "
                f"{'ON' if enhance_audio else 'OFF'}"
            )
            st.write(
                "Background suppression: "
                f"{'ON' if suppress_background else 'OFF'}"
            )

        # ----------------------------------------
        # PLAYBACK: IMPORTANT DEBUG STEP
        # ----------------------------------------
        with st.expander("🔊 Listen to the recording"):
            st.audio(
                audio_bytes,
                format="audio/wav",
            )

        # ----------------------------------------
        # OPTIONAL SOUND EVENTS
        # ----------------------------------------
        events = []

        if detect_events:
            with st.spinner(
                "😮 Checking optional sound events..."
            ):
                events = detect_sound_events(
                    raw_audio,
                    sample_rate,
                )

        # ----------------------------------------
        # DEEPGRAM TRANSCRIPTION
        # ----------------------------------------
        with st.spinner(
            "⚡ Transcribing with Deepgram Nova-3..."
        ):
            try:
                transcription_result = transcribe_with_deepgram(
                    processed_bytes,
                    debug=debug_mode,
                )

                text_from_voice = (
                    transcription_result["text"]
                    .strip()
                )

                confidence = float(
                    transcription_result["confidence"]
                )

                # Sound events are added only when the user explicitly
                # turns the feature on.
                if events:
                    event_tags = " ".join(
                        event[2]
                        for event in events
                    )
                    if event_tags:
                        text_from_voice = (
                            f"{text_from_voice} {event_tags}"
                        ).strip()

                if text_from_voice:
                    st.session_state.last_transcription = (
                        text_from_voice
                    )
                    st.session_state.last_confidence = (
                        confidence
                    )
                    st.success("✅ Complete!")
                else:
                    st.warning(
                        "⚠️ Deepgram did not return recognizable speech."
                    )

            except Exception as e:
                st.error(
                    f"❌ Transcription error: {e}"
                )

                if debug_mode:
                    st.exception(e)

# ============================
# 📝 DISPLAY OUTPUT
# ============================

st.divider()
st.subheader("📝 Transcribed Text")

if st.session_state.last_transcription:
    safe_text = html.escape(
        st.session_state.last_transcription
    )

    st.markdown(
        f"""
        <div class="output-card">
            <div class="output-title">Result:</div>
            <div class="output-text">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_confidence is not None:
        st.caption(
            "Deepgram confidence: "
            f"{st.session_state.last_confidence:.2f}"
        )
else:
    st.info(
        "Your transcription will appear here."
    )

# ============================
# 🛠️ CONTROLS
# ============================

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button(
        "🛑 Lock Text",
        use_container_width=True,
    ):
        if st.session_state.last_transcription:
            st.success(
                "Text saved in the current session."
            )
        else:
            st.warning(
                "No recorded text available."
            )

with col2:
    if st.button(
        "🗑️ Clear Text",
        use_container_width=True,
    ):
        st.session_state.last_transcription = ""
        st.session_state.last_confidence = None
        st.session_state.last_audio_bytes = None
        st.session_state.last_audio_duration = None
        st.session_state.last_sample_rate = None
        st.rerun()

# ============================
# 🧪 TROUBLESHOOTING
# ============================

with st.expander("🧪 Troubleshooting"):
    st.markdown(
        """
        Test in this order:

        1. Keep Light audio enhancement OFF.
        2. Keep Background speaker suppression OFF.
        3. Keep sound event detection OFF.
        4. Record a normal sentence.
        5. Open Listen to the recording.
        6. If playback is unclear, the problem is microphone/browser/recording.
        7. If playback is clear but the text is wrong, the problem is STT/model/language.

        The app uses Deepgram Nova-3 with language=multi for mixed-language
        speech. It also sends technical keyterms such as Python, Streamlit,
        Jupyter, Matplotlib, Plotly and API.
        """
    )

# ============================
# ℹ️ CURRENT CONFIGURATION
# ============================

with st.expander("ℹ️ Current configuration"):
    st.write("Model:", DEEPGRAM_MODEL)
    st.write("Language:", DEEPGRAM_LANGUAGE)
    st.write(
        "Enhancement:",
        "ON" if enhance_audio else "OFF",
    )
    st.write(
        "Background suppression:",
        "ON" if suppress_background else "OFF",
    )
    st.write(
        "Sound events:",
        "ON" if detect_events else "OFF",
    )
    st.write(
        "Transport:",
        "Deepgram REST API",
    )
