import io
import os
import re
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import noisereduce as nr
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode

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

# SECURITY: never hardcode a real API key here. If it's missing, stop and
# tell the user to set it properly instead of silently falling back to a
# key baked into the source file (that key becomes public the moment this
# file is shared, pasted, or committed anywhere).
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

# Whisper's "prompt" field biases vocabulary/style, it does not enforce
# rules. A short natural example in the target style/script works better
# than a list of instructions. Kept under 896 chars (hard API limit).
SYSTEM_PROMPT = (
    "Transcribe the audio accurately. English and Roman Urdu text like: "
    "kya haal hai, main theek hoon, billing amount kitna hua, cash or card "
    "payment failed status code 500 transaction approved number 1 2 3 "
    "plus minus."
)

if len(SYSTEM_PROMPT) > 896:
    st.error("SYSTEM_PROMPT exceeds character limit.")
    st.stop()


# --- FORCE ROMAN SCRIPT (guarantee, doesn't rely on Whisper "obeying") ---
def force_roman_script(text):
    """Any non-Latin character (Arabic, Urdu, Devanagari, etc.) gets
    converted to its closest Roman-letter form. Already-Roman text passes
    through unchanged."""
    if not text:
        return text
    has_non_ascii = bool(re.search(r'[^\x00-\x7F]', text))
    if not has_non_ascii:
        return text
    return unidecode(text)


# ============================
# 😮 NON-SPEECH SOUND EVENT DETECTION (coughing, laughing, etc.)
# ============================
# Whisper only transcribes WORDS -- it has no concept of "coughing" or
# "laughing". To caption those (like YouTube closed captions do), we run a
# second, separate model: PANNs, trained on Google's AudioSet (527 sound
# categories). This is a real audio classifier, not a language model, so
# it can reliably recognize non-speech sounds.
#
# NOTE: first run downloads a ~300MB checkpoint automatically -- needs
# normal internet access on whatever machine actually runs this app.
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
}

EVENT_CONFIDENCE_THRESHOLD = 0.20  # AudioSet models give modest confidences; tuned conservatively
EVENT_WINDOW_SECONDS = 1.5         # granularity of event detection (smaller = more precise timing, slower)
PANNS_SAMPLE_RATE = 32000          # PANNs models expect 32kHz mono audio


@st.cache_resource(show_spinner=False)
def load_sound_event_model():
    """Loads the PANNs AudioSet tagging model once and caches it for the
    life of the app process (not re-downloaded/reloaded on every recording)."""
    from panns_inference import AudioTagging
    return AudioTagging(checkpoint_path=None, device="cpu")


def detect_sound_events(audio_data, sample_rate):
    """
    Runs the audio through PANNs in fixed windows and returns a list of
    (start_seconds, end_seconds, tag) for any recognized non-speech event
    that crosses the confidence threshold. Consecutive windows with the
    same tag are merged into a single span.
    Returns [] if the model can't be loaded (e.g. no internet for the
    first-time download) -- event detection is a bonus, never blocks
    transcription.
    """
    try:
        model = load_sound_event_model()
    except Exception:
        return []

    try:
        import librosa

        audio_float = audio_data.astype(np.float32) / 32768.0
        if sample_rate != PANNS_SAMPLE_RATE:
            audio_float = librosa.resample(
                audio_float, orig_sr=sample_rate, target_sr=PANNS_SAMPLE_RATE
            )

        window_len = int(EVENT_WINDOW_SECONDS * PANNS_SAMPLE_RATE)
        total_len = len(audio_float)

        raw_events = []  # (start_sec, end_sec, tag)
        for start_sample in range(0, total_len, window_len):
            end_sample = min(start_sample + window_len, total_len)
            chunk = audio_float[start_sample:end_sample]
            if len(chunk) < PANNS_SAMPLE_RATE * 0.3:  # skip tiny leftover tail
                continue

            clipwise_output, _ = model.inference(chunk[None, :])
            probs = clipwise_output[0]

            from panns_inference import labels as audioset_labels

            for idx, prob in enumerate(probs):
                if prob < EVENT_CONFIDENCE_THRESHOLD:
                    continue
                label_name = audioset_labels[idx].strip().lower()
                if label_name in EVENT_LABEL_MAP:
                    start_sec = start_sample / PANNS_SAMPLE_RATE
                    end_sec = end_sample / PANNS_SAMPLE_RATE
                    raw_events.append((start_sec, end_sec, EVENT_LABEL_MAP[label_name]))

        # Merge consecutive/overlapping windows that share the same tag
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


def merge_speech_and_events(segments, events):
    """
    Combines Whisper's speech segments (each with start/end/text) and the
    detected non-speech events into one chronological, readable string.
    `segments` items need .start/.end/.text (or dict-style with those keys).
    """
    items = []
    for seg in segments:
        seg_start = getattr(seg, "start", None)
        seg_end = getattr(seg, "end", None)
        seg_text = getattr(seg, "text", None)
        if seg_start is None and isinstance(seg, dict):
            seg_start, seg_end, seg_text = seg.get("start"), seg.get("end"), seg.get("text")
        if seg_text:
            items.append((seg_start or 0.0, seg_end or 0.0, force_roman_script(seg_text.strip())))

    for start_sec, end_sec, tag in events:
        items.append((start_sec, end_sec, tag))

    items.sort(key=lambda x: x[0])

    parts = []
    for _, _, text in items:
        if parts and parts[-1] == text:
            continue  # avoid repeating the same tag back-to-back
        parts.append(text)

    return " ".join(parts).strip()


# ============================
# 🎚️ AUDIO PROCESSING (noise handling)
# ============================
MIN_RMS_ENERGY = 60.0        # below this = treated as silence/background noise only
MIN_DURATION_SECONDS = 0.8   # below this = too short, Whisper tends to hallucinate
MAX_DURATION_SECONDS = 120   # cap so one very long clip doesn't slow everything down

# Human speech mostly lives in this frequency band. Anything outside it
# (low rumble from fans/AC/traffic, high hiss) is very unlikely to be voice.
SPEECH_LOW_HZ = 85
SPEECH_HIGH_HZ = 3400


def bandpass_filter(audio_data, sample_rate, low_hz=SPEECH_LOW_HZ, high_hz=SPEECH_HIGH_HZ):
    """Cuts frequencies outside the human speech range, removing a lot of
    non-voice background noise (fans, traffic rumble, hiss) before it ever
    reaches the noise-reduction or transcription step."""
    nyquist = 0.5 * sample_rate
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)  # keep below Nyquist limit
    b, a = signal.butter(4, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, audio_data.astype(np.float64))
    return filtered


def normalize_audio(audio_data, target_peak=0.9):
    """Brings quiet recordings up to a consistent volume so a soft voice
    isn't drowned out relative to noise. Scales based on peak amplitude."""
    max_val = np.max(np.abs(audio_data))
    if max_val < 1e-6:
        return audio_data
    scale = (target_peak * 32767.0) / max_val
    return audio_data * scale


def process_audio_buffer(audio_bytes):
    """Returns (transcription_ready_bytes, raw_mono_audio, sample_rate), or
    None if the clip is silence/noise/too short (not worth sending to the
    API). raw_mono_audio is returned separately (before bandpass filtering)
    so the sound-event detector sees the full original spectrum."""
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)

        duration_seconds = len(audio_data) / float(sample_rate)
        if duration_seconds < MIN_DURATION_SECONDS:
            return None
        if duration_seconds > MAX_DURATION_SECONDS:
            audio_data = audio_data[: int(MAX_DURATION_SECONDS * sample_rate)]

        rms_energy = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        if rms_energy < MIN_RMS_ENERGY:
            return None

        raw_mono_audio = audio_data.copy()

        # 1) Keep only the frequency range where human speech lives
        filtered_audio = bandpass_filter(audio_data, sample_rate)

        # 2) Non-stationary noise reduction: adapts to changing background
        #    noise (traffic, chatter) instead of assuming a constant noise
        #    floor like the default "stationary" mode does.
        cleaned_audio_data = nr.reduce_noise(
            y=filtered_audio,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.75,
        )

        # 3) Normalize volume so quiet speech isn't lost against noise
        cleaned_audio_data = normalize_audio(cleaned_audio_data)

        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, cleaned_audio_data.astype(np.int16))
        output_buffer.seek(0)

        return output_buffer.read(), raw_mono_audio, sample_rate
    except Exception:
        return None


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

detect_events = st.checkbox(
    "VOICE RECORDER IS HERE",
    value=True,
    help="First use downloads a ~300MB model one time. Needs internet on this machine."
)

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
        result = process_audio_buffer(audio_bytes)

    if result is None:
        st.warning("⚠️ Noise, silence, or clip too short. Please speak clearly into the mic.")
    else:
        processed_bytes, raw_mono_audio, sample_rate = result

        with st.spinner("⚡ Transcribing speech..."):
            try:
                audio_file = io.BytesIO(processed_bytes)
                audio_file.name = "recording.wav"

                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model=STT_MODEL,
                    prompt=SYSTEM_PROMPT,
                    response_format="verbose_json",
                    temperature=0.0
                )

                segments = getattr(transcription, "segments", None) or []

                if detect_events:
                    with st.spinner("😮 Checking for coughing, laughing, etc..."):
                        events = detect_sound_events(raw_mono_audio, sample_rate)
                else:
                    events = []

                if segments:
                    text_from_voice = merge_speech_and_events(segments, events)
                else:
                    # Fallback: no segment timestamps available, just use
                    # the plain transcript text with events listed after it.
                    raw_text = getattr(transcription, "text", "").strip()
                    text_from_voice = force_roman_script(raw_text)
                    if events:
                        tags = " ".join(sorted(set(tag for _, _, tag in events)))
                        text_from_voice = f"{text_from_voice} {tags}".strip()

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
