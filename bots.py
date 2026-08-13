import io
import os
import re
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import noisereduce as nr
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode

load_dotenv()

# ============================
# 🔑 API KEY INITIALIZATION
# ============================
STT_MODEL_KEY = os.getenv("DEEPGRAM_API_KEY")
if not STT_MODEL_KEY:
    try:
        if "DEEPGRAM_API_KEY" in st.secrets:
            STT_MODEL_KEY = st.secrets["DEEPGRAM_API_KEY"]
    except Exception:
        pass

if not STT_MODEL_KEY:
    st.error("DEEPGRAM_API_KEY not found. Please set it in .env or Streamlit Secrets.")
    st.stop()

try:
    client = Groq(api_key=STT_MODEL_KEY)
except Exception as e:
    st.error(f"Could not initialize Groq client: {e}")
    st.stop()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    try:
        if "DEEPGRAM_API_KEY" in st.secrets:
            DEEPGRAM_API_KEY = st.secrets["DEEPGRAM_API_KEY"]
    except Exception:
        pass

deepgram_client = None
if DEEPGRAM_API_KEY:
    try:
        deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
    except Exception as e:
        st.warning(f"Deepgram client failed to initialize: {e}")

# ============================
# 🎯 MODEL & PROMPT
# ============================
STT_MODEL = "nova-3"

SYSTEM_PROMPT = (
    "Roman Urdu and English mixed conversation. Common words: kya haal hai, "
    "main theek hoon, billing amount kitna hua, cash or card, payment failed, "
    "status code 500, transaction approved, number 1 2 3, plus minus."
)

if len(SYSTEM_PROMPT) > 896:
    st.error("SYSTEM_PROMPT exceeds character limit.")
    st.stop()


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
    from panns_inference import AudioTagging
    return AudioTagging(checkpoint_path=None, device="cpu")


def detect_sound_events(audio_data, sample_rate):
    try:
        model = load_sound_event_model()
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


HALLUCINATION_FRAGMENTS = {
    "thank you", "thanks for watching", "please subscribe", "subscribe",
    "bye bye", "i'm going", "see you next time", "shukriya", "theek hai",
}
HALLUCINATION_PHRASES = {
    "bye", "bye.", "okay", "ok", "yeah", "hmm", "you", ".", "..", "...",
    "acha", "ji",
}
NO_SPEECH_PROB_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0


def is_likely_hallucinated(text, no_speech_prob, avg_logprob):
    cleaned = text.strip().lower()
    if no_speech_prob is not None and no_speech_prob > NO_SPEECH_PROB_THRESHOLD:
        return True
    if avg_logprob is not None and avg_logprob < AVG_LOGPROB_THRESHOLD:
        return True
    if cleaned in HALLUCINATION_PHRASES:
        return True
    if any(fragment in cleaned for fragment in HALLUCINATION_FRAGMENTS):
        return True
    return False


def segment_has_real_audio(seg_start, seg_end, audio_data, sample_rate):
    if seg_start is None or seg_end is None:
        return True
    start_sample = max(0, int(seg_start * sample_rate))
    end_sample = min(len(audio_data), int(seg_end * sample_rate))
    if end_sample <= start_sample:
        return True
    chunk = audio_data[start_sample:end_sample]
    energies = frame_energies(chunk, sample_rate)
    if not energies:
        return False
    return max(energies) > MIN_RMS_ENERGY


def merge_speech_and_events(segments, events, audio_data=None, sample_rate=None):
    items = []
    for seg in segments:
        seg_start = getattr(seg, "start", None)
        seg_end = getattr(seg, "end", None)
        seg_text = getattr(seg, "text", None)
        no_speech_prob = getattr(seg, "no_speech_prob", None)
        avg_logprob = getattr(seg, "avg_logprob", None)
        if seg_start is None and isinstance(seg, dict):
            seg_start = seg.get("start")
            seg_end = seg.get("end")
            seg_text = seg.get("text")
            no_speech_prob = seg.get("no_speech_prob")
            avg_logprob = seg.get("avg_logprob")

        if not seg_text:
            continue
        if is_likely_hallucinated(seg_text, no_speech_prob, avg_logprob):
            continue
        if audio_data is not None and sample_rate is not None:
            if not segment_has_real_audio(seg_start, seg_end, audio_data, sample_rate):
                continue

        items.append((seg_start or 0.0, seg_end or 0.0, force_roman_script(seg_text.strip())))

    for start_sec, end_sec, tag in events:
        items.append((start_sec, end_sec, tag))

    items.sort(key=lambda x: x[0])

    parts = []
    for _, _, text in items:
        if parts and parts[-1] == text:
            continue
        parts.append(text)

    return " ".join(parts).strip()


DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_CONFIDENCE_THRESHOLD = 0.55


def transcribe_with_deepgram(processed_bytes, raw_mono_audio, sample_rate, events):
    options = PrerecordedOptions(
        model=DEEPGRAM_MODEL,
        smart_format=True,
        punctuate=True,
        utterances=True,
        language="multi",
    )
    
    # ✅ Fixed Deepgram SDK call syntax
    payload = {"buffer": processed_bytes, "mimetype": "audio/wav"}
    response = deepgram_client.listen.rest.v("1").transcribe_file(payload, options)

    channel = response.results.channels[0]
    alternative = channel.alternatives[0]
    utterances = getattr(response.results, "utterances", None) or []

    items = []
    if utterances:
        for utt in utterances:
            text = getattr(utt, "transcript", "").strip()
            confidence = getattr(utt, "confidence", 1.0)
            start = getattr(utt, "start", 0.0)
            end = getattr(utt, "end", 0.0)

            if not text or confidence < DEEPGRAM_CONFIDENCE_THRESHOLD:
                continue
            if not segment_has_real_audio(start, end, raw_mono_audio, sample_rate):
                continue

            items.append((start, end, force_roman_script(text)))
    else:
        text = getattr(alternative, "transcript", "").strip()
        if text:
            items.append((0.0, 0.0, force_roman_script(text)))

    for start_sec, end_sec, tag in events:
        items.append((start_sec, end_sec, tag))

    items.sort(key=lambda x: x[0])

    parts = []
    for _, _, text in items:
        if parts and parts[-1] == text:
            continue
        parts.append(text)

    return " ".join(parts).strip()


# ============================
# 🎚️ AUDIO PROCESSING & CONFIGS
# ============================
MIN_RMS_ENERGY = 60.0
MIN_DURATION_SECONDS = 0.8
MAX_DURATION_SECONDS = 120
VAD_FRAME_MS = 30
MIN_SPEECH_SECONDS = 0.3
NOISE_FLOOR_PERCENTILE = 10
SPEECH_ABOVE_NOISE_FACTOR = 2.5
SPEECH_LOW_HZ = 85
SPEECH_HIGH_HZ = 3400

DOMINANT_FRAME_MS = 100
DOMINANT_WINDOW_SECONDS = 2.0
DOMINANT_MIN_GAIN = 0.30


def bandpass_filter(audio_data, sample_rate, low_hz=SPEECH_LOW_HZ, high_hz=SPEECH_HIGH_HZ):
    nyquist = 0.5 * sample_rate
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)
    b, a = signal.butter(4, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, audio_data.astype(np.float64))
    return filtered


def attenuate_background_speakers(audio_data, sample_rate, sensitivity):
    frame_len = max(1, int(sample_rate * DOMINANT_FRAME_MS / 1000))
    n_frames = int(np.ceil(len(audio_data) / frame_len))

    frame_energy = np.zeros(n_frames)
    for i in range(n_frames):
        chunk = audio_data[i * frame_len:(i + 1) * frame_len]
        if len(chunk) > 0:
            frame_energy[i] = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

    window_frames = max(1, int(DOMINANT_WINDOW_SECONDS * 1000 / DOMINANT_FRAME_MS))
    local_peak = np.zeros(n_frames)
    for i in range(n_frames):
        start = max(0, i - window_frames // 2)
        end = min(n_frames, i + window_frames // 2 + 1)
        local_peak[i] = np.max(frame_energy[start:end])

    with np.errstate(divide="ignore", invalid="ignore"):
        quieter_than_dominant = frame_energy < (sensitivity * local_peak)

    gain = np.ones(n_frames)
    gain[quieter_than_dominant] = DOMINANT_MIN_GAIN

    if len(gain) > 5:
        # ✅ Fixed SciPy window call
        smoothing_window = signal.get_window('hamming', 5)
        smoothing_window /= np.sum(smoothing_window)
        gain = signal.convolve(gain, smoothing_window, mode='same')
        gain = np.clip(gain, DOMINANT_MIN_GAIN, 1.0)

    output = audio_data.astype(np.float64).copy()
    for i in range(n_frames):
        start = i * frame_len
        end = min(len(audio_data), start + frame_len)
        output[start:end] *= gain[i]

    return output.astype(audio_data.dtype)


def normalize_audio(audio_data, target_peak=0.9):
    max_val = np.max(np.abs(audio_data))
    if max_val < 1e-6:
        return audio_data
    scale = (target_peak * 32767.0) / max_val
    return audio_data * scale


def frame_energies(audio_data, sample_rate, frame_ms=VAD_FRAME_MS):
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    energies = []
    for start in range(0, len(audio_data), frame_len):
        chunk = audio_data[start:start + frame_len]
        if len(chunk) == 0:
            continue
        energies.append(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    return energies


def contains_real_speech(audio_data, sample_rate):
    energies = frame_energies(audio_data, sample_rate)
    if not energies:
        return False

    noise_floor = np.percentile(energies, NOISE_FLOOR_PERCENTILE)
    dynamic_threshold = max(noise_floor * SPEECH_ABOVE_NOISE_FACTOR, MIN_RMS_ENERGY)

    speech_frame_count = sum(1 for e in energies if e > dynamic_threshold)
    speech_seconds = speech_frame_count * (VAD_FRAME_MS / 1000)

    return speech_seconds >= MIN_SPEECH_SECONDS


def process_audio_buffer(audio_bytes, sensitivity, debug=False):
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

        if not contains_real_speech(audio_data, sample_rate):
            return None

        focused_audio = attenuate_background_speakers(audio_data, sample_rate, sensitivity)

        if not contains_real_speech(focused_audio, sample_rate):
            focused_audio = audio_data

        raw_mono_audio = focused_audio.copy()
        filtered_audio = bandpass_filter(raw_mono_audio, sample_rate)

        cleaned_audio_data = nr.reduce_noise(
            y=filtered_audio,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.4,
        )

        cleaned_audio_data = normalize_audio(cleaned_audio_data)

        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, cleaned_audio_data.astype(np.int16))
        output_buffer.seek(0)

        return output_buffer.read(), raw_mono_audio, sample_rate
    except Exception as e:
        if debug:
            st.exception(e)
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

st.subheader("🎚️ Background Voice Control")
bg_sensitivity = st.slider(
    "Background speaker sensitivity (Higher = suppresses louder background talkers more)",
    min_value=0.2, max_value=0.9, value=0.5, step=0.05,
    help="Raise this if people talking in the background are getting picked up over you."
)

debug_mode = st.checkbox("🐞 Show real errors (debug mode)", value=False)

stt_engine_options = ["Groq (Whisper)"]
if deepgram_client:
    stt_engine_options.append("Deepgram (Nova-3)")

stt_engine = st.selectbox("STT Engine", stt_engine_options)

if not deepgram_client:
    st.caption("Add DEEPGRAM_API_KEY to .env or Streamlit Secrets to enable Deepgram.")

st.subheader("🎤 Voice Input")

detect_events = st.checkbox(
    "😮 Detect sounds like coughing, laughing, breathing (adds tags like [breathing])",
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
        result = process_audio_buffer(audio_bytes, bg_sensitivity, debug=debug_mode)

    if result is None:
        st.warning("⚠️ Noise, silence, or clip too short. Please adjust sliders or speak closer to the mic.")
    else:
        processed_bytes, raw_mono_audio, sample_rate = result

        with st.spinner("⚡ Transcribing speech..."):
            try:
                if detect_events:
                    with st.spinner("😮 Checking for coughing, laughing, breathing..."):
                        events = detect_sound_events(raw_mono_audio, sample_rate)
                else:
                    events = []

                if stt_engine == "Deepgram (Nova-3)":
                    text_from_voice = transcribe_with_deepgram(
                        processed_bytes, raw_mono_audio, sample_rate, events
                    )
                else:
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
                    if segments:
                        text_from_voice = merge_speech_and_events(segments, events, raw_mono_audio, sample_rate)
                    else:
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
