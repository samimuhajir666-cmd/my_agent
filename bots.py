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

SYSTEM_PROMPT = (
    "Transcribe the audio accurately. English and Roman Urdu text like: "
    "kya haal hai, main theek hoon, billing amount kitna hua, cash or card "
    "payment failed status code 500 transaction approved number 1 2 3 "
    "plus minus. If you do not hear clear speech, do not transcribe anything. "
    "Listen correctly, generate text correctly. Scan the primary user voice clearly."
)

if len(SYSTEM_PROMPT) > 896:
    st.error("SYSTEM_PROMPT exceeds character limit.")
    st.stop()


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
    """Loads the PANNs AudioSet tagging model once and caches it."""
    from panns_inference import AudioTagging
    return AudioTagging(checkpoint_path=None, device="cpu")


def detect_sound_events(audio_data, sample_rate):
    """Runs the audio through PANNs in fixed windows and returns a list of
    (start_seconds, end_seconds, tag) for recognized non-speech events."""
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


HALLUCINATION_PHRASES = {
    "thank you", "thank you.", "thanks for watching", "thanks for watching.",
    "please subscribe", "subscribe", "bye", "bye.", "bye bye", "i'm going",
    "i'm going.", "i'm going here", "see you next time", "thank you very much",
    "thank you so much", "okay", "ok", "yeah", "hmm", "you", ".", "..", "...",
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
    return False


def merge_speech_and_events(segments, events):
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

DOMINANCE_FRAME_MS = 40             
DOMINANCE_WINDOW_SECONDS = 0.6       
DOMINANCE_ATTENUATION = 0.02         

PITCH_TOLERANCE_HZ = 25              
PITCH_FMIN = 75
PITCH_FMAX = 280                     
PITCH_HOP = 512


def suppress_background_speaker(audio_data, sample_rate, bg_threshold, voiced_prob_threshold):
    """
    Upgraded voice-dominant spatial filter with dynamic sliders for live adjustments.
    """
    frame_len = max(1, int(sample_rate * DOMINANCE_FRAME_MS / 1000))
    n_frames = int(np.ceil(len(audio_data) / frame_len))

    frame_energy = np.zeros(n_frames)
    for i in range(n_frames):
        chunk = audio_data[i * frame_len:(i + 1) * frame_len]
        if len(chunk) > 0:
            frame_energy[i] = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

    window_frames = max(1, int(DOMINANCE_WINDOW_SECONDS * 1000 / DOMINANCE_FRAME_MS))
    local_dominant = np.zeros(n_frames)
    for i in range(n_frames):
        start = max(0, i - window_frames // 2)
        end = min(n_frames, i + window_frames // 2 + 1)
        local_dominant[i] = np.max(frame_energy[start:end])

    is_background = np.zeros(n_frames, dtype=bool)
    f0 = np.full(n_frames, np.nan)
    
    try:
        import librosa

        audio_float = audio_data.astype(np.float32) / 32768.0
        f0_fine, _voiced_flag, voiced_prob = librosa.pyin(
            audio_float,
            fmin=PITCH_FMIN,
            fmax=PITCH_FMAX,
            sr=sample_rate,
            hop_length=PITCH_HOP,
        )
        fine_sample_positions = librosa.frames_to_samples(
            np.arange(len(f0_fine)), hop_length=PITCH_HOP
        )
        
        for i in range(n_frames):
            frame_start = i * frame_len
            frame_end = frame_start + frame_len
            in_frame = (fine_sample_positions >= frame_start) & (fine_sample_positions < frame_end)
            
            vals = f0_fine[in_frame]
            probs = voiced_prob[in_frame]
            
            # Using custom slider value for voice confidence matching
            valid_mask = (~np.isnan(vals)) & (probs >= voiced_prob_threshold)
            valid_vals = vals[valid_mask]
            
            if len(valid_vals) > 0:
                f0[i] = np.median(valid_vals)
    except Exception:
        pass

    voiced_mask = ~np.isnan(f0)

    if voiced_mask.any():
        energy_cutoff = np.percentile(frame_energy[voiced_mask], 60)
        reference_mask = voiced_mask & (frame_energy >= energy_cutoff)
        reference_pitches = f0[reference_mask] if reference_mask.any() else f0[voiced_mask]
        admin_pitch = np.median(reference_pitches)

        pitch_mismatch = voiced_mask & (np.abs(f0 - admin_pitch) > PITCH_TOLERANCE_HZ)
        is_background |= pitch_mismatch

    with np.errstate(divide="ignore", invalid="ignore"):
        # Using custom background cutoff slider threshold
        unvoiced_quiet = (frame_energy < bg_threshold * local_dominant)
    is_background |= unvoiced_quiet

    gain = np.ones(n_frames)
    gain[is_background] = DOMINANCE_ATTENUATION

    if len(gain) > 5:
        smoothing_window = signal.windows.hamming(5)
        smoothing_window /= np.sum(smoothing_window)
        gain = signal.convolve(gain, smoothing_window, mode='same')
        gain = np.clip(gain, DOMINANCE_ATTENUATION, 1.0)

    output = audio_data.astype(np.float64).copy()
    for i in range(n_frames):
        start = i * frame_len
        end = min(len(audio_data), start + frame_len)
        output[start:end] *= gain[i]

    return output.astype(audio_data.dtype)


def bandpass_filter(audio_data, sample_rate, low_hz=SPEECH_LOW_HZ, high_hz=SPEECH_HIGH_HZ):
    nyquist = 0.5 * sample_rate
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)
    b, a = signal.butter(4, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, audio_data.astype(np.float64))
    return filtered


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


def process_audio_buffer(audio_bytes, bg_threshold, voiced_prob_threshold):
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

        # Pass active UI tuning parameters to the main function
        focused_audio = suppress_background_speaker(audio_data, sample_rate, bg_threshold, voiced_prob_threshold)

        if not contains_real_speech(focused_audio, sample_rate):
            return None

        raw_mono_audio = focused_audio.copy()
        filtered_audio = bandpass_filter(focused_audio, sample_rate)

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

# --- 🎛️ NEW: DYNAMIC AUDIO TUNING PANEL ---
st.subheader("🎛️ Audio Cocktail Filter Controls")
with st.expander("Fine-tune Background Voice Suppression Settings", expanded=True):
    bg_threshold = st.slider(
        "Background Cutoff Threshold (Higher = Suppresses louder background noises/voices)", 
        min_value=0.1, max_value=0.95, value=0.65, step=0.05
    )
    voiced_prob_threshold = st.slider(
        "Voice Confidence Filter (Higher = Rejects weak/faint human harmonics)", 
        min_value=0.1, max_value=0.9, value=0.45, step=0.05
    )

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
        # Pass slider updates directly into processing queue
        result = process_audio_buffer(audio_bytes, bg_threshold, voiced_prob_threshold)

    if result is None:
        st.warning("⚠️ Noise, silence, or clip too short. Please adjust sliders or speak closer to the mic.")
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
                    with st.spinner("😮 Checking for coughing, laughing, breathing..."):
                        events = detect_sound_events(raw_mono_audio, sample_rate)
                else:
                    events = []

                if segments:
                    text_from_voice = merge_speech_and_events(segments, events)
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
