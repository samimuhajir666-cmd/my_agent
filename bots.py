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

# SECURITY: never hardcode a real API key here.
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

# Whisper's "prompt" field biases vocabulary/style -- it does NOT follow
# instructions. Writing commands here risks Whisper literally transcribing
# those words back as if they were spoken. Keep this as a natural example.
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
# 😮 NON-SPEECH SOUND EVENT DETECTION (coughing, laughing, breathing, etc.)
# ============================
# Whisper only transcribes WORDS -- it has no concept of "coughing" or
# "breathing". To caption those (like YouTube closed captions do), we run a
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
    "breathing": "[breathing]",
    "wheeze": "[breathing]",
    "gasp": "[breathing]",
}

# Breathing is much quieter/subtler than coughing or laughing, so AudioSet
# models give it lower confidence even when it's really there. We use a
# separate, lower threshold just for it so normal breathing actually gets
# caught instead of being ignored.
EVENT_CONFIDENCE_THRESHOLD = 0.20
BREATHING_CONFIDENCE_THRESHOLD = 0.12
BREATHING_LABELS = {"breathing", "wheeze", "gasp"}

EVENT_WINDOW_SECONDS = 1.5
PANNS_SAMPLE_RATE = 32000


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
        from panns_inference import labels as audioset_labels

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
    """
    Whisper hallucinates stock phrases on unclear/weak audio because of its
    training data (lots of YouTube outros). We catch this two ways:
    1) Whisper's own confidence scores (no_speech_prob / avg_logprob).
    2) A fallback blocklist of common hallucinated phrases.
    """
    cleaned = text.strip().lower()

    if no_speech_prob is not None and no_speech_prob > NO_SPEECH_PROB_THRESHOLD:
        return True
    if avg_logprob is not None and avg_logprob < AVG_LOGPROB_THRESHOLD:
        return True
    if cleaned in HALLUCINATION_PHRASES:
        return True

    return False


def merge_speech_and_events(segments, events):
    """
    Combines Whisper's speech segments and detected non-speech events into
    one chronological, readable string. Likely-hallucinated segments are
    dropped using Whisper's own confidence signals.
    """
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
# 🎚️ AUDIO PROCESSING (noise handling)
# ============================
MIN_RMS_ENERGY = 60.0
MIN_DURATION_SECONDS = 0.8
MAX_DURATION_SECONDS = 120

# --- Voice Activity Detection (VAD) settings ---
# We check small frames individually instead of averaging energy across the
# WHOLE clip -- a long recording with silence...speech...silence would get
# its real speech "diluted" by a whole-clip average and wrongly rejected.
VAD_FRAME_MS = 30
MIN_SPEECH_SECONDS = 0.3
NOISE_FLOOR_PERCENTILE = 10
SPEECH_ABOVE_NOISE_FACTOR = 2.5

# Human speech FUNDAMENTAL frequency lives around 85-255 Hz, with harmonics
# extending up to ~3400 Hz.
SPEECH_LOW_HZ = 85
SPEECH_HIGH_HZ = 3400

# --- "Ignore the person talking in the background" settings ---
DOMINANCE_FRAME_MS = 100
DOMINANCE_WINDOW_SECONDS = 1.5
DOMINANCE_RELATIVE_THRESHOLD = 0.45
DOMINANCE_ATTENUATION = 0.15

# --- Pitch-based speaker identification ---
PITCH_TOLERANCE_HZ = 40
PITCH_FMIN = 75
PITCH_FMAX = 400
PITCH_HOP = 512


def suppress_background_speaker(audio_data, sample_rate):
    """
    Pitch-first background speaker filter:
    1) Estimates the admin's typical pitch (energy-weighted, so louder
       moments count more toward "whose pitch is this").
    2) ANY voiced frame whose pitch doesn't match the admin's pitch gets
       attenuated -- REGARDLESS of how loud it is. A background speaker on
       the same mic is often not much quieter than the admin, so a
       loudness-only gate never even considers them.
    3) For frames where pitch can't be detected (unvoiced sounds, breath,
       near-silence), fall back to the loudness-relative check.

    LIMITATION: if the admin and the background speaker have very similar
    voice pitch, this still can't tell them apart -- that needs a proper
    voice-identity model.
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
        f0_fine, _voiced_flag, _voiced_prob = librosa.pyin(
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
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                f0[i] = np.median(vals)
    except Exception:
        pass

    voiced_mask = ~np.isnan(f0)

    if voiced_mask.any():
        energy_cutoff = np.percentile(frame_energy[voiced_mask], 50)
        reference_mask = voiced_mask & (frame_energy >= energy_cutoff)
        reference_pitches = f0[reference_mask] if reference_mask.any() else f0[voiced_mask]
        admin_pitch = np.median(reference_pitches)

        pitch_mismatch = voiced_mask & (np.abs(f0 - admin_pitch) > PITCH_TOLERANCE_HZ)
        is_background |= pitch_mismatch

    with np.errstate(divide="ignore", invalid="ignore"):
        unvoiced_quiet = (~voiced_mask) & (frame_energy < DOMINANCE_RELATIVE_THRESHOLD * local_dominant)
    is_background |= unvoiced_quiet

    gain = np.ones(n_frames)
    gain[is_background] = DOMINANCE_ATTENUATION

    output = audio_data.astype(np.float64).copy()
    for i in range(n_frames):
        start = i * frame_len
        end = min(len(audio_data), start + frame_len)
        output[start:end] *= gain[i]

    return output.astype(audio_data.dtype)


def bandpass_filter(audio_data, sample_rate, low_hz=SPEECH_LOW_HZ, high_hz=SPEECH_HIGH_HZ):
    """Cuts frequencies outside the human speech range, removing a lot of
    non-voice background noise (fans, traffic rumble, hiss)."""
    nyquist = 0.5 * sample_rate
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)
    b, a = signal.butter(4, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, audio_data.astype(np.float64))
    return filtered


def normalize_audio(audio_data, target_peak=0.9):
    """Brings quiet recordings up to a consistent volume."""
    max_val = np.max(np.abs(audio_data))
    if max_val < 1e-6:
        return audio_data
    scale = (target_peak * 32767.0) / max_val
    return audio_data * scale


def frame_energies(audio_data, sample_rate, frame_ms=VAD_FRAME_MS):
    """Splits audio into short frames and returns the RMS energy of each."""
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    energies = []
    for start in range(0, len(audio_data), frame_len):
        chunk = audio_data[start:start + frame_len]
        if len(chunk) == 0:
            continue
        energies.append(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    return energies


def contains_real_speech(audio_data, sample_rate):
    """
    Frame-based check: does this clip contain enough speech-level audio,
    REGARDLESS of how much silence surrounds it?
    """
    energies = frame_energies(audio_data, sample_rate)
    if not energies:
        return False

    noise_floor = np.percentile(energies, NOISE_FLOOR_PERCENTILE)
    dynamic_threshold = max(noise_floor * SPEECH_ABOVE_NOISE_FACTOR, MIN_RMS_ENERGY)

    speech_frame_count = sum(1 for e in energies if e > dynamic_threshold)
    speech_seconds = speech_frame_count * (VAD_FRAME_MS / 1000)

    return speech_seconds >= MIN_SPEECH_SECONDS


def process_audio_buffer(audio_bytes):
    """Returns (transcription_ready_bytes, raw_mono_audio, sample_rate), or
    None if the clip is silence/noise/too short."""
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

        focused_audio = suppress_background_speaker(audio_data, sample_rate)

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
