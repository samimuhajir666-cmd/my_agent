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

try:
    from deepgram import DeepgramClient, PrerecordedOptions
    DEEPGRAM_V3 = True
except ImportError:
    from deepgram import Deepgram
    DEEPGRAM_V3 = False

load_dotenv()

# ============================
# 🔑 DEEPGRAM API KEY INITIALIZATION
# ============================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    try:
        if "DEEPGRAM_API_KEY" in st.secrets:
            DEEPGRAM_API_KEY = st.secrets["DEEPGRAM_API_KEY"]
    except Exception:
        pass

if not DEEPGRAM_API_KEY:
    st.error("DEEPGRAM_API_KEY not found. Please set it in .env or Streamlit Secrets.")
    st.stop()

try:
    deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
except Exception as e:
    st.error(f"Deepgram client failed to initialize: {e}")
    st.stop()

SYSTEM_PROMPT = (
    "Roman Urdu and English mixed conversation. Common words: kya haal hai, "
    "main theek hoon, billing amount kitna hua, cash or card, payment failed, "
    "status code 500, transaction approved, number 1 2 3, plus minus."
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

# ============================
# 🎙️ DEEPGRAM TRANSCRIBE
# ============================
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_CONFIDENCE_THRESHOLD = 0.55

def frame_energies(audio_chunk, sample_rate, frame_ms=30):
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    energies = []
    for i in range(0, len(audio_chunk), frame_len):
        frame = audio_chunk[i:i+frame_len]
        if len(frame) > 0:
            rms = np.sqrt(np.mean(frame.astype(np.float64)**2))
            energies.append(rms)
    return energies

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

def transcribe_with_deepgram(processed_bytes, raw_mono_audio, sample_rate, events):
    options = PrerecordedOptions(
        model=DEEPGRAM_MODEL,
        smart_format=True,
        punctuate=True,
        utterances=True,
        language="multi",
    )
    
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
    return signal.filtfilt(b, a, audio_data.astype(np.float64))

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
        start_w = max(0, i - window_frames // 2)
        end_w = min(n_frames, i + window_frames // 2 + 1)
        local_peak[i] = max(np.max(frame_energy[start_w:end_w]), 1e-5)

    processed_audio = np.zeros_like(audio_data, dtype=np.float64)
    for i in range(n_frames):
        chunk_start = i * frame_len
        chunk_end = min((i + 1) * frame_len, len(audio_data))
        if chunk_start >= len(audio_data):
            break
            
        ratio = frame_energy[i] / local_peak[i]
        gain = DOMINANT_MIN_GAIN + (1.0 - DOMINANT_MIN_GAIN) * (ratio ** (2.0 * sensitivity))
        gain = min(max(gain, DOMINANT_MIN_GAIN), 1.0)
        
        processed_audio[chunk_start:chunk_end] = audio_data[chunk_start:chunk_end].astype(np.float64) * gain
        
    return np.clip(processed_audio, -32768, 32767).astype(np.int16)

# ============================
# 🖥️ STREAMLIT APPLICATION DRAWER
# ============================
st.title("🎙️ Speech Analytics Audio Pipeline")

audio_upload = st.file_uploader("Upload Audio", type=["wav", "mp3"])
recorded_audio = mic_recorder(start_prompt="🔴 Start Recording", stop_prompt="⏹️ Stop Recording", key="recorder")

audio_source = None

# Corrected Block with Proper Indentation
if audio_upload is not None:
    audio_bytes = audio_upload.read()
    audio_source = io.BytesIO(audio_bytes)
elif recorded_audio is not None:
    audio_bytes = recorded_audio['bytes']
    audio_source = io.BytesIO(audio_bytes)

if audio_source is not None:
    with st.spinner("Processing audio & transcribing..."):
        try:
            sample_rate, raw_signal = wav.read(audio_source)
            if len(raw_signal.shape) > 1:
                raw_signal = np.mean(raw_signal, axis=1).astype(raw_signal.dtype)
            
            filtered = bandpass_filter(raw_signal, sample_rate)
            denoised = nr.reduce_noise(y=filtered, sr=sample_rate)
            clean_signal = attenuate_background_speakers(denoised, sample_rate, sensitivity=0.5)
            
            buffer = io.BytesIO()
            wav.write(buffer, sample_rate, clean_signal)
            processed_bytes = buffer.getvalue()
            
            tags = detect_sound_events(clean_signal, sample_rate)
            transcript = transcribe_with_deepgram(processed_bytes, clean_signal, sample_rate, tags)
            
            st.markdown("### Transcript Output")
            st.code(transcript)
        except Exception as err:
            st.error(f"Error processing audio: {err}")
