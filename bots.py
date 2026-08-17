import io
import os
import re
import html
import threading
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

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
# 🔑 ASSEMBLYAI API KEY
# ============================
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not ASSEMBLYAI_API_KEY:
    try:
        ASSEMBLYAI_API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY")
    except Exception:
        ASSEMBLYAI_API_KEY = None

if not ASSEMBLYAI_API_KEY:
    st.error(
        "ASSEMBLYAI_API_KEY not found. "
        "Put it in .env or Streamlit Secrets."
    )
    st.stop()

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
# 🔤 ROMAN SCRIPT HELPER
# ============================
def force_roman_script(text):
    if not text:
        return text
    has_non_ascii = bool(re.search(r'[^\x00-\x7F]', text))
    if not has_non_ascii:
        return text
    return unidecode(text)


# ============================
# 🎙️ ASSEMBLYAI TRANSCRIBE
# ============================
def transcribe_with_assemblyai(
    processed_bytes,
    sample_rate,
    voice_focus_mode="far-field",
    voice_focus_threshold=0.7,
    debug=False,
):
    """
    Stream WAV bytes through AssemblyAI Streaming STT SDK.

    Uses:
    - universal-3-5-pro  → best model for noisy/multi-speaker environments
    - voice_focus        → server-side background suppression (human ears model)
    - format_turns=True  → clean final transcripts only
    """
    collected_turns = []
    done_event = threading.Event()

    def on_begin(client: StreamingClient, event: BeginEvent):
        pass

    def on_turn(client: StreamingClient, event: TurnEvent):
        if event.transcript and event.end_of_turn:
            collected_turns.append(event.transcript.strip())

    def on_terminated(client: StreamingClient, event: TerminationEvent):
        done_event.set()

    def on_error(client: StreamingClient, error: StreamingError):
        if debug:
            st.warning(f"AssemblyAI streaming error: {error}")
        done_event.set()

    client = StreamingClient(
        StreamingClientOptions(
            api_key=ASSEMBLYAI_API_KEY,
            terminate_timeout=10.0,
        )
    )
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    # ✅ KEY: voice_focus = "human ears" — isolate speech, kill background
    client.connect(
        StreamingParameters(
            sample_rate=sample_rate,
            speech_model="universal-3-5-pro",
            format_turns=True,
            voice_focus=voice_focus_mode,
            voice_focus_threshold=voice_focus_threshold,
            end_of_turn_confidence_threshold=0.6,
        )
    )

    # Stream WAV bytes in chunks
    CHUNK_SIZE = 4096
    offset = 0
    while offset < len(processed_bytes):
        chunk = processed_bytes[offset: offset + CHUNK_SIZE]
        client.send_audio(chunk)
        offset += CHUNK_SIZE

    client.disconnect(terminate=True)
    done_event.wait(timeout=15)

    full_text = " ".join(collected_turns).strip()
    full_text = force_roman_script(full_text)

    return {"text": full_text, "confidence": 1.0}


# ============================
# 🎚️ AUDIO PROCESSING
# ============================
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


def bandpass_filter(audio_data, sample_rate, low_hz=SPEECH_LOW_HZ, high_hz=SPEECH_HIGH_HZ):
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


def attenuate_background_speakers(audio_data, sample_rate, sensitivity):
    try:
        frame_len = max(1, int(sample_rate * DOMINANT_FRAME_MS / 1000))
        n_frames = int(np.ceil(len(audio_data) / frame_len))
        if n_frames <= 1:
            return audio_data

        frame_energy = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            chunk = audio_data[i * frame_len:(i + 1) * frame_len]
            if len(chunk) > 0:
                frame_energy[i] = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

        window_frames = max(1, int(DOMINANT_WINDOW_SECONDS * 1000 / DOMINANT_FRAME_MS))
        local_peak = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            start = max(0, i - window_frames // 2)
            end = min(n_frames, i + window_frames // 2 + 1)
            local_peak[i] = np.max(frame_energy[start:end])

        sensitivity = float(np.clip(sensitivity, 0.05, 0.80))
        quieter = frame_energy < (sensitivity * local_peak)
        gain = np.ones(n_frames, dtype=np.float64)
        gain[quieter] = DOMINANT_MIN_GAIN

        if len(gain) > 5:
            smoothing_window = signal.get_window("hamming", 5)
            smoothing_window /= np.sum(smoothing_window)
            gain = signal.convolve(gain, smoothing_window, mode="same")
            gain = np.clip(gain, DOMINANT_MIN_GAIN, 1.0)

        output = audio_data.astype(np.float64).copy()
        for i in range(n_frames):
            start = i * frame_len
            end = min(len(audio_data), start + frame_len)
            output[start:end] *= gain[i]

        return output
    except Exception:
        return audio_data


def normalize_audio(audio_data, target_peak=0.90):
    max_val = np.max(np.abs(audio_data))
    if max_val < 1e-8:
        return audio_data
    return audio_data * (target_peak * 32767.0 / max_val)


def frame_energies(audio_data, sample_rate, frame_ms=VAD_FRAME_MS):
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    energies = []
    for start in range(0, len(audio_data), frame_len):
        chunk = audio_data[start:start + frame_len]
        if len(chunk) == 0:
            continue
        energies.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))))
    return energies


def contains_real_speech(audio_data, sample_rate):
    if audio_data is None or len(audio_data) == 0:
        return False
    energies = frame_energies(audio_data, sample_rate)
    if not energies:
        return False
    noise_floor = np.percentile(energies, NOISE_FLOOR_PERCENTILE)
    dynamic_threshold = max(noise_floor * SPEECH_ABOVE_NOISE_FACTOR, MIN_RMS_ENERGY)
    speech_frame_count = sum(1 for e in energies if e > dynamic_threshold)
    return (speech_frame_count * VAD_FRAME_MS / 1000.0) >= MIN_SPEECH_SECONDS


def process_audio_buffer(
    audio_bytes,
    enhance_audio=False,
    suppress_background=False,
    background_sensitivity=0.35,
    debug=False,
):
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

        if duration > MAX_DURATION_SECONDS:
            audio_data = audio_data[:int(MAX_DURATION_SECONDS * sample_rate)]
            duration = len(audio_data) / float(sample_rate)

        if not contains_real_speech(audio_data, sample_rate):
            return None

        raw_mono_audio = audio_data.copy()
        processed_audio = audio_data.copy()

        # Optional: client-side background attenuation
        # NOTE: Voice Focus (server-side) is preferred. Use this only as extra help.
        if suppress_background:
            processed_audio = attenuate_background_speakers(
                processed_audio, sample_rate, background_sensitivity
            )

        # Optional: light bandpass filter only (no noisereduce — it hurts STT accuracy)
        if enhance_audio:
            processed_audio = bandpass_filter(processed_audio, sample_rate)
            processed_audio = normalize_audio(processed_audio)

        if not contains_real_speech(processed_audio, sample_rate):
            processed_audio = raw_mono_audio.copy()

        processed_audio = np.clip(processed_audio, -32768, 32767).astype(np.int16)

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
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
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
defaults = {
    "last_transcription": "",
    "last_confidence": None,
    "last_audio_bytes": None,
    "last_audio_duration": None,
    "last_sample_rate": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================
# 🧩 PAGE CONTENT
# ============================
load_css("style.css")
load_html("index.html")

st.title("🎤 SPEECH TO TEXT")
st.caption(
    "AssemblyAI Universal-3.5 Pro • Voice Focus • English + Roman Urdu"
)
st.info(
    "Record your voice, stop the recording, and the app will transcribe it. "
    "Voice Focus runs server-side — it isolates speech and ignores background noise automatically."
)

# ============================
# 👂 VOICE FOCUS (Human Ears) CONTROLS
# ============================
st.subheader("👂 Voice Focus — Human Ears Mode")

col_vf1, col_vf2 = st.columns(2)

with col_vf1:
    voice_focus_mode = st.selectbox(
        "Microphone type",
        options=["far-field", "near-field"],
        index=0,
        help=(
            "far-field: laptop mic, room mic, conference — best for 2 people talking.\n"
            "near-field: headset or handset held close to mouth."
        ),
    )

with col_vf2:
    voice_focus_threshold = st.slider(
        "Background suppression strength",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help=(
            "Higher = more aggressive background removal. "
            "0.7 is the default. Lower if your own voice gets cut."
        ),
    )

# ============================
# 🎚️ OPTIONAL CLIENT-SIDE CONTROLS
# ============================
st.subheader("🎚️ Optional Audio Controls")
st.caption(
    "Voice Focus (above) handles noise server-side. "
    "These are extra client-side options — keep OFF unless needed."
)

col_a, col_b = st.columns(2)

with col_a:
    enhance_audio = st.checkbox(
        "✨ Light bandpass filter",
        value=False,
        help="Gentle speech-frequency filter only. No noisereduce (it hurts STT accuracy).",
    )

with col_b:
    suppress_background = st.checkbox(
        "🔇 Client-side background attenuation",
        value=False,
        help="Extra quiet-frame suppression. Keep OFF if Voice Focus is enough.",
    )

if suppress_background:
    bg_sensitivity = st.slider(
        "Attenuation sensitivity",
        min_value=0.05, max_value=0.80, value=0.35, step=0.05,
    )
else:
    bg_sensitivity = 0.35

debug_mode = st.checkbox("🐞 Show technical errors", value=False)

detect_events = st.checkbox(
    "😮 Detect coughing/laughing/breathing",
    value=False,
    help="Optional. Requires PANNs/librosa. OFF by default.",
)

# ============================
# 🎤 MICROPHONE
# ============================
st.subheader("🎤 Voice Input")
st.write("Press Start, speak normally, then press Stop.")

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

        with st.expander("🔧 Audio information"):
            st.write(f"Duration: {duration:.2f} seconds")
            st.write(f"Sample rate: {sample_rate} Hz")
            st.write(f"Original size: {len(audio_bytes):,} bytes")
            st.write(f"Processed size: {len(processed_bytes):,} bytes")
            st.write(f"Voice Focus mode: {voice_focus_mode}")
            st.write(f"Voice Focus threshold: {voice_focus_threshold}")
            st.write(f"Bandpass filter: {'ON' if enhance_audio else 'OFF'}")
            st.write(f"Client attenuation: {'ON' if suppress_background else 'OFF'}")

        with st.expander("🔊 Listen to the recording"):
            st.audio(audio_bytes, format="audio/wav")

        events = []
        if detect_events:
            with st.spinner("😮 Checking optional sound events..."):
                events = detect_sound_events(raw_audio, sample_rate)

        with st.spinner("⚡ Transcribing with AssemblyAI Universal-3.5 Pro + Voice Focus..."):
            try:
                transcription_result = transcribe_with_assemblyai(
                    processed_bytes,
                    sample_rate,
                    voice_focus_mode=voice_focus_mode,
                    voice_focus_threshold=voice_focus_threshold,
                    debug=debug_mode,
                )

                text_from_voice = transcription_result["text"].strip()

                if events:
                    event_tags = " ".join(event[2] for event in events)
                    if event_tags:
                        text_from_voice = f"{text_from_voice} {event_tags}".strip()

                if text_from_voice:
                    st.session_state.last_transcription = text_from_voice
                    st.session_state.last_confidence = transcription_result["confidence"]
                    st.success("✅ Complete!")
                else:
                    st.warning(
                        "⚠️ AssemblyAI did not return recognizable speech. "
                        "Try lowering the Voice Focus threshold or switching to near-field."
                    )

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
        <div class="output-card">
            <div class="output-title">Result:</div>
            <div class="output-text">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
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
            st.success("Text saved in the current session.")
        else:
            st.warning("No recorded text available.")

with col2:
    if st.button("🗑️ Clear Text", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ============================
# 🧪 TROUBLESHOOTING
# ============================
with st.expander("🧪 Troubleshooting"):
    st.markdown(
        """
        **Human Ears Mode — how it works:**

        - **Voice Focus (server-side)** isolates speech and suppresses background noise
          before audio reaches the model. This is the main noise killer.
        - Use **far-field** for laptop/room mics (2 people in a room).
        - Use **near-field** for headsets or phones held close.
        - If your voice gets cut off, **lower the threshold** (e.g. 0.5).
        - If background is still leaking, **raise the threshold** (e.g. 0.85).
        - Do NOT enable client-side noisereduce — it introduces artifacts that hurt accuracy.

        **Test order:**
        1. Keep bandpass filter OFF.
        2. Keep client attenuation OFF.
        3. Set Voice Focus to far-field, threshold 0.7.
        4. Record and check playback first.
        5. If playback is clear but text is wrong → adjust Voice Focus threshold.
        """
    )

with st.expander("ℹ️ Current configuration"):
    st.write("Model: AssemblyAI Universal-3.5 Pro")
    st.write(f"Voice Focus: {voice_focus_mode} (threshold: {voice_focus_threshold})")
    st.write(f"Bandpass filter: {'ON' if enhance_audio else 'OFF'}")
    st.write(f"Client attenuation: {'ON' if suppress_background else 'OFF'}")
    st.write(f"Sound events: {'ON' if detect_events else 'OFF'}")
    st.write("Transport: AssemblyAI WebSocket SDK")
