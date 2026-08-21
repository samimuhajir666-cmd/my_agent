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
    page_title="Speech to Text (Deepgram Only)",
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
    st.error("DEEPGRAM_API_KEY not found. Please set it in .env or Streamlit Secrets.")
    st.stop()


# ============================
# 🎙️ DEEPGRAM CONFIGURATION (Urdu + English support)
# ============================
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_TIMEOUT = 60

# Common technical and conversational terms for Deepgram boosting
DEEPGRAM_KEYTERMS = [
    "Python", "Streamlit", "Jupyter", "Matplotlib", "Plotly", "NumPy", "SciPy",
    "Deepgram", "AI", "machine learning", "deep learning", "API", "API key",
    "variable", "function", "class", "list", "dictionary", "tuple", "integer",
    "string", "float", "Flask", "FastAPI", "JavaScript", "HTML", "CSS",
    "aap", "kaise", "hain", "main", "theek", "hun", "kya", "kar", "rahe", "ho",
    "shukriya", "haan", "nahi", "bhai", "sahi", "ho", "gaya", "urdu", "roman",
    "hello", "hi", "yes", "no", "ok", "please", "help", "support"
]


# ============================
# 🧹 ROMAN URDU TRANSLITERATION FUNCTION (FIXED)
# ============================
def force_roman_script(text):
    """Convert Deepgram ISO-15919 transliteration to clean Roman Urdu."""
    if not text:
        return ""

    # Remove non-standard characters, accents & diacritics
    text = re.sub(r"[’'‘`\^\~]", "", text)

    # Standardize common phonetic transliteration patterns
    replacements = {
        "iN": "in",
        "aN": "an",
        "uN": "un",
        "eN": "en",
        "oN": "on",
        "N": "n",
        "gii": "gi",
        "uu": "u",
        "aa": "a",
        "ii": "i",
        "ee": "e",
        "oo": "o",
        "DD": "d",
        "TT": "t",
        "RR": "r",
        "khh": "kh",
        "bbaakh": "baakh",
        "zh": "z",
        "sh": "sh",
        "ch": "ch",
        "th": "th",
        "ph": "ph",
        "gh": "gh",
        "ng": "ng",
        "ny": "ny",
        "ty": "ty",
        "dy": "dy",
        "ry": "ry",
        "ly": "ly",
        "my": "my",
        "vy": "vy",
        "sy": "sy",
        "hy": "hy",
        "ky": "ky",
        "gy": "gy",
        "py": "py",
        "by": "by",
        "fy": "fy",
        "wy": "wy",
    }

    for word, repl in replacements.items():
        text = text.replace(word, repl)

    # Cleanup multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove any remaining non-Roman characters
    text = re.sub(r"[^a-zA-Z0-9 .,'?!]", "", text)
    
    return text


def clean_text(text):
    """Basic cleanup for extra spaces and unwanted symbols."""
    if not text:
        return ""
    
    # First apply roman transliteration
    text = force_roman_script(text)
    
    # Then do basic cleanup
    text = re.sub(r"['''`\^\~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================
# 😮 NON-SPEECH SOUND EVENT DETECTION
# ============================
EVENT_LABEL_MAP = {
    "cough": "[coughing]",
    "laughter": "[laughing]",
    "crying, sobbing": "[crying]",
    "screaming": "[screaming]",
    "clapping": "[clapping]",
    "sneeze": "[sneezing]",
    "sigh": "[sighing]",
    "breathing": "[breathing]",
}
EVENT_CONFIDENCE_THRESHOLD = 0.20
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
                if label_name in EVENT_LABEL_MAP and prob >= EVENT_CONFIDENCE_THRESHOLD:
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
# 🎙️ DEEPGRAM TRANSCRIBE (FIXED — Urdu + English support)
# ============================
def transcribe_with_deepgram(processed_bytes, debug=False):
    # 🔥 FIX: "language" parameter hata diya — Deepgram ko detect karne do
    params = [
        ("model", DEEPGRAM_MODEL),
        ("detect_language", "true"),  # Auto-detect Urdu, English, mixed
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("utterances", "true"),
        ("numerals", "true"),
        ("callback", "false"),
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
        raise RuntimeError(f"Deepgram API error {response.status_code}: {detail}")

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

    # 🔥 Apply roman transliteration to transcript
    cleaned_transcript = clean_text(transcript)

    return {
        "text": cleaned_transcript,
        "confidence": confidence,
        "raw": data,
    }


# ============================
# 🎚️ AUDIO PROCESSING CONFIGS
# ============================
MIN_RMS_ENERGY = 5.0
MIN_DURATION_SECONDS = 0.20
MAX_DURATION_SECONDS = 180


def normalize_audio(audio_data, target_peak=0.95):
    max_val = np.max(np.abs(audio_data))
    if max_val < 1e-8:
        return audio_data
    scale = target_peak * 32767.0 / max_val
    return audio_data * scale


def process_audio_buffer(audio_bytes, enhance_audio=False, debug=False):
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

        # Basic audio normalization for clearer audio detection
        processed_audio = normalize_audio(audio_data)
        processed_audio = np.clip(processed_audio, -32768, 32767).astype(np.int16)

        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, processed_audio)
        output_buffer.seek(0)

        return {
            "processed_bytes": output_buffer.read(),
            "raw_audio": audio_data,
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
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

# ============================
# 🧩 PAGE CONTENT
# ============================
load_css("style.css")
load_html("index.html")
st.title("🎤 SPEECH TO TEXT (Deepgram Nova-3)")
st.caption("🔊 Urdu + English + Roman Urdu Support")

# ============================
# 🎤 MICROPHONE INPUT
# ============================
st.subheader("🎤 Voice Input")
st.write("Press Start, speak clearly into the microphone, then press Stop.")

audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="listener_mic",
)

debug_mode = st.checkbox("🐞 Show Technical Errors", value=False)
detect_events = st.checkbox("😮 Detect Sound Events (Laughter/Coughing)", value=False)

# ============================
# 🧠 TRANSCRIPTION LOGIC
# ============================
if audio_output:
    audio_bytes = audio_output.get("bytes")
    if not audio_bytes:
        st.error("No audio data received.")
        st.stop()

    with st.spinner("⏳ Processing Audio..."):
        result = process_audio_buffer(audio_bytes, debug=debug_mode)

    if result is None:
        st.warning("⚠️ Recording was too short or silent. Please try speaking again.")
    else:
        processed_bytes = result["processed_bytes"]
        raw_audio = result["raw_audio"]
        sample_rate = result["sample_rate"]

        with st.expander("🔊 Listen to Recording"):
            st.audio(audio_bytes, format="audio/wav")

        events = []
        if detect_events:
            with st.spinner("😮 Checking sound events..."):
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
                    st.success("✅ Done!")
                elif text_from_voice and confidence < 0.35:
                    st.warning("⚠️ Low confidence. Please speak clearly.")
                    st.session_state.last_transcription = ""
                else:
                    st.warning("⚠️ Could not recognize speech. Try speaking louder or closer to the mic.")

            except Exception as e:
                st.error(f"❌ Transcription error: {e}")
                if debug_mode:
                    st.exception(e)

# ============================
# 📝 DISPLAY OUTPUT
# ============================
st.divider()
st.subheader("📝 Transcribed Text (Roman Urdu / English)")
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
        confidence_pct = st.session_state.last_confidence * 100
        st.caption(f"📊 Deepgram Confidence: {confidence_pct:.1f}%")
else:
    st.info("Your transcription will appear here.")

# ============================
# 🛠️ CONTROLS
# ============================
st.divider()
if st.button("🗑️ Clear Output", use_container_width=True):
    st.session_state.last_transcription = ""
    st.session_state.last_confidence = None
    st.rerun()
