import io
import os
import re
import html
import time

import requests
import scipy.io.wavfile as wav
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode


# ============================================================
# 1. PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="LISTENER - AssemblyAI STT",
    page_icon="🎤",
    layout="centered",
)

load_dotenv()


# ============================================================
# 2. ASSEMBLYAI API KEY
# ============================================================

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not ASSEMBLYAI_API_KEY:
    try:
        ASSEMBLYAI_API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY")
    except Exception:
        ASSEMBLYAI_API_KEY = None

if not ASSEMBLYAI_API_KEY:
    st.error(
        "ASSEMBLYAI_API_KEY was not found.\n\n"
        "Add it to your .env file or Streamlit Secrets."
    )
    st.stop()


# ============================================================
# 3. ASSEMBLYAI SETTINGS
# ============================================================

ASSEMBLYAI_UPLOAD_URL = (
    "https://api.assemblyai.com/v2/upload"
)

ASSEMBLYAI_TRANSCRIPT_URL = (
    "https://api.assemblyai.com/v2/transcript"
)

ASSEMBLYAI_TIMEOUT = 60
ASSEMBLYAI_POLL_SECONDS = 0.8

# Universal-3 Pro is AssemblyAI's current flagship async model.
# Automatic language detection is enabled below.
ASSEMBLYAI_SPEECH_MODELS = [
    "universal-3-pro"
]

# Keep the prompt focused. It is used to tell the model what
# transcription behavior we want; it is NOT an instruction to answer.
ASSEMBLYAI_PROMPT = (
    "Transcribe exactly what the speaker says. "
    "Preserve the original language and mixed-language speech. "
    "Do not answer questions. "
    "Do not summarize. "
    "Do not translate. "
    "Do not invent missing words. "
    "Keep technical terms such as Python, Streamlit, Jupyter, "
    "Matplotlib, Plotly, NumPy, API, AI, machine learning, "
    "Flask, HTML and CSS."
)


# ============================================================
# 4. OPTIONAL TECHNICAL TERMS
# ============================================================

TECHNICAL_TERMS = [
    "Python",
    "Streamlit",
    "Jupyter",
    "Matplotlib",
    "Plotly",
    "NumPy",
    "SciPy",
    "AssemblyAI",
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


# ============================================================
# 5. ROMAN SCRIPT HELPER
# ============================================================

def force_roman_script(text):
    """
    If AssemblyAI returns Urdu/Arabic-script characters,
    transliterate them into Latin/Roman characters.

    English text is left untouched.
    """

    if not text:
        return ""

    if not re.search(r"[^\x00-\x7F]", text):
        return text

    try:
        return unidecode(text)
    except Exception:
        return text


# ============================================================
# 6. BASIC WAV VALIDATION ONLY
# ============================================================

MIN_DURATION_SECONDS = 0.25
MAX_DURATION_SECONDS = 120.0


def inspect_audio(audio_bytes):
    """
    Only checks that the browser gave us a valid WAV.

    IMPORTANT:
    We do NOT remove noise, filter frequencies, detect coughs,
    detect speakers, or reject quiet speech here.
    """

    try:
        buffer = io.BytesIO(audio_bytes)

        sample_rate, audio = wav.read(buffer)

        if sample_rate <= 0:
            return None

        if audio is None or len(audio) == 0:
            return None

        duration = len(audio) / float(sample_rate)

        if duration < MIN_DURATION_SECONDS:
            return None

        if duration > MAX_DURATION_SECONDS:
            return {
                "sample_rate": int(sample_rate),
                "duration": float(duration),
                "channels": (
                    int(audio.shape[1])
                    if getattr(audio, "ndim", 1) > 1
                    else 1
                ),
            }

        return {
            "sample_rate": int(sample_rate),
            "duration": float(duration),
            "channels": (
                int(audio.shape[1])
                if getattr(audio, "ndim", 1) > 1
                else 1
            ),
        }

    except Exception:
        return None


# ============================================================
# 7. ASSEMBLYAI UPLOAD
# ============================================================

def upload_audio_to_assemblyai(audio_bytes):
    """
    Upload the original WAV bytes to AssemblyAI.

    No audio modification happens before this call.
    """

    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/octet-stream",
    }

    response = requests.post(
        ASSEMBLYAI_UPLOAD_URL,
        headers=headers,
        data=audio_bytes,
        timeout=ASSEMBLYAI_TIMEOUT,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            "AssemblyAI upload failed "
            f"({response.status_code}): {response.text}"
        )

    data = response.json()

    upload_url = data.get("upload_url")

    if not upload_url:
        raise RuntimeError(
            "AssemblyAI did not return an upload_url."
        )

    return upload_url


# ============================================================
# 8. START ASSEMBLYAI TRANSCRIPTION
# ============================================================

def create_assemblyai_transcript(audio_url):
    """
    Create an asynchronous AssemblyAI transcription job.
    """

    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json",
    }

    payload = {
        "audio_url": audio_url,

        # Automatic language detection.
        "language_detection": True,

        # Current high-accuracy async model.
        "speech_models": ASSEMBLYAI_SPEECH_MODELS,

        # Preserve what was actually spoken.
        "prompt": ASSEMBLYAI_PROMPT,

        # Helpful formatting without asking the model to answer.
        "punctuate": True,
        "format_text": True,

        # Do not turn this into a speaker-diarization project.
        "speaker_labels": False,
    }

    response = requests.post(
        ASSEMBLYAI_TRANSCRIPT_URL,
        headers=headers,
        json=payload,
        timeout=ASSEMBLYAI_TIMEOUT,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            "AssemblyAI transcription request failed "
            f"({response.status_code}): {response.text}"
        )

    data = response.json()

    transcript_id = data.get("id")

    if not transcript_id:
        raise RuntimeError(
            "AssemblyAI did not return a transcript ID."
        )

    return transcript_id


# ============================================================
# 9. POLL TRANSCRIPTION RESULT
# ============================================================

def wait_for_assemblyai_transcript(
    transcript_id,
    debug=False,
):
    """
    Wait until AssemblyAI finishes the transcript.
    """

    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
    }

    started = time.time()

    while True:

        if time.time() - started > ASSEMBLYAI_TIMEOUT:
            raise TimeoutError(
                "AssemblyAI transcription timed out."
            )

        response = requests.get(
            f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}",
            headers=headers,
            timeout=ASSEMBLYAI_TIMEOUT,
        )

        if response.status_code != 200:
            raise RuntimeError(
                "AssemblyAI status request failed "
                f"({response.status_code}): {response.text}"
            )

        data = response.json()

        status = data.get("status")

        if status == "completed":
            return data

        if status == "error":
            error_message = data.get(
                "error",
                "Unknown AssemblyAI transcription error.",
            )
            raise RuntimeError(error_message)

        if debug:
            st.caption(
                f"AssemblyAI status: {status}"
            )

        time.sleep(
            ASSEMBLYAI_POLL_SECONDS
        )


# ============================================================
# 10. MAIN ASSEMBLYAI TRANSCRIPTION FUNCTION
# ============================================================

def transcribe_with_assemblyai(
    audio_bytes,
    debug=False,
):
    """
    Complete AssemblyAI pipeline:

        WAV bytes
            ↓
        Upload
            ↓
        Transcript job
            ↓
        Poll
            ↓
        Return transcript
    """

    upload_url = upload_audio_to_assemblyai(
        audio_bytes
    )

    transcript_id = create_assemblyai_transcript(
        upload_url
    )

    result = wait_for_assemblyai_transcript(
        transcript_id,
        debug=debug,
    )

    text = (
        result.get("text")
        or ""
    ).strip()

    confidence = result.get(
        "confidence"
    )

    language_code = result.get(
        "language_code"
    )

    return {
        "text": text,
        "confidence": confidence,
        "language_code": language_code,
        "transcript_id": transcript_id,
        "raw_result": result,
    }


# ============================================================
# 11. CSS / HTML HELPERS
# ============================================================

def load_css(file_path="style.css"):

    if not os.path.exists(file_path):
        return

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:

            css = file.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True,
        )

    except Exception:
        pass


def load_html(file_path="index.html"):

    if not os.path.exists(file_path):
        return

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:

            content = file.read()

        try:
            st.html(content)
        except Exception:
            st.markdown(
                content,
                unsafe_allow_html=True,
            )

    except Exception:
        pass


# ============================================================
# 12. SESSION STATE
# ============================================================

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

if "last_language" not in st.session_state:
    st.session_state.last_language = None

if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None

if "last_audio_info" not in st.session_state:
    st.session_state.last_audio_info = None


# ============================================================
# 13. PAGE CONTENT
# ============================================================

load_css("style.css")
load_html("index.html")

st.title(
    "🎤 LISTENER — AssemblyAI Speech to Text"
)

st.caption(
    "Original audio → AssemblyAI Universal-3 Pro → transcription"
)

st.info(
    "This version intentionally keeps the audio path simple. "
    "It does not remove background voices or detect coughs, laughs, "
    "breathing, or other sound events."
)


# ============================================================
# 14. DEBUG OPTION
# ============================================================

debug_mode = st.checkbox(
    "🐞 Show technical status",
    value=False,
)


# ============================================================
# 15. MICROPHONE
# ============================================================

st.subheader("🎤 Voice Input")

st.write(
    "Click Start, speak normally, then click Stop."
)

audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="assemblyai_listener_mic",
)


# ============================================================
# 16. RECORDING PROCESS
# ============================================================

if audio_output:

    audio_bytes = audio_output.get(
        "bytes"
    )

    if not audio_bytes:

        st.error(
            "No audio data was received."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate only. Do not modify the recording.
    # --------------------------------------------------------

    audio_info = inspect_audio(
        audio_bytes
    )

    if audio_info is None:

        st.warning(
            "⚠️ The WAV recording is empty, invalid, "
            "or too short. Please record again."
        )

        st.stop()


    st.session_state.last_audio_bytes = (
        audio_bytes
    )

    st.session_state.last_audio_info = (
        audio_info
    )


    # --------------------------------------------------------
    # ORIGINAL RECORDING PLAYBACK
    # --------------------------------------------------------

    with st.expander(
        "🔊 Listen to the original recording",
        expanded=True,
    ):

        st.audio(
            audio_bytes,
            format="audio/wav",
        )


    # --------------------------------------------------------
    # AUDIO INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "🔧 Audio information"
    ):

        st.write(
            "Duration:",
            f"{audio_info['duration']:.2f} seconds",
        )

        st.write(
            "Sample rate:",
            f"{audio_info['sample_rate']} Hz",
        )

        st.write(
            "Channels:",
            audio_info["channels"],
        )

        st.write(
            "Original size:",
            f"{len(audio_bytes):,} bytes",
        )

        st.write(
            "Audio processing:",
            "NONE",
        )


    # --------------------------------------------------------
    # ASSEMBLYAI TRANSCRIPTION
    # --------------------------------------------------------

    with st.spinner(
        "⚡ AssemblyAI is listening..."
    ):

        try:

            result = transcribe_with_assemblyai(
                audio_bytes,
                debug=debug_mode,
            )

            text = (
                result["text"]
                or ""
            ).strip()

            confidence = result[
                "confidence"
            ]

            language_code = result[
                "language_code"
            ]


            # ------------------------------------------------
            # Preserve Roman Urdu style if AssemblyAI returns
            # Urdu/Arabic script.
            # ------------------------------------------------

            text = force_roman_script(
                text
            ).strip()


            if text:

                st.session_state.last_transcription = (
                    text
                )

                st.session_state.last_confidence = (
                    confidence
                )

                st.session_state.last_language = (
                    language_code
                )

                st.success(
                    "✅ Transcription complete."
                )

            else:

                st.warning(
                    "⚠️ AssemblyAI completed the job "
                    "but returned no text."
                )


        except requests.RequestException as error:

            st.error(
                "❌ Network error while contacting AssemblyAI."
            )

            if debug_mode:
                st.exception(error)


        except Exception as error:

            st.error(
                f"❌ AssemblyAI error: {error}"
            )

            if debug_mode:
                st.exception(error)


# ============================================================
# 17. TRANSCRIPT OUTPUT
# ============================================================

st.divider()

st.subheader(
    "📝 Transcribed Text"
)


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


    if (
        st.session_state.last_confidence
        is not None
    ):

        st.caption(
            "AssemblyAI confidence: "
            f"{float(st.session_state.last_confidence):.2f}"
        )


    if st.session_state.last_language:

        st.caption(
            "Detected language: "
            f"{st.session_state.last_language}"
        )

else:

    st.info(
        "Your transcription will appear here."
    )


# ============================================================
# 18. BUTTONS
# ============================================================

st.divider()

col1, col2 = st.columns(2)


with col1:

    if st.button(
        "🛑 Lock Text",
        use_container_width=True,
    ):

        if (
            st.session_state.last_transcription
        ):

            st.success(
                "Text saved in the current session."
            )

        else:

            st.warning(
                "No transcription is available."
            )


with col2:

    if st.button(
        "🗑️ Clear Text",
        use_container_width=True,
    ):

        st.session_state.last_transcription = ""
        st.session_state.last_confidence = None
        st.session_state.last_language = None
        st.session_state.last_audio_bytes = None
        st.session_state.last_audio_info = None

        st.rerun()


# ============================================================
# 19. TROUBLESHOOTING
# ============================================================

with st.expander(
    "🧪 Troubleshooting"
):

    st.markdown(
        """
### If the agent hears the wrong words

1. First open **Listen to the original recording**.
2. Make sure your actual voice is clear in the recording.
3. The app sends that original WAV to AssemblyAI.
4. There is no cough detector, speaker suppression, noise filter,
   band-pass filter, or aggressive audio processing between the
   microphone and AssemblyAI.

### If the recording is clear but transcription is wrong

That means the problem is on the speech-recognition side rather
than an audio filter changing your voice.

### Important

Roman Urdu is spoken Urdu. The model may return Urdu script for
Urdu speech; this app transliterates non-Latin output into Roman
characters after transcription. English words are kept as English.
        """
    )


# ============================================================
# 20. CURRENT CONFIGURATION
# ============================================================

with st.expander(
    "ℹ️ Current configuration"
):

    st.write(
        "STT provider:",
        "AssemblyAI",
    )

    st.write(
        "Speech model:",
        ", ".join(
            ASSEMBLYAI_SPEECH_MODELS
        ),
    )

    st.write(
        "Language detection:",
        "ON",
    )

    st.write(
        "Speaker diarization:",
        "OFF",
    )

    st.write(
        "Sound-event detection:",
        "OFF",
    )

    st.write(
        "Noise cancellation:",
        "OFF",
    )

    st.write(
        "Background-speaker suppression:",
        "OFF",
    )

    st.write(
        "Audio modification:",
        "NONE",
    )
