import html
import io
import os
import re

import numpy as np
import scipy.io.wavfile as wav
import streamlit as st

from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder


st.set_page_config(
    page_title="Speech to Text",
    page_icon="🎤",
    layout="centered",
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY not found. "
        "Add it to .env or Streamlit Secrets."
    )
    st.stop()

try:
    client = Groq(
        api_key=GROQ_API_KEY
    )
except Exception as exc:
    st.error(
        f"Could not initialize Groq: {exc}"
    )
    st.stop()



STT_MODEL = "whisper-large-v3"
MIN_DURATION_SECONDS = 0.30
MAX_DURATION_SECONDS = 120.0
MIN_RMS = 0.00010

QUIET_RMS = 0.025

MAX_GAIN = 3.0


MAX_NO_SPEECH_FOR_AUDIO = 0.90
MAX_COMPRESSION_RATIO = 2.8



if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

if "last_language" not in st.session_state:
    st.session_state.last_language = ""

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

if "last_quality_note" not in st.session_state:
    st.session_state.last_quality_note = ""

def romanize_text(text):
    """
    Convert Urdu/Arabic script to approximate Roman text
    AFTER transcription.

    IMPORTANT:
    No global replacements such as:
        N -> n
        aa -> a
        uu -> u

    are used.
    """

    if not text:
        return ""

    if not re.search(
        r"[^\x00-\x7F]",
        text,
    ):
        return text.strip()

    try:
        from unidecode import unidecode

        return unidecode(
            text
        ).strip()

    except Exception:
        return text.strip()


def read_wav(audio_bytes):

    buffer = io.BytesIO(
        audio_bytes
    )

    sample_rate, audio = wav.read(
        buffer
    )

    if sample_rate <= 0:
        raise ValueError(
            "Invalid sample rate."
        )

    if audio is None or len(audio) == 0:
        raise ValueError(
            "Empty recording."
        )

    # Stereo -> mono
    if getattr(
        audio,
        "ndim",
        1,
    ) > 1:

        audio = audio.mean(
            axis=1
        )

    # Integer -> float [-1, 1]
    if np.issubdtype(
        audio.dtype,
        np.integer,
    ):

        info = np.iinfo(
            audio.dtype
        )

        scale = max(
            abs(info.min),
            info.max,
        )

        audio = (
            audio.astype(
                np.float32
            )
            / float(scale)
        )

    else:

        audio = audio.astype(
            np.float32
        )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    return (
        int(sample_rate),
        audio,
    )



def rms_level(audio):

    if (
        audio is None
        or len(audio) == 0
    ):
        return 0.0

    return float(
        np.sqrt(
            np.mean(
                audio.astype(
                    np.float64
                ) ** 2
            )
        )
    )


# ============================================================
# 10. GENTLE QUIET-VOICE BOOST
# ============================================================

def gently_boost_quiet_voice(audio):

    audio = audio.astype(
        np.float32
    ).copy()

    rms = rms_level(
        audio
    )

    # Normal voice -> leave it untouched.
    if rms >= QUIET_RMS:
        return audio

    if rms <= 0.0:
        return audio

    gain = min(
        QUIET_RMS / rms,
        MAX_GAIN,
    )

    boosted = (
        audio * gain
    )

    # Prevent clipping.
    peak = float(
        np.max(
            np.abs(
                boosted
            )
        )
    )

    if peak > 0.95:

        boosted *= (
            0.95 / peak
        )

    return np.clip(
        boosted,
        -1.0,
        1.0,
    )


# ============================================================
# 11. FLOAT -> WAV
# ============================================================

def to_wav_bytes(
    audio,
    sample_rate,
):

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    int_audio = (
        audio * 32767.0
    ).astype(
        np.int16
    )

    buffer = io.BytesIO()

    wav.write(
        buffer,
        sample_rate,
        int_audio,
    )

    buffer.seek(0)

    return buffer.read()


# ============================================================
# 12. PREPARE AUDIO
# ============================================================

def prepare_audio(
    audio_bytes,
    boost_quiet=True,
):

    sample_rate, audio = read_wav(
        audio_bytes
    )

    duration = (
        len(audio)
        / float(sample_rate)
    )

    if duration < MIN_DURATION_SECONDS:
        raise ValueError(
            "Recording is too short."
        )

    if duration > MAX_DURATION_SECONDS:

        max_samples = int(
            MAX_DURATION_SECONDS
            * sample_rate
        )

        audio = audio[
            :max_samples
        ]

        duration = (
            len(audio)
            / float(sample_rate)
        )

    rms = rms_level(
        audio
    )

    # Only reject completely empty/silent audio.
    if rms < MIN_RMS:

        raise ValueError(
            "Recording is essentially silent."
        )

    if boost_quiet:

        processed = (
            gently_boost_quiet_voice(
                audio
            )
        )

    else:

        processed = (
            audio.copy()
        )

    processed_bytes = (
        to_wav_bytes(
            processed,
            sample_rate,
        )
    )

    return {
        "original_bytes": audio_bytes,
        "processed_bytes": processed_bytes,
        "sample_rate": sample_rate,
        "duration": duration,
        "rms": rms,
    }


# ============================================================
# 13. WHOLE-RECORDING QUALITY CHECK
# ============================================================

def whole_recording_quality(
    transcription,
):

    segments = getattr(
        transcription,
        "segments",
        None,
    ) or []

    if not segments:

        # No metadata: don't reject useful text.
        return {
            "reject": False,
            "reason": (
                "No segment metadata available."
            ),
        }


    # Collect metadata.
    no_speech_values = []
    compression_values = []

    for segment in segments:

        no_speech = getattr(
            segment,
            "no_speech_prob",
            None,
        )

        compression = getattr(
            segment,
            "compression_ratio",
            None,
        )

        if no_speech is not None:

            no_speech_values.append(
                float(no_speech)
            )

        if compression is not None:

            compression_values.append(
                float(compression)
            )


    # --------------------------------------------------------
    # Very strong no-speech signal for the WHOLE recording.
    # --------------------------------------------------------

    if (
        no_speech_values
        and max(
            no_speech_values
        ) >= MAX_NO_SPEECH_FOR_AUDIO
        and all(
            value >= MAX_NO_SPEECH_FOR_AUDIO
            for value in no_speech_values
        )
    ):

        return {
            "reject": True,
            "reason": (
                "Whisper reported very high no-speech "
                "probability for the whole recording."
            ),
        }


    # --------------------------------------------------------
    # Strong compression/hallucination signal across all
    # segments. Use only as a secondary guard.
    # --------------------------------------------------------

    if (
        compression_values
        and all(
            value > MAX_COMPRESSION_RATIO
            for value in compression_values
        )
        and len(compression_values) >= 2
    ):

        return {
            "reject": True,
            "reason": (
                "The transcription metadata strongly "
                "resembles repeated/hallucinated text."
            ),
        }


    return {
        "reject": False,
        "reason": "Passed whole-recording quality check.",
    }


# ============================================================
# 14. WHISPER TRANSCRIPTION
# ============================================================

def transcribe_with_whisper(
    audio_bytes,
    debug=False,
):

    audio_file = io.BytesIO(
        audio_bytes
    )

    audio_file.name = (
        "recording.wav"
    )

    try:

        transcription = (
            client
            .audio
            .transcriptions
            .create(

                file=(
                    audio_file.name,
                    audio_file.read(),
                    "audio/wav",
                ),

                model=STT_MODEL,

                # IMPORTANT:
                # No language="en".
                #
                # The recording may contain:
                # English
                # Urdu
                # Roman Urdu
                # English + Urdu

                response_format=(
                    "verbose_json"
                ),

                timestamp_granularities=[
                    "segment"
                ],

                temperature=0.0,

                # Small context hint only.
                # Do not tell Whisper to translate.
                prompt=(
                    "The speaker may speak English, "
                    "Urdu, Roman Urdu, or mixed English "
                    "and Urdu. Transcribe exactly what "
                    "is spoken. Do not translate. "
                    "Do not answer. Keep names and "
                    "technical terms."
                ),
            )
        )

    except Exception as exc:

        if debug:
            st.exception(exc)

        raise RuntimeError(
            f"Whisper transcription failed: {exc}"
        ) from exc


    text = str(
        getattr(
            transcription,
            "text",
            "",
        )
        or ""
    ).strip()

    language = str(
        getattr(
            transcription,
            "language",
            "",
        )
        or ""
    )

    quality = whole_recording_quality(
        transcription
    )


    if quality["reject"]:

        return {
            "text": "",
            "language": language,
            "accepted": False,
            "reason": quality["reason"],
        }


    if not text:

        return {
            "text": "",
            "language": language,
            "accepted": False,
            "reason": (
                "Whisper returned no transcript."
            ),
        }


    # IMPORTANT:
    # Return the COMPLETE transcript.
    # We do NOT delete individual Urdu words or segments.
    return {
        "text": text,
        "language": language,
        "accepted": True,
        "reason": quality["reason"],
    }


# ============================================================
# 15. TRANSCRIBE WITH ORIGINAL FALLBACK
# ============================================================

def transcribe_audio(
    original_bytes,
    processed_bytes,
    debug=False,
):

    # First try the gently boosted audio.
    first = transcribe_with_whisper(
        processed_bytes,
        debug=debug,
    )


    # If good -> return it.
    if first["accepted"]:

        return first


    # If boosting changed the audio, give the original one
    # chance. This is useful for quiet speech that was altered
    # unexpectedly.
    if (
        processed_bytes
        != original_bytes
    ):

        second = transcribe_with_whisper(
            original_bytes,
            debug=debug,
        )

        if second["accepted"]:

            return second


    # Both are unclear.
    return first


# ============================================================
# 16. SESSION STATE
# ============================================================

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

if "last_language" not in st.session_state:
    st.session_state.last_language = ""

if "last_quality_note" not in st.session_state:
    st.session_state.last_quality_note = ""

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None


# ============================================================
# 17. UI
# ============================================================

st.title(
    "🎤 High-Precision Speech to Text"
)

st.caption(
    "Groq Whisper Large-v3 • English + Urdu + Roman Urdu"
)

st.info(
    "This is STT only. It does not answer your questions "
    "and does not use an LLM to invent text."
)


# ============================================================
# 18. SETTINGS
# ============================================================

col1, col2 = st.columns(
    2
)

with col1:

    boost_quiet = st.checkbox(
        "🔊 Gentle quiet-voice boost",
        value=True,
        help=(
            "Only genuinely quiet recordings are amplified. "
            "No noise cancellation."
        ),
    )

with col2:

    debug_mode = st.checkbox(
        "🐞 Debug mode",
        value=False,
    )


# ============================================================
# 19. MICROPHONE
# ============================================================

st.subheader(
    "🎤 Voice Input"
)

st.write(
    "Press Start → speak → Stop."
)

audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="whisper_mic",
)


# ============================================================
# 20. RECORDING
# ============================================================

if audio_output:

    audio_bytes = audio_output.get(
        "bytes"
    )

    if not audio_bytes:

        st.error(
            "No audio data received."
        )

        st.stop()

    st.session_state.last_audio = (
        audio_bytes
    )


    try:

        with st.spinner(
            "⏳ Preparing audio..."
        ):

            prepared = prepare_audio(
                audio_bytes,
                boost_quiet=boost_quiet,
            )


        original_bytes = (
            prepared["original_bytes"]
        )

        processed_bytes = (
            prepared["processed_bytes"]
        )

        duration = (
            prepared["duration"]
        )

        sample_rate = (
            prepared["sample_rate"]
        )

        rms = (
            prepared["rms"]
        )


        # ----------------------------------------------------
        # ORIGINAL RECORDING
        # ----------------------------------------------------

        st.subheader(
            "🔊 Original Recording"
        )

        st.audio(
            original_bytes,
            format="audio/wav",
        )


        # ----------------------------------------------------
        # PROCESSED RECORDING
        # ----------------------------------------------------

        if boost_quiet:

            with st.expander(
                "🔊 Recording sent to Whisper"
            ):

                st.audio(
                    processed_bytes,
                    format="audio/wav",
                )


        # ----------------------------------------------------
        # AUDIO INFO
        # ----------------------------------------------------

        with st.expander(
            "🔧 Audio information"
        ):

            st.write(
                f"Duration: {duration:.2f} sec"
            )

            st.write(
                f"Sample rate: {sample_rate} Hz"
            )

            st.write(
                f"RMS level: {rms:.5f}"
            )

            st.write(
                "Noise cancellation: OFF"
            )

            st.write(
                "Background speaker suppression: OFF"
            )

            st.write(
                "Cough/laugh/breath detection: OFF"
            )

            st.write(
                "Forced language: NONE"
            )


        # ----------------------------------------------------
        # TRANSCRIPTION
        # ----------------------------------------------------

        with st.spinner(
            "⚡ Whisper Large-v3 is listening..."
        ):

            result = transcribe_audio(
                original_bytes,
                processed_bytes,
                debug=debug_mode,
            )


        raw_text = str(
            result.get(
                "text",
                "",
            )
            or ""
        ).strip()


        detected_language = str(
            result.get(
                "language",
                "",
            )
            or ""
        )


        accepted = bool(
            result.get(
                "accepted",
                False,
            )
        )


        quality_note = str(
            result.get(
                "reason",
                "",
            )
            or ""
        )


        # ----------------------------------------------------
        # ONLY romanize accepted transcript.
        # ----------------------------------------------------

        if (
            accepted
            and raw_text
        ):

            final_text = romanize_text(
                raw_text
            )

            st.session_state.last_transcription = (
                final_text
            )

            st.session_state.last_language = (
                detected_language
            )

            st.session_state.last_quality_note = (
                quality_note
            )

            st.success(
                "✅ Transcription complete!"
            )

        else:

            # IMPORTANT:
            # Never show a guessed transcription when the
            # complete recording looks like non-speech.
            st.session_state.last_transcription = (
                "[unclear audio]"
            )

            st.session_state.last_language = (
                detected_language
            )

            st.session_state.last_quality_note = (
                quality_note
            )

            st.warning(
                "⚠️ I could not understand this recording "
                "reliably, so I did not guess the words."
            )


    except Exception as exc:

        st.error(
            f"❌ Error: {exc}"
        )

        if debug_mode:
            st.exception(exc)


# ============================================================
# 21. OUTPUT
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
        <div style="
            padding:18px;
            border-radius:10px;
            background:#ffffff;
            border:1px solid #dcdcdc;
            box-shadow:0 2px 5px rgba(0,0,0,0.05);
        ">

            <div style="
                font-weight:bold;
                font-size:14px;
                color:#555555;
                margin-bottom:8px;
            ">
                Result:
            </div>

            <div style="
                font-size:18px;
                color:#111111;
                line-height:1.5;
                font-weight:500;
            ">
                {safe_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    if st.session_state.last_language:

        st.caption(
            "Detected language: "
            f"{st.session_state.last_language}"
        )


    if st.session_state.last_quality_note:

        st.caption(
            "Quality check: "
            f"{st.session_state.last_quality_note}"
        )

else:

    st.info(
        "Your transcription will appear here."
    )


# ============================================================
# 22. CONTROLS
# ============================================================

st.divider()

col1, col2 = st.columns(
    2
)


with col1:

    if st.button(
        "🛑 Save Text",
        use_container_width=True,
    ):

        if st.session_state.last_transcription:

            st.success(
                "Text locked in this session."
            )

        else:

            st.warning(
                "No text available."
            )


with col2:

    if st.button(
        "🗑️ Clear Text",
        use_container_width=True,
    ):

        st.session_state.last_transcription = ""
        st.session_state.last_language = ""
        st.session_state.last_quality_note = ""
        st.session_state.last_audio = None

        st.rerun()


# ============================================================
# 23. CURRENT CONFIGURATION
# ============================================================

with st.expander(
    "ℹ️ Current configuration"
):

    st.write(
        "Provider: Groq"
    )

    st.write(
        "Model: whisper-large-v3"
    )

    st.write(
        "Language: automatic / multilingual"
    )

    st.write(
        "Quiet voice boost:",
        "ON" if boost_quiet else "OFF",
    )

    st.write(
        "Noise cancellation: OFF"
    )

    st.write(
        "Background speaker suppression: OFF"
    )

    st.write(
        "Sound-event detection: OFF"
    )

    st.write(
        "LLM: OFF"
    )

    st.write(
        "Hallucination guard: whole-recording only"
    )
