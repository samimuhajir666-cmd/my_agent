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


# ============================================================
# LISTENER - FINAL SPEECH TO TEXT AGENT
# ============================================================
#
# GOAL
# ----
# 1. Listen to English, Urdu, Roman Urdu, or mixed speech.
# 2. Do NOT force Urdu through English recognition.
# 3. Do NOT use an LLM to invent/answer anything.
# 4. Do NOT use cough/laugh/breath detection.
# 5. Do NOT use aggressive noise cancellation.
# 6. Gently boost only genuinely quiet recordings.
# 7. If the complete recording strongly looks like non-speech,
#    return [unclear audio] instead of inventing a transcript.
#
# PIPELINE
# --------
# Microphone
#     ↓
# Original WAV
#     ↓
# Gentle quiet-voice boost (only when needed)
#     ↓
# Whisper Large-v3
#     ↓
# Quality gate
#     ↓
# Romanize only AFTER transcription
#     ↓
# Final text
#
# STREAMLIT CLOUD
# ---------------
# requirements.txt:
#     streamlit
#     groq
#     python-dotenv
#     scipy
#     numpy
#     streamlit-mic-recorder
#
# Streamlit Secrets:
#     GROQ_API_KEY = "YOUR_NEW_KEY"
#
# ============================================================


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="High-Precision Speech to Text",
    page_icon="🎤",
    layout="centered",
)

load_dotenv()


# ============================================================
# 2. API KEY
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY was not found. "
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


# ============================================================
# 3. MODEL
# ============================================================

STT_MODEL = "whisper-large-v3"


# ============================================================
# 4. AUDIO SETTINGS
# ============================================================

MIN_DURATION_SECONDS = 0.30
MAX_DURATION_SECONDS = 120.0

# Only reject an essentially empty recording.
# This is deliberately very low so quiet speech is not rejected.
MIN_RMS = 0.00015

# Only recordings below this RMS are boosted.
QUIET_RMS = 0.025

# Never amplify a quiet recording more than this much.
MAX_QUIET_GAIN = 3.0


# ============================================================
# 5. QUALITY-GATE SETTINGS
# ============================================================
#
# These are conservative engineering thresholds, not a guarantee.
# Whisper's verbose JSON gives segment metadata that we can inspect.
#
# High no_speech_prob:
#     model thinks the segment may not contain speech.
#
# Very low avg_logprob:
#     model is uncertain about the generated tokens.
#
# We use both so "unclear audio" is safer than inventing words.
# ============================================================

MAX_NO_SPEECH_PROB = 0.85
MIN_AVG_LOGPROB = -1.40

# If a transcript is only one short word and the model is not
# confident enough, we prefer [unclear audio].
MIN_WORDS_FOR_WEAK_TRANSCRIPT = 2


# ============================================================
# 6. SESSION STATE
# ============================================================

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

if "last_language" not in st.session_state:
    st.session_state.last_language = ""

if "last_quality" not in st.session_state:
    st.session_state.last_quality = ""

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None


# ============================================================
# 7. ROMANIZATION
# ============================================================

def romanize_text(text: str) -> str:
    """
    IMPORTANT: Do not run Unidecode on Urdu transcription.

    Unidecode produces machine transliteration such as:
        myN / khhh / hyN

    That is not natural Roman Urdu and was one of the reasons the
    displayed Urdu results looked wrong. Keep the real Whisper text.
    """

    if not text:
        return ""

    return text.strip()


# ============================================================
# 8. READ WAV
# ============================================================

def read_wav(audio_bytes: bytes):
    """
    Read the browser's WAV recording and convert it to mono float32.
    """

    buffer = io.BytesIO(
        audio_bytes
    )

    sample_rate, audio = wav.read(
        buffer
    )

    if sample_rate <= 0:
        raise ValueError(
            "Invalid WAV sample rate."
        )

    if audio is None or len(audio) == 0:
        raise ValueError(
            "The recording contains no audio samples."
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

    # Integer WAV -> float [-1, 1]
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


# ============================================================
# 9. AUDIO LEVELS
# ============================================================

def rms_level(audio) -> float:
    if audio is None or len(audio) == 0:
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


def peak_level(audio) -> float:
    if audio is None or len(audio) == 0:
        return 0.0

    return float(
        np.max(
            np.abs(
                audio.astype(
                    np.float64
                )
            )
        )
    )


# ============================================================
# 10. GENTLE QUIET-VOICE BOOST
# ============================================================

def gently_boost_quiet_voice(audio):
    """
    Only boost genuinely quiet recordings.

    No:
      - noise cancellation
      - band-pass filter
      - background suppression
      - speech deletion
    """

    audio = audio.astype(
        np.float32
    ).copy()

    rms = rms_level(
        audio
    )

    # Normal/loud recording:
    # leave it completely alone.
    if rms >= QUIET_RMS:
        return audio

    if rms <= 0.0:
        return audio

    gain = min(
        QUIET_RMS / rms,
        MAX_QUIET_GAIN,
    )

    boosted = (
        audio * gain
    )

    # Prevent clipping.
    peak = peak_level(
        boosted
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
# 11. FLOAT AUDIO -> WAV
# ============================================================

def audio_to_wav_bytes(
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

    output = io.BytesIO()

    wav.write(
        output,
        sample_rate,
        int_audio,
    )

    output.seek(0)

    return output.read()


# ============================================================
# 12. PREPARE AUDIO
# ============================================================

def prepare_audio(
    audio_bytes: bytes,
    boost_quiet: bool,
):
    """
    Preserve the original recording and create an optional
    gently boosted version.
    """

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

    # Only reject an essentially empty recording.
    if rms < MIN_RMS:
        raise ValueError(
            "The recording is essentially silent."
        )

    original_audio = (
        audio.copy()
    )

    if boost_quiet:
        processed_audio = (
            gently_boost_quiet_voice(
                audio
            )
        )
    else:
        processed_audio = (
            audio.copy()
        )

    processed_bytes = (
        audio_to_wav_bytes(
            processed_audio,
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
# 13. WHISPER SEGMENT QUALITY
# ============================================================

def segment_quality(segment):
    """
    Return:
        text,
        no_speech_prob,
        avg_logprob
    """

    text = str(
        getattr(
            segment,
            "text",
            "",
        )
        or ""
    ).strip()

    no_speech_prob = float(
        getattr(
            segment,
            "no_speech_prob",
            0.0,
        )
        or 0.0
    )

    avg_logprob_value = getattr(
        segment,
        "avg_logprob",
        None,
    )

    if avg_logprob_value is None:
        avg_logprob = -0.5
    else:
        avg_logprob = float(
            avg_logprob_value
        )

    return (
        text,
        no_speech_prob,
        avg_logprob,
    )


# ============================================================
# 14. WHISPER TRANSCRIPTION
# ============================================================

def transcribe_with_whisper(
    audio_bytes: bytes,
    debug=False,
):
    """
    Whisper Large-v3 only.

    IMPORTANT FIX:
    There is NO language="en".

    The input can be:
      - English
      - Urdu
      - Roman Urdu
      - English + Urdu

    We ask the model to transcribe, not translate or answer.
    """

    audio_file = io.BytesIO(
        audio_bytes
    )

    audio_file.name = (
        "input_speech.wav"
    )

    try:

        transcription = (
            client
            .audio
            .transcriptions
            .create(
                file=("input_speech.wav", audio_file.read(), "audio/wav"),
                model=STT_MODEL,

                # DO NOT force English here.
                # Whisper must be free to recognize Urdu/English.

                response_format="verbose_json",

                # Segment metadata is useful for our quality gate.
                timestamp_granularities=[
                    "segment"
                ],

                temperature=0.2,

                # Context only.
                # We do NOT tell Whisper to "convert" or "translate".
                prompt=(
                    "Examples of expected speech: 'main theek hoon', 'kya haal hai', "
                    "'yes I need help', 'payment failed', 'P-512 error', "
                    "'mera terminal kaam nahi kar raha', 'hello agent'. "
                    "Transcribe exactly what is spoken. Do not translate or answer."
                ),
            )
        )

    except Exception as exc:

        if debug:
            st.exception(exc)

        raise RuntimeError(
            f"Whisper transcription failed: {exc}"
        ) from exc


    # --------------------------------------------------------
    # Main transcript
    # --------------------------------------------------------

    full_text = str(
        getattr(
            transcription,
            "text",
            "",
        )
        or ""
    ).strip()


    detected_language = str(
        getattr(
            transcription,
            "language",
            "",
        )
        or ""
    )


    segments = (
        getattr(
            transcription,
            "segments",
            None,
        )
        or []
    )


    # If segment metadata is unavailable, return the full text
    # rather than silently discarding a valid transcription.
    if not segments:

        if not full_text:
            return {
                "text": "",
                "language": detected_language,
                "accepted": False,
                "reason": "No transcript returned.",
            }

        return {
            "text": full_text,
            "language": detected_language,
            "accepted": True,
            "reason": "No segment metadata returned.",
        }


    accepted_segments = []
    rejected_segments = []


    # --------------------------------------------------------
    # Quality gate
    # --------------------------------------------------------

    for segment in segments:

        (
            segment_text,
            no_speech_prob,
            avg_logprob,
        ) = segment_quality(
            segment
        )

        if not segment_text:
            continue

        # Strong sign of silence/non-speech.
        if (
            no_speech_prob
            >= MAX_NO_SPEECH_PROB
        ):
            rejected_segments.append(
                {
                    "text": segment_text,
                    "reason": (
                        f"no_speech_prob={no_speech_prob:.2f}"
                    ),
                }
            )
            continue

        # Very weak token confidence.
        if (
            avg_logprob
            < MIN_AVG_LOGPROB
        ):
            rejected_segments.append(
                {
                    "text": segment_text,
                    "reason": (
                        f"avg_logprob={avg_logprob:.2f}"
                    ),
                }
            )
            continue

        accepted_segments.append(
            segment_text
        )


    # --------------------------------------------------------
    # If EVERYTHING is rejected:
    # do not invent a transcript.
    # --------------------------------------------------------

    if not accepted_segments:

        return {
            "text": "[unclear audio]",
            "language": detected_language,
            "accepted": False,
            "reason": (
                "All returned segments failed the "
                "conservative quality gate."
            ),
            "rejected_segments": rejected_segments,
        }


    final_text = " ".join(
        accepted_segments
    ).strip()


    # --------------------------------------------------------
    # Single-word weak transcript protection
    # --------------------------------------------------------

    word_count = len(
        final_text.split()
    )

    if (
        word_count
        < MIN_WORDS_FOR_WEAK_TRANSCRIPT
        and rejected_segments
    ):

        return {
            "text": "[unclear audio]",
            "language": detected_language,
            "accepted": False,
            "reason": (
                "Only a very short transcript remained while "
                "other segments were rejected."
            ),
            "rejected_segments": rejected_segments,
        }


    return {
        "text": final_text,
        "language": detected_language,
        "accepted": True,
        "reason": "Passed quality gate.",
        "rejected_segments": rejected_segments,
    }


# ============================================================
# 15. TRANSCRIBE WITH ORIGINAL FALLBACK
# ============================================================

def transcribe_audio(
    original_bytes,
    processed_bytes,
    debug=False,
):
    """
    Use the lightly boosted audio first.

    If that produces no accepted transcript, send the untouched
    original recording once.

    We NEVER ask another AI to guess/correct the transcript.
    """

    first = (
        transcribe_with_whisper(
            processed_bytes,
            debug=debug,
        )
    )

    first_text = str(
        first.get("text")
        or ""
    ).strip()


    # Good result:
    if (
        first.get("accepted")
        and first_text
    ):
        return first


    # If processing didn't change the audio, no need for a
    # duplicate request.
    if (
        processed_bytes
        == original_bytes
    ):
        return first


    # Otherwise try original audio once.
    second = (
        transcribe_with_whisper(
            original_bytes,
            debug=debug,
        )
    )

    second_text = str(
        second.get("text")
        or ""
    ).strip()


    # Prefer the accepted original result.
    if second.get("accepted"):
        return second


    # If neither passed, keep the first result rather than
    # inventing anything.
    if first_text:
        return first

    return second


# ============================================================
# 16. SESSION STATE
# ============================================================

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

if "last_language" not in st.session_state:
    st.session_state.last_language = ""

if "last_quality" not in st.session_state:
    st.session_state.last_quality = ""

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None


# ============================================================
# 17. PAGE UI
# ============================================================

st.title(
    "🎤 High-Precision Speech to Text"
)

st.caption(
    "Groq Whisper Large-v3 • English + Urdu + Mixed Speech"
)

st.info(
    "The agent only transcribes speech. "
    "It does not answer or invent a response."
)


# ============================================================
# 18. CONTROLS
# ============================================================

col1, col2 = st.columns(
    2
)

with col1:

    boost_quiet = st.checkbox(
        "🔊 Gently boost quiet voice",
        value=True,
        help=(
            "Only genuinely quiet recordings are amplified. "
            "No noise cancellation is applied."
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
    start_prompt=(
        "🎤 Click to Start Recording"
    ),
    stop_prompt=(
        "🛑 Stop Recording"
    ),
    just_once=True,
    use_container_width=True,
    format="wav",
    key="whisper_mic",
)


# ============================================================
# 20. RECORDING
# ============================================================

if audio_output:

    audio_bytes = (
        audio_output.get(
            "bytes"
        )
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

        # ----------------------------------------------------
        # Prepare audio
        # ----------------------------------------------------

        with st.spinner(
            "⏳ Preparing audio..."
        ):

            prepared = (
                prepare_audio(
                    audio_bytes,
                    boost_quiet=boost_quiet,
                )
            )


        original_bytes = (
            prepared[
                "original_bytes"
            ]
        )

        processed_bytes = (
            prepared[
                "processed_bytes"
            ]
        )

        sample_rate = (
            prepared[
                "sample_rate"
            ]
        )

        duration = (
            prepared[
                "duration"
            ]
        )

        rms = (
            prepared[
                "rms"
            ]
        )


        # ----------------------------------------------------
        # Original playback
        # ----------------------------------------------------

        st.subheader(
            "🔊 Original Recording"
        )

        st.audio(
            original_bytes,
            format="audio/wav",
        )


        # ----------------------------------------------------
        # Processed playback
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
        # Audio information
        # ----------------------------------------------------

        with st.expander(
            "🔧 Audio information"
        ):

            st.write(
                f"Duration: {duration:.2f} seconds"
            )

            st.write(
                f"Sample rate: {sample_rate} Hz"
            )

            st.write(
                f"RMS: {rms:.5f}"
            )

            st.write(
                "Noise cancellation: OFF"
            )

            st.write(
                "Background-speaker suppression: OFF"
            )

            st.write(
                "Cough/laugh/breath detection: OFF"
            )

            st.write(
                "Forced language: NONE"
            )


        # ----------------------------------------------------
        # Transcription
        # ----------------------------------------------------

        with st.spinner(
            "⚡ Whisper Large-v3 is listening..."
        ):

            result = (
                transcribe_audio(
                    original_bytes,
                    processed_bytes,
                    debug=debug_mode,
                )
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

        quality_reason = str(
            result.get(
                "reason",
                "",
            )
            or ""
        )


        # ----------------------------------------------------
        # Romanize ONLY after accepted transcription
        # ----------------------------------------------------

        if (
            accepted
            and raw_text
            and raw_text
            != "[unclear audio]"
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

            st.session_state.last_quality = (
                quality_reason
            )

            st.success(
                "✅ Transcription complete!"
            )


        else:

            # Do not show a guessed transcript.
            st.session_state.last_transcription = (
                "[unclear audio]"
            )

            st.session_state.last_language = (
                detected_language
            )

            st.session_state.last_quality = (
                quality_reason
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
        f"""<div style="padding:18px;border-radius:10px;background:#ffffff;border:1px solid #dcdcdc;box-shadow:0 2px 5px rgba(0,0,0,0.05);">
<div style="font-weight:bold;font-size:14px;color:#555555;margin-bottom:8px;">Result:</div>
<div style="font-size:18px;color:#111111;line-height:1.5;font-weight:500;">{safe_text}</div>
</div>""",
        unsafe_allow_html=True,
    )

    if st.session_state.last_language:

        st.caption(
            "Detected language: "
            f"{st.session_state.last_language}"
        )

    if st.session_state.last_quality:

        st.caption(
            "Quality check: "
            f"{st.session_state.last_quality}"
        )

else:

    st.info(
        "Your transcription will appear here."
    )


# ============================================================
# 22. BUTTONS
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
        st.session_state.last_quality = ""
        st.session_state.last_audio = None

        st.rerun()


# ============================================================
# 23. FINAL CONFIGURATION
# ============================================================

with st.expander(
    "ℹ️ Current configuration"
):

    st.write(
        "STT provider: Groq Whisper"
    )

    st.write(
        "Model: whisper-large-v3"
    )

    st.write(
        "Language: automatic / multilingual"
    )

    st.write(
        "Quiet-voice boost:",
        "ON" if boost_quiet else "OFF",
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
        "LLM answer generation: OFF"
    )

    st.write(
        "Unclear speech handling: [unclear audio]"
    )
