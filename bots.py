import io
import os
import re
import html

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import noisereduce as nr
import requests

import streamlit as st

from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode



st.set_page_config(
    page_title="LISTENER - Speech to Text",
    page_icon="🎤",
    layout="centered"
)


# ============================================================
# 2. LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 3. API KEY
# ============================================================

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


# Streamlit Cloud / Secrets fallback
if not DEEPGRAM_API_KEY:

    try:
        DEEPGRAM_API_KEY = st.secrets["DEEPGRAM_API_KEY"]

    except Exception:
        DEEPGRAM_API_KEY = None


# No API key = stop application
if not DEEPGRAM_API_KEY:

    st.error(
        "❌ DEEPGRAM_API_KEY was not found.\n\n"
        "Put your new Deepgram API key inside .env "
        "or Streamlit Secrets."
    )

    st.stop()

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

DEEPGRAM_MODEL = "nova-3"

DEEPGRAM_CONFIDENCE_THRESHOLD = 0.35


# ============================================================
# 5. TRANSCRIPTION PROMPT
# ============================================================
#
# Keep this short.
#
# Do NOT force Whisper/Deepgram to answer questions.
# The STT engine should ONLY transcribe.
# ============================================================

SYSTEM_PROMPT = (
    "The speaker may use English, Urdu, Roman Urdu, "
    "or a mixture of these languages. "
    "Transcribe the actual spoken words accurately. "
    "Keep names, numbers, Python terms and technical "
    "words accurate. Do not answer questions. "
    "Only transcribe what was spoken."
    "don,t wrong guesses guess as user say and listen as possible as clear ."
)


# ============================================================
# 6. SESSION STATE
# ============================================================

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""


if "last_audio" not in st.session_state:
    st.session_state.last_audio = None


if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None


if "last_error" not in st.session_state:
    st.session_state.last_error = ""


if "recording_count" not in st.session_state:
    st.session_state.recording_count = 0


# ============================================================
# 7. ROMAN SCRIPT CONVERSION
# ============================================================

def force_roman_script(text):
    """
    Convert non-Latin script to approximate Roman text.

    If Deepgram already returns English/Roman Urdu,
    the original text is kept.
    """

    if not text:
        return ""

    try:

        has_non_ascii = bool(
            re.search(r"[^\x00-\x7F]", text)
        )

        if not has_non_ascii:
            return text

        return unidecode(text)

    except Exception:

        return text


# ============================================================
# 8. AUDIO CONFIGURATION
# ============================================================

MIN_DURATION_SECONDS = 0.35

MAX_DURATION_SECONDS = 120.0

MIN_RMS_ENERGY = 20.0

VAD_FRAME_MS = 30

MIN_SPEECH_SECONDS = 0.20

NOISE_FLOOR_PERCENTILE = 10

SPEECH_ABOVE_NOISE_FACTOR = 1.8


# ============================================================
# 9. AUDIO ANALYSIS
# ============================================================

def convert_to_mono(audio_data):
    """
    Convert stereo/multichannel audio to mono.
    """

    if audio_data is None:
        return None

    if len(audio_data.shape) <= 1:
        return audio_data

    return audio_data.mean(axis=1)


# ============================================================

def frame_energies(
    audio_data,
    sample_rate,
    frame_ms=VAD_FRAME_MS
):
    """
    Calculate RMS energy for small audio frames.
    """

    if audio_data is None:
        return []

    frame_len = max(
        1,
        int(sample_rate * frame_ms / 1000)
    )

    energies = []

    for start in range(
        0,
        len(audio_data),
        frame_len
    ):

        chunk = audio_data[
            start:start + frame_len
        ]

        if len(chunk) == 0:
            continue

        energy = np.sqrt(
            np.mean(
                chunk.astype(np.float64) ** 2
            )
        )

        energies.append(float(energy))

    return energies


# ============================================================

def get_audio_duration(
    audio_data,
    sample_rate
):
    """
    Return audio duration in seconds.
    """

    if audio_data is None:
        return 0.0

    if sample_rate <= 0:
        return 0.0

    return len(audio_data) / float(sample_rate)


# ============================================================

def contains_real_speech(
    audio_data,
    sample_rate
):
    """
    Conservative speech check.

    IMPORTANT:
    This function should NOT be too aggressive.
    The old version could reject quiet speech.
    """

    if audio_data is None:
        return False

    if len(audio_data) == 0:
        return False

    energies = frame_energies(
        audio_data,
        sample_rate
    )

    if not energies:
        return False

    maximum_energy = max(energies)

    # Very quiet recording
    if maximum_energy < MIN_RMS_ENERGY:
        return False

    noise_floor = np.percentile(
        energies,
        NOISE_FLOOR_PERCENTILE
    )

    dynamic_threshold = max(
        noise_floor * SPEECH_ABOVE_NOISE_FACTOR,
        MIN_RMS_ENERGY
    )

    speech_frame_count = sum(
        1
        for energy in energies
        if energy > dynamic_threshold
    )

    speech_seconds = (
        speech_frame_count *
        (VAD_FRAME_MS / 1000)
    )

    return speech_seconds >= MIN_SPEECH_SECONDS


# ============================================================
# 10. OPTIONAL BANDPASS FILTER
# ============================================================
#
# We DO NOT use this by default.
#
# The old code always applied it.
# That can remove useful speech information.
# ============================================================

SPEECH_LOW_HZ = 70

SPEECH_HIGH_HZ = 6000


def bandpass_filter(
    audio_data,
    sample_rate,
    low_hz=SPEECH_LOW_HZ,
    high_hz=SPEECH_HIGH_HZ
):

    if audio_data is None:
        return audio_data

    if len(audio_data) < 20:
        return audio_data

    try:

        nyquist = 0.5 * sample_rate

        low = low_hz / nyquist

        high = min(
            high_hz / nyquist,
            0.95
        )

        if low <= 0 or high >= 1 or low >= high:
            return audio_data

        b, a = signal.butter(
            4,
            [low, high],
            btype="band"
        )

        filtered = signal.filtfilt(
            b,
            a,
            audio_data.astype(np.float64)
        )

        return filtered

    except Exception:

        return audio_data


# ============================================================
# 11. OPTIONAL NORMALIZATION
# ============================================================

def normalize_audio(
    audio_data,
    target_peak=0.90
):

    if audio_data is None:
        return audio_data

    try:

        max_val = np.max(
            np.abs(
                audio_data.astype(np.float64)
            )
        )

        if max_val < 1e-6:
            return audio_data

        scale = (
            target_peak *
            32767.0 /
            max_val
        )

        normalized = (
            audio_data.astype(np.float64)
            * scale
        )

        normalized = np.clip(
            normalized,
            -32768,
            32767
        )

        return normalized.astype(np.int16)

    except Exception:

        return audio_data


# ============================================================
# 12. OPTIONAL NOISE REDUCTION
# ============================================================
#
# This is OFF by default.
#
# Why?
#
# Noise reduction can sometimes remove parts of the
# speaker's voice, especially in short recordings.
# ============================================================

def reduce_noise_safely(
    audio_data,
    sample_rate
):

    if audio_data is None:
        return audio_data

    try:

        cleaned = nr.reduce_noise(
            y=audio_data.astype(np.float64),
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.25
        )

        return cleaned

    except Exception:

        return audio_data


# ============================================================
# 13. AUDIO PROCESSING
# ============================================================

def process_audio_buffer(
    audio_bytes,
    enhance_audio=False,
    debug=False
):

    """
    Process microphone WAV.

    Default behavior:
        Original audio → Deepgram

    If enhance_audio=True:
        original
          ↓
        optional filter
          ↓
        gentle noise reduction
          ↓
        normalization
    """

    try:

        # ----------------------------------------------------
        # Read WAV
        # ----------------------------------------------------

        audio_file = io.BytesIO(
            audio_bytes
        )

        sample_rate, audio_data = wav.read(
            audio_file
        )


        # ----------------------------------------------------
        # Convert to mono
        # ----------------------------------------------------

        audio_data = convert_to_mono(
            audio_data
        )


        if audio_data is None:
            return None


        # ----------------------------------------------------
        # Convert datatype
        # ----------------------------------------------------

        if np.issubdtype(
            audio_data.dtype,
            np.integer
        ):

            original_audio = audio_data.copy()

        else:

            # Float audio → int16
            max_value = np.max(
                np.abs(audio_data)
            )

            if max_value <= 1.0:

                original_audio = (
                    audio_data * 32767
                ).astype(np.int16)

            else:

                original_audio = (
                    audio_data
                    .clip(-32768, 32767)
                    .astype(np.int16)
                )


        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration_seconds = get_audio_duration(
            original_audio,
            sample_rate
        )


        if duration_seconds < MIN_DURATION_SECONDS:

            if debug:
                st.warning(
                    f"Recording too short: "
                    f"{duration_seconds:.2f}s"
                )

            return None


        # ----------------------------------------------------
        # Limit extremely long recordings
        # ----------------------------------------------------

        if duration_seconds > MAX_DURATION_SECONDS:

            max_samples = int(
                MAX_DURATION_SECONDS *
                sample_rate
            )

            original_audio = (
                original_audio[:max_samples]
            )


        # ----------------------------------------------------
        # Very basic silence check
        # ----------------------------------------------------
        #
        # This is intentionally conservative.
        # We don't want to throw away quiet speech.
        # ----------------------------------------------------

        energies = frame_energies(
            original_audio,
            sample_rate
        )

        if not energies:

            return None


        maximum_energy = max(energies)


        if maximum_energy < MIN_RMS_ENERGY:

            if debug:
                st.warning(
                    "Recording is extremely quiet."
                )

            return None


        # ====================================================
        # DEFAULT = RAW AUDIO
        # ====================================================

        processed_audio = original_audio.copy()


        # ====================================================
        # OPTIONAL ENHANCEMENT
        # ====================================================

        if enhance_audio:

            # -----------------------------------------------
            # Bandpass
            # -----------------------------------------------

            filtered = bandpass_filter(
                processed_audio,
                sample_rate
            )


            # -----------------------------------------------
            # Gentle noise reduction
            # -----------------------------------------------

            cleaned = reduce_noise_safely(
                filtered,
                sample_rate
            )


            # -----------------------------------------------
            # Normalize
            # -----------------------------------------------

            processed_audio = normalize_audio(
                cleaned
            )


        # ====================================================
        # CREATE WAV FOR DEEPGRAM
        # ====================================================

        output_buffer = io.BytesIO()

        wav.write(
            output_buffer,
            sample_rate,
            processed_audio.astype(np.int16)
        )

        output_buffer.seek(0)

        processed_bytes = (
            output_buffer.read()
        )


        return {
            "processed_bytes": processed_bytes,
            "raw_audio": original_audio,
            "sample_rate": sample_rate,
            "duration": duration_seconds
        }


    except Exception as e:

        if debug:
            st.exception(e)

        return None


# ============================================================
# 14. DEEPGRAM TRANSCRIPTION
# ============================================================

def transcribe_with_deepgram(
    audio_bytes,
    debug=False
):

    """
    Send WAV directly to Deepgram.

    No Deepgram SDK is used.
    """

    headers = {
        "Authorization": (
            f"Token {DEEPGRAM_API_KEY}"
        ),
        "Content-Type": "audio/wav",
    }


    params = {

        # Main STT model
        "model": DEEPGRAM_MODEL,

        # English + Urdu + Roman Urdu
        "language": "multi",

        # Formatting
        "smart_format": "true",

        # Punctuation
        "punctuate": "true",

        # Give us utterance segments
        "utterances": "true",

        # One best alternative
        "alternatives": "1",

        # Our transcription instructions
        "prompt": SYSTEM_PROMPT,
    }


    try:

        response = requests.post(
            DEEPGRAM_URL,
            headers=headers,
            params=params,
            data=audio_bytes,
            timeout=60
        )


    except requests.RequestException as e:

        return {
            "text": "",
            "confidence": None,
            "error": (
                f"Network error while contacting "
                f"Deepgram: {e}"
            )
        }


    # ========================================================
    # HTTP ERROR
    # ========================================================

    if response.status_code != 200:

        try:

            error_data = response.json()

        except Exception:

            error_data = response.text


        return {
            "text": "",
            "confidence": None,
            "error": (
                f"Deepgram HTTP "
                f"{response.status_code}: "
                f"{error_data}"
            )
        }


    # ========================================================
    # JSON
    # ========================================================

    try:

        data = response.json()

    except Exception as e:

        return {
            "text": "",
            "confidence": None,
            "error": (
                f"Deepgram returned invalid JSON: "
                f"{e}"
            )
        }


    # ========================================================
    # EXTRACT CHANNEL
    # ========================================================

    try:

        channels = (
            data
            .get("results", {})
            .get("channels", [])
        )


        if not channels:

            return {
                "text": "",
                "confidence": None,
                "error": "No audio channels returned."
            }


        alternatives = (
            channels[0]
            .get("alternatives", [])
        )


        if not alternatives:

            return {
                "text": "",
                "confidence": None,
                "error": (
                    "Deepgram returned no "
                    "transcription alternatives."
                )
            }


        best = alternatives[0]


        transcript = (
            best.get(
                "transcript",
                ""
            )
            .strip()
        )


        confidence = best.get(
            "confidence",
            None
        )


        # ====================================================
        # GET UTTERANCES
        # ====================================================

        utterances = (
            data
            .get("results", {})
            .get("utterances", [])
        )


        # ====================================================
        # PREFER UTTERANCES WHEN AVAILABLE
        # ====================================================

        if utterances:

            utterance_parts = []

            confidence_values = []


            for utterance in utterances:

                text = (
                    utterance
                    .get("transcript", "")
                    .strip()
                )


                conf = utterance.get(
                    "confidence",
                    None
                )


                if not text:
                    continue


                # Don't throw away too much speech.
                if (
                    conf is not None
                    and conf < DEEPGRAM_CONFIDENCE_THRESHOLD
                ):
                    continue


                utterance_parts.append(
                    text
                )


                if conf is not None:

                    confidence_values.append(
                        float(conf)
                    )


            if utterance_parts:

                transcript = " ".join(
                    utterance_parts
                ).strip()


                if confidence_values:

                    confidence = (
                        sum(confidence_values)
                        /
                        len(confidence_values)
                    )


        # ====================================================
        # EMPTY RESULT
        # ====================================================

        if not transcript:

            return {
                "text": "",
                "confidence": confidence,
                "error": (
                    "Deepgram could not detect "
                    "clear speech."
                )
            }


        # ====================================================
        # ROMAN SCRIPT
        # ====================================================

        transcript = force_roman_script(
            transcript
        )


        return {
            "text": transcript,
            "confidence": confidence,
            "error": None
        }


    except Exception as e:

        if debug:
            st.exception(e)

        return {
            "text": "",
            "confidence": None,
            "error": (
                f"Could not parse Deepgram "
                f"response: {e}"
            )
        }


# ============================================================
# 15. NON-SPEECH EVENT DETECTION
# ============================================================
#
# OPTIONAL FEATURE.
#
# This does NOT control transcription.
#
# This is deliberately separated from STT so that if PANNs
# fails, your speech transcription still works.
# ============================================================

EVENT_LABEL_MAP = {

    "cough":
        "[coughing]",

    "laughter":
        "[laughing]",

    "baby laughter":
        "[laughing]",

    "giggle":
        "[laughing]",

    "snicker":
        "[laughing]",

    "belly laugh":
        "[laughing]",

    "chuckle, chortle":
        "[laughing]",

    "crying, sobbing":
        "[crying]",

    "baby cry, infant cry":
        "[crying]",

    "whimper":
        "[crying]",

    "screaming":
        "[screaming]",

    "clapping":
        "[clapping]",

    "applause":
        "[clapping]",

    "sneeze":
        "[sneezing]",

    "sigh":
        "[sighing]",

    "sniff":
        "[sniffing]",

    "throat clearing":
        "[clearing throat]",

    "whistling":
        "[whistling]",

    "breathing":
        "[breathing]",

    "wheeze":
        "[breathing]",

    "gasp":
        "[breathing]",
}


EVENT_CONFIDENCE_THRESHOLD = 0.35

BREATHING_CONFIDENCE_THRESHOLD = 0.25

BREATHING_LABELS = {
    "breathing",
    "wheeze",
    "gasp"
}

EVENT_WINDOW_SECONDS = 1.5

PANNS_SAMPLE_RATE = 32000


# ============================================================
# 16. LOAD PANNs ONLY WHEN NEEDED
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_sound_event_model():

    try:

        from panns_inference import AudioTagging

        model = AudioTagging(
            checkpoint_path=None,
            device="cpu"
        )

        return model

    except Exception:

        return None


# ============================================================
# 17. SOUND EVENT DETECTION
# ============================================================

def detect_sound_events(
    audio_data,
    sample_rate
):

    try:

        model = load_sound_event_model()

        if model is None:
            return []

    except Exception:

        return []


    try:

        import librosa

        from panns_inference import (
            labels as audioset_labels
        )


        # ----------------------------------------------------
        # Convert int16 → float32
        # ----------------------------------------------------

        audio_float = (
            audio_data
            .astype(np.float32)
            / 32768.0
        )


        # ----------------------------------------------------
        # Resample if required
        # ----------------------------------------------------

        if sample_rate != PANNS_SAMPLE_RATE:

            audio_float = librosa.resample(
                audio_float,
                orig_sr=sample_rate,
                target_sr=PANNS_SAMPLE_RATE
            )


        window_len = int(
            EVENT_WINDOW_SECONDS *
            PANNS_SAMPLE_RATE
        )


        total_len = len(
            audio_float
        )


        raw_events = []


        # ====================================================
        # PROCESS WINDOWS
        # ====================================================

        for start_sample in range(
            0,
            total_len,
            window_len
        ):

            end_sample = min(
                start_sample + window_len,
                total_len
            )


            chunk = audio_float[
                start_sample:end_sample
            ]


            if len(chunk) < (
                PANNS_SAMPLE_RATE * 0.3
            ):

                continue


            try:

                clipwise_output, _ = (
                    model.inference(
                        chunk[None, :]
                    )
                )

            except Exception:

                continue


            probs = clipwise_output[0]


            for idx, prob in enumerate(probs):

                if idx >= len(
                    audioset_labels
                ):
                    continue


                label_name = (
                    audioset_labels[idx]
                    .strip()
                    .lower()
                )


                if label_name not in (
                    EVENT_LABEL_MAP
                ):
                    continue


                threshold = (
                    BREATHING_CONFIDENCE_THRESHOLD
                    if label_name
                    in BREATHING_LABELS
                    else EVENT_CONFIDENCE_THRESHOLD
                )


                if prob < threshold:
                    continue


                start_sec = (
                    start_sample
                    /
                    PANNS_SAMPLE_RATE
                )


                end_sec = (
                    end_sample
                    /
                    PANNS_SAMPLE_RATE
                )


                raw_events.append(
                    (
                        start_sec,
                        end_sec,
                        EVENT_LABEL_MAP[
                            label_name
                        ]
                    )
                )


        # ====================================================
        # MERGE DUPLICATE EVENTS
        # ====================================================

        raw_events.sort(
            key=lambda x: x[0]
        )


        merged = []


        for (
            start_sec,
            end_sec,
            tag
        ) in raw_events:

            if (
                merged
                and merged[-1][2] == tag
                and start_sec
                <= merged[-1][1] + 0.1
            ):

                merged[-1] = (
                    merged[-1][0],
                    end_sec,
                    tag
                )

            else:

                merged.append(
                    (
                        start_sec,
                        end_sec,
                        tag
                    )
                )


        return merged


    except Exception:

        return []


# ============================================================
# 18. MERGE TRANSCRIPTION + EVENTS
# ============================================================

def merge_transcription_and_events(
    transcript,
    events
):

    if not transcript:
        return ""


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We do NOT insert event tags randomly into the
    # transcription.
    #
    # The old system could make output confusing.
    #
    # For now, events are shown separately.
    # --------------------------------------------------------

    return transcript.strip()


# ============================================================
# 22. MAIN UI
# ============================================================

st.title(
    "🎤 LISTENER"
)

st.write(
    "Speech-to-text agent for English, "
    "Urdu and Roman Urdu."
)


# ============================================================
# 23. STATUS
# ============================================================

st.success(
    "🟢 STT engine ready"
)


# ============================================================
# 24. AUDIO SETTINGS
# ============================================================

st.subheader(
    "🎚️ Audio Settings"
)


# ------------------------------------------------------------
# Enhancement is OFF by default.
# ------------------------------------------------------------

enhance_audio = st.checkbox(
    "✨ Enhance audio "
    "(gentle filtering + noise reduction)",
    value=False,
    help=(
        "Keep this OFF initially. "
        "Raw microphone audio is usually the safest "
        "baseline for transcription."
    )
)


# ------------------------------------------------------------
# Event detection is OFF by default.
# ------------------------------------------------------------


st.subheader(
    "🎙️ Voice Input"
)


st.info(
    "Click Start Recording, speak normally, "
    "then click Stop Recording."
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

    key="listener_mic"
)


# ============================================================
# 26. AUDIO RECEIVED
# ============================================================

if audio_output:

    audio_bytes = audio_output.get(
        "bytes"
    )


    if not audio_bytes:

        st.error(
            "❌ No audio data received."
        )

        st.stop()


    # --------------------------------------------------------
    # Count recordings
    # --------------------------------------------------------

    st.session_state.recording_count += 1


    # --------------------------------------------------------
    # Save original recording
    # --------------------------------------------------------

    st.session_state.last_audio = (
        audio_bytes
    )


    # ========================================================
    # 27. PLAY ORIGINAL AUDIO
    # ========================================================
    #
    # THIS IS VERY IMPORTANT FOR DEBUGGING.
    #
    # If playback sounds wrong, the problem is microphone/
    # browser/audio capture.
    #
    # If playback sounds correct but transcript is wrong,
    # the problem is STT.
    # ========================================================

    st.subheader(
        "🔊 Your Recording"
    )


    st.caption(
        "Listen to this recording first. "
        "This is the audio being sent to the STT engine."
    )


    st.audio(
        audio_bytes,
        format="audio/wav"
    )


    # ========================================================
    # 28. PROCESS AUDIO
    # ========================================================

    with st.spinner(
        "⏳ Checking audio..."
    ):

        result = process_audio_buffer(
            audio_bytes,
            enhance_audio=enhance_audio,
        )


    if result is None:

        st.warning(
            "⚠️ The recording was too quiet, "
            "too short, or contained no detectable audio."
        )


    else:

        processed_bytes = (
            result["processed_bytes"]
        )

        raw_audio = (
            result["raw_audio"]
        )

        sample_rate = (
            result["sample_rate"]
        )

        duration = (
            result["duration"]
        )


        # ====================================================
        # 29. AUDIO INFO
        # ====================================================

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
                f"Original size: "
                f"{len(audio_bytes):,} bytes"
            )

            st.write(
                f"Processed size: "
                f"{len(processed_bytes):,} bytes"
            )

            st.write(
                "Enhancement: "
                f"{'ON' if enhance_audio else 'OFF'}"
            )


        # ====================================================
        # 30. OPTIONAL SOUND EVENTS
        # ====================================================

        events = []


        if detect_events:

            with st.spinner(
                "😮 Detecting sound events..."
            ):

                events = detect_sound_events(
                    raw_audio,
                    sample_rate
                )


            if events:

                st.subheader(
                    "🔎 Detected Sound Events"
                )


                for (
                    start,
                    end,
                    tag
                ) in events:

                    st.write(
                        f"{tag} "
                        f"({start:.1f}s - {end:.1f}s)"
                    )


            else:

                st.caption(
                    "No supported sound events detected."
                )



        with st.spinner(
            "⚡ Deepgram is listening..."
        ):

            result = (
                transcribe_with_deepgram(
                    processed_bytes,
                    debug=debug_mode
                )
            )

        if result["error"]:

            st.session_state.last_error = (
                result["error"]
            )


            st.error(
                "❌ Transcription failed"
            )


            with st.expander(
                "Show error details"
            ):

                st.code(
                    result["error"]
                )


        # ====================================================
        # 33. SUCCESS
        # ====================================================

        else:

            transcript = result[
                "text"
            ]


            confidence = result[
                "confidence"
            ]


            if transcript:

                final_text = (
                    merge_transcription_and_events(
                        transcript,
                        events
                    )
                )


                st.session_state.last_transcription = (
                    final_text
                )


                st.session_state.last_confidence = (
                    confidence
                )


                st.session_state.last_error = ""


                st.success(
                    "✅ Transcription complete!"
                )


            else:

                st.warning(
                    "⚠️ Deepgram did not detect "
                    "clear speech."
                )


# ============================================================
# 34. TRANSCRIPTION OUTPUT
# ============================================================

if st.session_state.last_transcription:

    st.divider()


    st.subheader(
        "📝 Transcribed Text"
    )


    # Escape HTML so user speech cannot break UI
    safe_text = html.escape(
        st.session_state.last_transcription
    )


    st.markdown(
        f"""
        <div class="output-card">
            <div class="output-title">
                LISTENER HEARD
            </div>

            <div class="output-text">
                {safe_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Also provide editable plain text box
    st.text_area(
        "Editable text:",
        value=st.session_state.last_transcription,
        height=150,
        key="editable_transcription"
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    if (
        st.session_state.last_confidence
        is not None
    ):

        confidence = (
            st.session_state.last_confidence
        )


        st.caption(
            f"Deepgram confidence: "
            f"{confidence:.1%}"
        )
st.divider()
with st.expander(
    "🐞 Debug / System Information"
):
    st.write(
        f"Deepgram model: {DEEPGRAM_MODEL}"
    )
    st.write(
        "Language mode: multilingual"
    )
    st.write(
        "Deepgram SDK: NOT USED"
    )

    st.write(
        "Deepgram API: HTTP REST"
    )
    st.write(
        f"Recordings this session: "
        f"{st.session_state.recording_count}"
    )
    if st.session_state.last_audio:
        st.write(
            f"Last audio size: "
            f"{len(st.session_state.last_audio):,} bytes"
        )
st.divider()


col1, col2 = st.columns(2)


with col1:

    if st.button(
        "🛑 Lock Text",
        use_container_width=True
    ):

        if (
            st.session_state.last_transcription
        ):

            st.success(
                "✅ Text locked for this session."
            )

        else:

            st.warning(
                "No transcription available."
            )


# ============================================================
# 37. CLEAR TEXT
# ============================================================

with col2:

    if st.button(
        "🗑️ Clear Text",
        use_container_width=True
    ):

        st.session_state.last_transcription = ""

        st.session_state.last_audio = None

        st.session_state.last_confidence = None

        st.session_state.last_error = ""

        st.rerun()
