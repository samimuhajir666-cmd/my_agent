import io
import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 2. GET GROQ API KEY
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# If .env doesn't contain the key, try Streamlit Secrets
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None


# Stop if API key doesn't exist
if not GROQ_API_KEY:
    st.error(
        "❌ GROQ_API_KEY was not found.\n\n"
        "Put your new key inside .env or Streamlit Secrets."
    )
    st.stop()


# ============================================================
# 3. GROQ CLIENT
# ============================================================

try:
    client = Groq(api_key=GROQ_API_KEY)

except Exception as e:
    st.error(f"❌ Could not initialize Groq: {e}")
    st.stop()


# ============================================================
# 4. SPEECH-TO-TEXT SETTINGS
# ============================================================

# Accuracy-focused model
STT_MODEL = "whisper-large-v3"


# Keep the prompt simple.
# Do NOT force the model to rewrite speech.
STT_PROMPT = """
The speaker may use English, Urdu, Roman Urdu, or a mixture.
Transcribe the actual spoken words as accurately as possible.
Keep names, numbers, Python terms, and technical words accurate.
Do not answer the speaker's question.
Only transcribe the speech.
"""


# ============================================================
# 5. STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="LISTENER - Speech to Text",
    page_icon="🎤",
    layout="centered"
)


# ============================================================
# 6. TITLE
# ============================================================

st.title("🎤 LISTENER")

st.write(
    "Speak naturally and LISTENER will convert your speech "
    "into text."
)


# ============================================================
# 7. SESSION STATE
# ============================================================

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""


# ============================================================
# 8. MICROPHONE
# ============================================================

st.subheader("🎙️ Voice Input")

st.info(
    "Click Start Recording → speak clearly → "
    "click Stop Recording."
)


audio_output = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="listener_microphone"
)


# ============================================================
# 9. PROCESS AUDIO
# ============================================================

if audio_output:

    # Get audio bytes
    audio_bytes = audio_output.get("bytes")

    if not audio_bytes:

        st.error("❌ No audio was received.")
        st.stop()


    # --------------------------------------------------------
    # IMPORTANT:
    # We are NOT using noisereduce here.
    #
    # We send the original microphone recording directly
    # to Whisper.
    # --------------------------------------------------------

    audio_file = io.BytesIO(audio_bytes)

    # Groq needs a filename on the file-like object
    audio_file.name = "recording.wav"


    # ========================================================
    # 10. SEND AUDIO TO WHISPER
    # ========================================================

    with st.spinner("🎧 Listening and transcribing..."):

        try:

            transcription = client.audio.transcriptions.create(
                file=audio_file,

                model=STT_MODEL,

                prompt=STT_PROMPT,

                response_format="json",

                # More deterministic transcription
                temperature=0.0
            )


            # Get text
            text = transcription.text.strip()


            # =================================================
            # 11. SAVE RESULT
            # =================================================

            if text:

                st.session_state.last_transcription = text

            else:

                st.warning(
                    "⚠️ I couldn't detect clear speech."
                )


        except Exception as e:

            st.error(
                f"❌ Transcription error:\n\n{e}"
            )


# ============================================================
# 12. DISPLAY RESULT
# ============================================================

if st.session_state.last_transcription:

    st.divider()

    st.subheader("📝 Transcribed Text")

    st.text_area(
        "What LISTENER heard:",
        value=st.session_state.last_transcription,
        height=150
    )


# ============================================================
# 13. CLEAR BUTTON
# ============================================================

st.divider()

if st.button(
    "🗑️ Clear Text",
    use_container_width=True
):

    st.session_state.last_transcription = ""

    st.rerun()


# ============================================================
# 14. TESTING INFORMATION
# ============================================================

st.divider()

st.caption(
    "STT Model: Whisper Large-v3 | "
    "Original microphone audio | "
    "No aggressive noise reduction"
)
