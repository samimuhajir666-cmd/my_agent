import streamlit as st
import io
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder

# Set up clean web browser layout configuration
st.set_page_config(page_title="OpenAI Voice Agent", page_icon="🎤", layout="centered")

st.title("🎤 VOICE CONFIGURATION AGENT (OpenAI)")
st.write("Tell me your wants, I'm here to listen to everything.")

# ==========================================
# 🔑 OPENAI API KEY CONFIGURATION
# ==========================================
# Sami bhai, your OpenAI key is securely placed here now
OPENAI_API_KEY = "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"

# Initialize the official OpenAI Client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Initialization Error: Please check your OpenAI API key setup. Details: {e}")

st.write("---")
st.subheader("Step 1: Capture Browser Audio")
st.info("Click the button below to record your voice. ")

# Safely capture web audio streams directly via browser API hooks
audio_output = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="🛑 Stop & Process Audio",
    just_once=False,
    use_container_width=True,
    format="wav"
)

# If the user has finished recording audio through the web interface
if audio_output:
    # 1. Display interactive audio player widget right in the UI viewport
    st.subheader("Captured Audio Playback")
    st.audio(audio_output['bytes'], format='audio/wav')
    
    # 2. Process via OpenAI Whisper API
    st.subheader("Step 2: OpenAI Audio Processing")
    with st.spinner("🧠 OpenAI Whisper is transcribing your voice..."):
        try:
            # Convert raw web audio bytes directly into an in-memory binary stream structure
            audio_buffer = io.BytesIO(audio_output['bytes'])
            # Whisper expects a proper filename extension to detect the container type
            audio_buffer.name = "audio.wav"
            
            # Send the audio stream to OpenAI Whisper for Transcription
            # We add a custom prompt to guide Whisper on how to handle the audio style
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_buffer,
                prompt="The user might be speaking in English or Urdu mixed with English. Capture exact words."
            )
            
            st.success("Analysis Complete!")
            
            # 3. Render the Output Layout
            st.markdown("### 💬 Transcription:")
            st.write(transcript_response.text)
            
            st.markdown("### 🔊 Ambient Note:")
            st.caption("Note: OpenAI Whisper focuses heavily on isolating human speech and automatically cleans out background ambient noises like fans or traffic.")
            
        except Exception as ai_error:
            st.error(f"An execution breakdown occurred during OpenAI evaluation: {ai_error}")
            
