import streamlit as st
import io
from groq import Groq
from streamlit_mic_recorder import mic_recorder

# Set up clean web browser layout configuration
st.set_page_config(page_title="Voice & Ambient Agent", page_icon="🎤", layout="centered")

st.title("🎤 VOICE CONFIGURATION AGENT (Groq Cloud)")
st.write("Tell me your wants, I'm here to listen to everything.")

# ==========================================
# 🔑 GROQ API KEY CONFIGURATION
# ==========================================
GROQ_API_KEY = "gsk_i3QV1qoGHWHDAcASXfd8WGdyb3FYEL1NexDk4G4UQNHy1FnxNV9Ggroq"

# Initialize the official Groq Client
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Initialization Error: Please check your Groq API key setup. Details: {e}")

st.write("---")
st.subheader("Step 1: Capture Browser Audio")
st.info("Click the button below to record your voice. Make sure to allow microphone access in your browser!")

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
    
    # 2. Process via Groq Cloud Audio API (Whisper Large V3)
    st.subheader("Step 2: Groq Audio Processing")
    with st.spinner("⚡ Groq Whisper is transcribing your voice at lightning speed..."):
        try:
            # Convert raw web audio bytes into a file-like object structure for Groq
            audio_file = ("audio.wav", audio_output['bytes'])
            
            # Setup specialized prompts forcing contextual acoustic matrix assessments
            prompt_instruction = (
                "You are an advanced voice support agent. Analyze this audio file carefully. "
                "Listen to the user's voice clearly and identify the exact words as spoken. "
                "Also identify the ambient sounds in the background. "
                "The important thing is to analyze the raw real user audio and background sounds, not just the text code.\n\n"
                "convert user's voice to text and listen very carefully and identify the exact words as spoken. "
                "generate text as you hear the user's voice and also identify the ambient sounds in the background. "
                "The important thing is to analyze the raw real user audio and background sounds, not just the text code.\n\n"
                "Provide your analysis in the following strict format:\n\n"
                "### 💬 Transcription:\n"
                "Transcribe the exact words spoken by the user in the language they spoke.\n\n"
                "### 🔊 Ambient Sound & Context:\n"
                "Describe what is happening in the background. Identify specific sounds "
                "(e.g., fan humming, traffic honking, typing, papers shuffling, absolute silence) "
                "and determine the user's environment."
            )
            
            # Send the audio data straight to Groq's Whisper API with system prompt injection
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3", 
                prompt=prompt_instruction,
                response_format="json"
            )
            
            st.success("Analysis Complete!")
            
            # 3. Render the Output Layout dynamically on the browser screen
            st.markdown("### 💬 Transcription:")
            st.write(transcription.text)
            
        except Exception as ai_error:
            st.error(f"An execution breakdown occurred during Groq evaluation: {ai_error}")
