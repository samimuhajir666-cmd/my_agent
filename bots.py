import streamlit as st
import io
import openai
from streamlit_mic_recorder import mic_recorder

# Set up clean web browser layout configuration
st.set_page_config(page_title="Voice & Ambient Agent", page_icon="🎤", layout="centered")

st.title("🎤 VOICE CONFIGRATION AGENT")
st.write("Tell me your wants, I'm here to listen to your everything.")

# ==========================================
# 🔑 API KEY CONFIGURATION
# ==========================================
# Sami bhai, apni Gemini API Key yahan niche double quotes (" ") ke andar paste kar dein!
OPENAI_API_KEY="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"

# Initialize the modern Google GenAI Client directly with the token
try:
    client = genai.Client(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Initialization Error: Please check your API key setup. Details: {e}")

st.write("---")
st.subheader("Step 1: Capture Browser Audio")
st.info("Click the button below to record your voice. Make sure to allow microphone access in your browser!")

# Safely capture web audio streams directly via browser API hooks
audio_output = mic_recorder(
    start_prompt="🎤 Start Recording ",
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
    
    # 2. Process via the Native Multimodal AI Framework
    st.subheader("Step 2: Multimodal AI Processing")
    with st.spinner("🧠 Analyzing audio frequency waves and environmental acoustics..."):
        try:
            # Convert raw web audio bytes directly into an in-memory binary stream structure
            audio_buffer = io.BytesIO(audio_output['bytes'])
            
            # Stream the temporary byte structure straight up to Google Cloud Servers
            uploaded_file = client.files.upload(
                file=audio_buffer,
                mime_type="audio/wav"
            )
            
            # Setup specialized prompts forcing contextual acoustic matrix assessments
            prompt = (
                "You are an advanced voice support agent. Analyze this audio file carefully. "
                "Listen to the user's voice clearly and identify the exact words as spoken. "
                "Also identify the ambient sounds in the background. "
                "The important thing is to analyze the raw real user audio and background sounds, not just the text code.\n\n"
                "convert user,s vioce to text and listen very clearfully and identify the exact words as spoken. "
                "generate text as you hear the user,s voice and also identify the ambient sounds in the background. "
                "The important thing is to analyze the raw real user audio and background sounds, not just the text code.\n\n"
                "Provide your analysis in the following strict format:\n\n"
                "### 💬 Transcription:\n"
                "Transcribe the exact words spoken by the user in the language they spoke.\n\n"
                "### 🔊 Ambient Sound & Context:\n"
                "Describe what is happening in the background. Identify specific sounds "
                "(e.g., fan humming, traffic honking, typing, papers shuffling, absolute silence) "
                "and determine the user's environment."
            )
            
            # FIXED: Passing both the uploaded audio file AND the prompt explicitly inside contents array
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[uploaded_file, prompt]
            )
            
            # Render the results elegantly via Markdown elements
            st.success("Analysis Complete!")
            st.markdown(response.text)
            
        except Exception as ai_error:
            st.error(f"An execution breakdown occurred during AI evaluation: {ai_error}")
