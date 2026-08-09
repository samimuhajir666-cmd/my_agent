import io
import os
import base64
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# --- 1. CONFIGURATION (MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="SPEECH TO TEXT - AI Assistant",
    page_icon="🌙",
    layout="centered"
)

# --- 2. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# Look into your .env file or hosting environment variables first
STT_MODEL_KEY = os.getenv("GROQ_API_KEY")

# Look into Streamlit/Hosting Secrets safely without crashing if .env isn't loaded
if not STT_MODEL_KEY:
    try:
        if "GROQ_API_KEY" in st.secrets:
            STT_MODEL_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# Hardcoded fallback token backup string validation (Kept exactly as requested)
if not STT_MODEL_KEY:
    STT_MODEL_KEY = "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"

if not STT_MODEL_KEY:
    st.error(
        "STT Model API key not found. "
        "Add GROQ_API_KEY to your .env file or platform environment configs."
    )
    st.stop()

try:
    # Initialize the Speech-to-Text Client using the active model key
    client = Groq(api_key=STT_MODEL_KEY)
except Exception as e:
    st.error(f"Could not initialize STT Model client: {e}")
    st.stop()

# Assign the absolute target Whisper model for transcription
STT_MODEL = "whisper-large-v3"

# --- 3. LOAD EXTERNAL UI FILES SAFELY ---
def load_css(file_path="style.css"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_html(file_path="index.html"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)

# Load UI Elements
load_css("style.css")
load_html("index.html")

# --- 4. APP STATE & AUDIO INGESTION ---
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

st.subheader("🎤 Voice Input")

# Native Browser-Side Audio Recording Element with Wave Layout Architecture
# This bypasses Streamlit re-run lag and guarantees the waves animate smoothly in the UI.
recording_component_html = """
<div class="recorder-wrapper">
    <button id="recordBtn" class="mic-button-element">🎤 Click to Start Recording</button>
    
    <div id="waveContainer" class="audio-wave-container" style="display: none;">
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
        <div class="wave-bar"></div>
    </div>
</div>

<script>
let mediaRecorder;
let audioChunks = [];
const recordBtn = document.getElementById('recordBtn');
const waveContainer = document.getElementById('waveContainer');
let isRecording = false;

recordBtn.addEventListener('click', async () => {
    if (!isRecording) {
        // Start Recording Process
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            reader.onloadend = () => {
                const base64Audio = reader.result.split(',')[1];
                // Send the voice file back to Streamlit backend storage context
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: base64Audio
                }, '*');
            };
        };
        
        mediaRecorder.start();
        isRecording = true;
        recordBtn.innerText = "🛑 Stop Recording";
        recordBtn.classList.add('recording-active');
        waveContainer.style.display = 'flex';
    } else {
        // Stop Recording Process
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        recordBtn.innerText = "🎤 Click to Start Recording";
        recordBtn.classList.remove('recording-active');
        waveContainer.style.display = 'none';
    }
});
</script>
"""

# Render the dynamic browser recorder
audio_data_base64 = st.components.v1.html(recording_component_html, height=140)

# Process data immediately if an audio stream finishes processing
if audio_data_base64:
    with st.spinner("⚡ Converting speech to text using Whisper Model..."):
        try:
            raw_audio_bytes = base64.b64decode(audio_data_base64)
            audio_file = io.BytesIO(raw_audio_bytes)
            audio_file.name = "recording.wav"
            
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=STT_MODEL,
                prompt=(
                    "The speaker may use English, Urdu, "
                    "or Roman Urdu. Preserve names and technical "
                    "Python/AI terminology accurately."
                ),
                response_format="json",
                temperature=0.0
            )
            
            text_from_voice = getattr(transcription, 'text', '')
            if isinstance(text_from_voice, str):
                text_from_voice = text_from_voice.strip()
            
            if text_from_voice:
                st.session_state.last_transcription = text_from_voice
                st.success("✅ Transcription complete")
            else:
                st.warning("I couldn't detect clear speech.")
        except Exception as e:
            st.error(f"Speech-to-text error: {e}")

# --- 5. UI OUTPUT RENDER DISPLAY ---
if st.session_state.last_transcription:
    st.markdown("### 📝 Transcribed Text")
    st.markdown(
        f'<div class="output-box">{st.session_state.last_transcription}</div>', 
        unsafe_allow_html=True
    )

st.divider()

# Action Layout Interface Control Row
col1, col2 = st.columns(2)

with col1:
    if st.button("🛑 Stop & Process Text", use_container_width=True):
        if st.session_state.last_transcription:
            st.success("Current text processing locked successfully.")
        else:
            st.warning("No live recording stream active to commit.")

with col2:
    if st.button("🗑️ Clear Text", use_container_width=True):
        st.session_state.last_transcription = ""
        st.experimental_rerun()
