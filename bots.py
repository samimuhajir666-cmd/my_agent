import io
import os
import base64
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_js_eval import streamlit_js_eval

# --- 1. CONFIGURATION (MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="SPEECH TO TEXT - AI Assistant",
    page_icon="🌙",
    layout="centered"
)

# --- 2. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

STT_MODEL_KEY = os.getenv("GROQ_API_KEY")

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
    client = Groq(api_key=STT_MODEL_KEY)
except Exception as e:
    st.error(f"Could not initialize STT Model client: {e}")
    st.stop()

STT_MODEL = "whisper-large-v3-turbo"

# --- 3. LOAD EXTERNAL UI FILES SAFELY ---
def load_css(file_path="style.css"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_html(file_path="index.html"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)

load_css("style.css")
load_html("index.html")

# --- 4. APP STATE & AUDIO INGESTION ---
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

st.subheader("🎤 Voice Input")

# Pure JavaScript Recorder that returns data to Python safely via streamlit_js_eval
js_recorder_code = """
(async () => {
    if (!window.myRecorderState) {
        window.myRecorderState = { isRecording: false, mediaRecorder: null, chunks: [], audioBase64: null };
    }

    // Create container wrapper if it doesn't exist
    let wrapper = document.getElementById('custom-recorder-root');
    if (!wrapper) {
        wrapper = document.createElement('div');
        wrapper.id = 'custom-recorder-root';
        wrapper.className = 'recorder-wrapper';
        
        let btn = document.createElement('button');
        btn.id = 'jsRecordBtn';
        btn.className = 'mic-button-element';
        btn.innerText = '🎤 Click to Start Recording';
        
        let wave = document.createElement('div');
        wave.id = 'jsWaveContainer';
        wave.className = 'audio-wave-container';
        wave.style.display = 'none';
        for(let i=0; i<5; i++) {
            let bar = document.createElement('div');
            bar.className = 'wave-bar';
            wave.appendChild(bar);
        }
        
        wrapper.appendChild(btn);
        wrapper.appendChild(wave);
        
        // Find Streamlit's element container to attach cleanly
        let target = document.querySelector('.stMarkdown') || document.body;
        target.appendChild(wrapper);
        
        btn.addEventListener('click', async () => {
            let state = window.myRecorderState;
            if (!state.isRecording) {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                state.mediaRecorder = new MediaRecorder(stream);
                state.chunks = [];
                
                state.mediaRecorder.ondataavailable = e => state.chunks.push(e.data);
                state.mediaRecorder.onstop = () => {
                    const blob = new Blob(state.chunks, { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.readAsDataURL(blob);
                    reader.onloadend = () => {
                        state.audioBase64 = reader.result.split(',')[1];
                        // Force a click on a hidden layout tracker to send data back
                        btn.setAttribute('data-output', state.audioBase64);
                    };
                };
                
                state.mediaRecorder.start();
                state.isRecording = true;
                btn.innerText = "🛑 Stop Recording";
                btn.classList.add('recording-active');
                wave.style.display = 'flex';
            } else {
                state.mediaRecorder.stop();
                state.mediaRecorder.stream.getTracks().forEach(t => t.stop());
                state.isRecording = false;
                btn.innerText = "🎤 Click to Start Recording";
                btn.classList.remove('recording-active');
                wave.style.display = 'none';
            }
        });
    }

    // Polling mechanism to return data to Streamlit runtime variable smoothly
    return new Promise((resolve) => {
        let checkInterval = setInterval(() => {
            let btn = document.getElementById('jsRecordBtn');
            if (btn && btn.getAttribute('data-output')) {
                let data = btn.getAttribute('data-output');
                btn.removeAttribute('data-output');
                clearInterval(checkInterval);
                resolve(data);
            }
        }, 500);
    });
})()
"""

# Safely capture the return string directly from the browser window instance
audio_data_base64 = streamlit_js_eval(js_expressions=js_recorder_code, key="browser_audio_bridge")

# Process data immediately if a secure string is received from JavaScript
if audio_data_base64 and isinstance(audio_data_base64, str):
    with st.spinner("⚡ Processing audio and clearing background noise..."):
        try:
            raw_audio_bytes = base64.b64decode(audio_data_base64)
            audio_file = io.BytesIO(raw_audio_bytes)
            audio_file.name = "recording.wav"
            
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=STT_MODEL,
                prompt=(
                    "Environment Note: This audio contains background noise, room echo, hums, and microphone crackle. "
                    "Ignore all background sounds, static hiss, breathing artifacts, and ambient noise completely. "
                    "Transcribe ONLY the explicit spoken human words. "
                    "Language Profile: The user speaks fluently in English, Urdu, or mixed Roman Urdu. "
                    "Do not try to translate or fix the vocabulary—transcribe exactly what was said literally."
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
                st.warning("I couldn't detect clear speech over the room background noise.")
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
