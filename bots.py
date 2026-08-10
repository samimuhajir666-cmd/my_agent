import io
import os
import requests
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

load_dotenv()

# =========================================================
# 🔑 API KEYS & CONFIGURATION (ADD YOUR OLLAMA DETAILS HERE)
# =========================================================

# --- GROQ API KEY (For Fast Whisper Speech-to-Text) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        if "GROQ_API_KEY" in st.secrets:
            GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

if not GROQ_API_KEY:
    # ⬇️ ADD / CHANGE YOUR GROQ API KEY HERE ⬇️
    GROQ_API_KEY = "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"

# --- OLLAMA CONFIGURATION (Local or Cloud API) ---
# ⬇️ ADD YOUR OLLAMA API KEY OR URL HERE ⬇️
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Local Ollama URL
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")                 # Leave empty if running locally, or enter key if using hosted Ollama Cloud
OLLAMA_MODEL = "llama3"                                           # Change to "llama3:8b", "mistral", or your downloaded Ollama model name

if not GROQ_API_KEY:
    st.error("Groq API Key missing. Please check your configuration.")
    st.stop()

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Could not initialize Groq client: {e}")
    st.stop()

# ==========================================
# 🎯 STT MODEL & LLM SYSTEM PROMPTS
# ==========================================
STT_MODEL = "whisper-large-v3-turbo"

# Clean STT Prompt - pure script hint, zero rules
STT_PROMPT = "Roman Urdu, English, numbers 1 2 3, plus, minus, equal, kya haal hai, main theek hoon."

# --- LLM AGENT SYSTEM PROMPT (Ollama Agent Rules) ---
LLM_SYSTEM_PROMPT = (
    "You are a smart voice to text generator AI assistant and math listener. "
    "Rules:\n"
    "1. Always reply ONLY in Roman Urdu (English letters) or English or english alphabetic scripts .\n"
    "2. If the user asks a math question or calculation, don,t solve it just listen and generate text as you listen.\n"
    "3. Don,t Keep  answers direct just generate text .\n"
    "4. Do NOT use Urdu Arabic script or Hindi Devanagari script."
)

# --- BACKGROUND NOISE & SILENCE CHECK ---
def process_audio_buffer(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)
            
        rms_energy = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
        
        if rms_energy < 15.0:
            return None  # Pure silence or low noise
            
        cleaned_audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.75)
        
        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, cleaned_audio_data.astype(np.int16))
        output_buffer.seek(0)
        
        return output_buffer.read()
    except Exception:
        return audio_bytes

# --- OLLAMA LLM CALL FUNCTION ---
def query_ollama_llm(user_text):
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
        
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{LLM_SYSTEM_PROMPT}\n\nUser Question: {user_text}\n\nAnswer in Roman Urdu:",
        "stream": False
    }
    
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"Ollama Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"Could not connect to Ollama: {e}"

# --- UI SETUP ---
st.set_page_config(
    page_title="Voice & Math AI Assistant",
    page_icon="🎤",
    layout="centered"
)

def load_css(file_path="style.css"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_html(file_path="index.html"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                st.html(f.read())
            except Exception:
                st.markdown(f.read(), unsafe_allow_html=True)

load_css("style.css")
load_html("index.html")

# --- SESSION STATE ---
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""
if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""

st.subheader("🎤 Voice Input")

audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="listener_mic"
)

# --- AUDIO PROCESSING & LLM PIPELINE ---
if audio_output:
    audio_bytes = audio_output.get("bytes")
    if not audio_bytes:
        st.error("No audio data received.")
        st.stop()
        
    with st.spinner("⏳ Analyzing sound levels..."):
        processed_bytes = process_audio_buffer(audio_bytes)
        
    if processed_bytes is None:
        st.warning("⚠️ Noise or silence detected. Please speak clearly into the mic.")
    else:
        with st.spinner("⚡ Transcribing speech..."):
            try:
                audio_file = io.BytesIO(processed_bytes)
                audio_file.name = "recording.wav"
                
                # Step 1: STT via Whisper Turbo
                transcription = groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model=STT_MODEL,
                    prompt=STT_PROMPT,
                    response_format="json",
                    temperature=0.0
                )
                
                text_from_voice = transcription.text.strip()
                
                if text_from_voice and len(text_from_voice) > 1:
                    st.session_state.last_transcription = text_from_voice
                    
                    # Step 2: Query Ollama LLM for reasoning/math answer
                    with st.spinner("🤖 Ollama AI Agent thinking..."):
                        llm_reply = query_ollama_llm(text_from_voice)
                        st.session_state.llm_response = llm_reply
                        
                    st.success("✅ Complete!")
                else:
                    st.warning("⚠️ Could not detect clear speech. Try again.")
                    
            except Exception as e:
                st.error(f"Processing error: {e}")

# --- DISPLAY OUTPUT ---
if st.session_state.last_transcription:
    st.markdown("### 🎙️ Heard Voice (Transcribed):")
    st.info(st.session_state.last_transcription)

if st.session_state.llm_response:
    st.markdown("### 🤖 AI Agent Answer (Ollama):")
    st.markdown(
        f"""
        <div class="output-card">
            <div class="output-text">{st.session_state.llm_response}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🛑 Lock Answer", use_container_width=True):
        if st.session_state.llm_response:
            st.success("Saved in session state.")
        else:
            st.warning("No data available.")

with col2:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.last_transcription = ""
        st.session_state.llm_response = ""
        st.rerun()
