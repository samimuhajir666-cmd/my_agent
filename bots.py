import io
import os
import re
import numpy as np
import scipy.signal as signal
import noisereduce as nr
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode

# Strict version-controlled initialization for Deepgram SDK V3
try:
    from deepgram import DeepgramClient, PrerecordedOptions, FileSource
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False

load_dotenv()

# ==============================================================================
# 🔑 AGENT SECURITY PROTOCOL & SYSTEM KEYS MATRIX
# ==============================================================================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    try:
        if "DEEPGRAM_API_KEY" in st.secrets:
            DEEPGRAM_API_KEY = st.secrets["DEEPGRAM_API_KEY"]
    except Exception:
        pass

if not DEEPGRAM_API_KEY:
    st.error("❌ DEEPGRAM_API_KEY missing! Please check your .env file or Streamlit Secrets.")
    st.stop()

if not DEEPGRAM_AVAILABLE:
    st.error("❌ Deepgram modern Python SDK is missing. Please run pip install deepgram-sdk")
    st.stop()

try:
    # Initialized using explicit keyword architecture to resolve the positional argument crash
    deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
except Exception as init_error:
    st.error(f"❌ Failed to boot up Deepgram Core Client: {init_error}")
    st.stop()

# Operational constraints for the AI Agent
SYSTEM_PROMPT = (
    "Roman Urdu, Arabic, and English mixed linguistic pipeline. Normalizing non-ascii "
    "characters to plain text vectors while filtering environmental background sound tags."
)

def force_roman_script(text):
    """Converts native Urdu/Arabic scripts or accented characters into clean Romanized English text."""
    if not text:
        return text
    # Check for any non-ascii alphabet systems
    if bool(re.search(r'[^\x00-\x7F]', text)):
        return unidecode(text)
    return text

# ==============================================================================
# 🎚️ ADVANCED ACOUSTIC HARDWARE CLEANUP & LIMITER (ANTI-SHOUTING)
# ==============================================================================
SPEECH_LOW_HZ = 85.0
SPEECH_HIGH_HZ = 3500.0

def apply_hardware_acoustic_filters(raw_bytes, sensitivity=0.5):
    """
    Simulates human ears by suppressing high-volume screaming spikes (clipping)
    and dropping background mechanical noise floors by 85%.
    """
    import scipy.io.wavfile as wav
    
    # Read the audio byte arrays safely into internal matrix memory
    sample_rate, data = wav.read(io.BytesIO(raw_bytes))
    
    # Normalize integer data streams to unified floating points
    if data.dtype == np.int16:
        audio_float = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio_float = data.astype(np.float32) / 2147483648.0
    else:
        audio_float = data.astype(np.float32)
        
    # Collapse multi-channel stereo configurations down to clean mono paths
    if len(audio_float.shape) > 1:
        audio_float = np.mean(audio_float, axis=1)
        
    # ANTI-SHOUTING ENGINE: Smooth out extreme vocal peaks to prevent audio distortion
    max_peak = np.max(np.abs(audio_float))
    if max_peak > 0.75:
        audio_float = np.tanh(audio_float / max_peak) * 0.75
        
    # BUTTERWORTH BANDPASS FILTER: Cut out low engine hums and high static hiss strings
    nyquist = 0.5 * sample_rate
    low_cut = SPEECH_LOW_HZ / nyquist
    high_cut = min(SPEECH_HIGH_HZ / nyquist, 0.99)
    b, a = signal.butter(4, [low_cut, high_cut], btype="band")
    filtered_signal = signal.filtfilt(b, a, audio_float)
    
    # ENVIRONMENTAL BACKGROUND NOISE CANCELLATION
    reduced_noise = nr.reduce_noise(
        y=filtered_signal, 
        sr=sample_rate, 
        prop_decrease=0.85, # Suppress ambient noise floor by 85%
        n_fft=1024
    )
    
    # Cast safely back into industry-standard 16-bit PCM WAV parameters
    clean_pcm = np.clip(reduced_noise * 32768.0, -32768, 32767).astype(np.int16)
    
    output_io = io.BytesIO()
    wav.write(output_io, sample_rate, clean_pcm)
    return output_io.getvalue()

# ==============================================================================
# 🎙️ MULTI-LANGUAGE DEEPGRAM TRANSCRIBE INTEGRATION (STT CORE)
# ==============================================================================
def execute_agent_transcription(processed_wav_bytes):
    """Queries Deepgram Nova-3 Multi-Language models with conversational tuning rules."""
    try:
        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            punctuate=True,
            utterances=True,
            language="multi", # Instructs the AI to trace mixed Roman Urdu/English conversations fluidly
        )
        
        payload: FileSource = {"buffer": processed_wav_bytes}
        response = deepgram_client.listen.rest.v("1").transcribe_file(payload, options)
        
        # Accessing nested channels safe metrics structures securely
        channel = response.results.channels[0]
        alternative = channel.alternatives[0]
        
        raw_text = getattr(alternative, "transcript", "").strip()
        confidence = getattr(alternative, "confidence", 0.0)
        
        # Apply the Romanization script normalization layer
        final_roman_text = force_roman_script(raw_text)
        return final_roman_text, confidence
    except Exception as api_error:
        st.error(f"Deepgram processing array connection error: {api_error}")
        return "", 0.0

# ==============================================================================
# 🖥️ STREAMLIT FRONTEND WEB APPLICATION DASHBOARD
# ==============================================================================
st.set_page_config(page_title="Multi-Language STT Agent", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Language AI Speech-To-Text Agent")
st.caption("Production Build: Audio Preprocessing Engine (Anti-Shouting Limiter + 85% Noise Filter)")

st.sidebar.header("⚙️ Agent Environmental Controls")
noise_reduction_sensitivity = st.sidebar.slider("Background Cancellation Power", 0.1, 1.0, 0.7, step=0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Operational Directive:**\n`" + SYSTEM_PROMPT + "`")

# Main dual-input configuration dropzones
uploaded_file = st.file_uploader("Upload an Audio File (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])
st.write("✨ **-- OR SPEAK LIVE TO THE AGENT --** ✨")
recorded_audio = mic_recorder(start_prompt="🔴 Wake Agent (Record Live)", stop_prompt="⏹️ Submit Audio", key="live_agent_mic")

audio_payload_bytes = None

if uploaded_file is not None:
    audio_payload_bytes = uploaded_file.read()
elif recorded_audio is not None:
    audio_payload_bytes = recorded_audio['bytes']

if audio_payload_bytes is not None:
    st.info("📁 Audio matrix received. Initiating hardware processing filters...")
    
    try:
        # Step 1: Filter background noise floor and balance screaming thresholds
        cleaned_bytes = apply_hardware_acoustic_filters(audio_payload_bytes, sensitivity=noise_reduction_sensitivity)
        
        # Step 2: Query Deepgram REST API arrays using our clean processing wave
        with st.spinner("🧠 Agent is transcribing and converting linguistic matrices..."):
            transcript, confidence_score = execute_agent_transcription(cleaned_bytes)
            
        st.success("🎉 Processing complete!")
        
        # UI metric telemetry cards display
        meta_col1, meta_col2 = st.columns(2)
        meta_col1.metric("🌐 Language Framework Mode", "MULTI (Mixed English/Roman Urdu)")
        meta_col2.metric("📊 Agent Decoding Confidence", f"{confidence_score * 100:.2f}%")
        
        st.markdown("### 📝 Cleaned Output Transcript (Forced Roman Script):")
        if transcript:
            st.code(transcript, language="text")
            
            # Offer standard production txt subtitle export download options cleanly
            st.download_button(
                label="📥 Download Clean Text File",
                data=transcript,
                file_name=f"agent_output_{int(time.time())}.txt",
                mime="text/plain"
            )
        else:
            st.warning("⚠️ The agent could not decipher any valid words. Check input device thresholds.")
            
    except Exception as pipeline_error:
        st.error(f"Fatal System Pipeline Crash: {pipeline_error}")
        st.info("💡 Tip: Ensure your audio file is a valid standard WAV or MP3 recording.")
