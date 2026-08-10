import io
import os
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

load_dotenv()

# --- STT MODEL KEY INITIALIZATION ---
STT_MODEL_KEY = os.getenv("GROQ_API_KEY")

if not STT_MODEL_KEY:
    try:
        if "GROQ_API_KEY" in st.secrets:
            STT_MODEL_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

if not STT_MODEL_KEY:
    STT_MODEL_KEY = "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"

if not STT_MODEL_KEY:
    st.error("GROQ_API_KEY not found in .env or Streamlit Secrets.")
    st.stop()

try:
    client = Groq(api_key=STT_MODEL_KEY)
except Exception as e:
    st.error(f"Could not initialize Groq client: {e}")
    st.stop()

# ==========================================
# 🎯 MODEL & PROMPT CONFIGURATION
# ==========================================
STT_MODEL = "whisper-large-v3-turbo"

# Pure script hint (No rules, no AI persona)
SYSTEM_PROMPT = """
You are a voice-to-text transcription assistant. Your only job is to listen to audio and output exactly what was spoken as text.

STRICT RULES:

1. OUTPUT SCRIPT: Always write ONLY in Roman script (English alphabet letters). This includes Roman Urdu and English words exactly as spoken.
   - Never use Urdu/Arabic script.
   - Never use Hindi/Devanagari script.
   - If a word is Urdu, spell it phonetically in English letters (e.g., "aap kaisay hain", not "آپ کیسے ہیں").

2. TRANSCRIBE ONLY — DO NOT SOLVE OR INTERPRET:
   - If the speaker says a math problem or numbers (e.g., "2 plus 2" or "1, 2, 3"), just transcribe the words/numbers exactly as spoken.
   - Do NOT calculate, solve, or explain anything.
   - Do NOT answer questions asked in the audio — only transcribe them as text.

3. NO COMMENTARY:
   - Do not add explanations, greetings, opinions, or extra text.
   - Output ONLY the transcribed text — nothing before or after it.

4. HANDLING UNCLEAR AUDIO / BACKGROUND NOISE:
   - If part of the audio is unclear, muffled, or has background noise, transcribe whatever words you can confidently make out.
   - For portions that are truly inaudible or indistinguishable, insert the placeholder: [inaudible] — do not guess random words to fill gaps.
   - Never say "I couldn't understand" or "wrong input" — just transcribe what's audible and mark unclear parts with [inaudible].
   - If there is background noise but no speech (silence, music, static), output: [no speech detected]

5. NUMBERS AND MATH EXPRESSIONS:
   - Write numbers as spoken (either digits or words is fine, but be consistent) — e.g., "one two three" or "1 2 3" as heard.
   - Do not convert spoken math into solved equations or answers.

6. FORMATTING:
   - Keep sentence breaks natural, based on pauses in speech.
   - Use basic punctuation (commas, periods, question marks) only where clearly implied by intonation — don't over-punctuate.
"""
# --- AUDIO PROCESSING FOR NOISE REDUCTION ---
def process_audio_buffer(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)
            
        rms_energy = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
        
        # Silence check
        if rms_energy < 15.0:
            return None
            
        cleaned_audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.75)
        
        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, cleaned_audio_data.astype(np.int16))
        output_buffer.seek(0)
        
        return output_buffer.read()
    except Exception:
        return audio_bytes

st.set_page_config(
    page_title="SPEECH TO TEXT",
    page_icon="🎤",
    layout="centered"
)

# --- LOAD HTML & CSS SAFELY ---
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

st.subheader("🎤 Voice Input")

audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="listener_mic"
)

# --- AUDIO PROCESSING LOGIC ---
if audio_output:
    audio_bytes = audio_output.get("bytes")
    if not audio_bytes:
        st.error("No audio data received.")
        st.stop()
        
    with st.spinner("⏳ Processing sound..."):
        processed_bytes = process_audio_buffer(audio_bytes)
        
    if processed_bytes is None:
        st.warning("⚠️ Noise or silence detected. Please speak clearly into the mic.")
    else:
        with st.spinner("⚡ Transcribing speech..."):
            try:
                audio_file = io.BytesIO(processed_bytes)
                audio_file.name = "recording.wav"
                
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model=STT_MODEL,
                    prompt=SYSTEM_PROMPT,
                    response_format="json",
                    temperature=0.0
                )
                
                text_from_voice = transcription.text.strip()
                
                if text_from_voice and len(text_from_voice) > 1:
                    st.session_state.last_transcription = text_from_voice
                    st.success("✅ Complete!")
                else:
                    st.warning("⚠️ Could not detect clear speech.")
                    
            except Exception as e:
                st.error(f"Transcription error: {e}")

# --- DISPLAY OUTPUT ---
if st.session_state.last_transcription:
    st.markdown("### 📝 Transcribed Text")
    st.markdown(
        f"""
        <div class="output-card">
            <div class="output-title">Result:</div>
            <div class="output-text">{st.session_state.last_transcription}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🛑 Lock Text", use_container_width=True):
        if st.session_state.last_transcription:
            st.success("Text saved.")
        else:
            st.warning("No recorded text available.")

with col2:
    if st.button("🗑️ Clear Text", use_container_width=True):
        st.session_state.last_transcription = ""
        st.rerun()
