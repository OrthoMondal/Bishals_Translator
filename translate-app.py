import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch
import os

# Set up page configuration for a professional look
st.set_page_config(
    page_title="Bishal's Advanced Translator", 
    page_icon="🌐", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS to inject a clean, modern card-based theme
st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        max-width: 750px;
    }
    /* Modern title style */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #FF4B4B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6a6a6a;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    /* Card design for output */
    .output-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #FF4B4B;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .output-header {
        font-weight: 700;
        color: #262730;
        margin-bottom: 0.5rem;
    }
    .output-text {
        font-size: 1.2rem;
        color: #111111;
        white-space: pre-wrap;
    }
    </style>
""", unsafe_allow_html=True)

# Render Header UI
st.markdown('<h1 class="main-title">🌐 Bishal\'s Translator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Next-generation local English ↔ Bangla machine translation platform</p>', unsafe_allow_html=True)

# Set the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Paths (matching your GitHub repository layout)
e2b_model_path = './E2B/checkpoint-4000'
e2b_tokenizer_path = './E2B/checkpoint-4000'

b2e_model_path = './B2E/checkpoint-4000'
b2e_tokenizer_path = './B2E/checkpoint-4000'

# Load Models with Caching
@st.cache_resource
def load_e2b_model():
    tokenizer = AutoTokenizer.from_pretrained(e2b_tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(e2b_model_path)
    model.to(device)
    return tokenizer, model

@st.cache_resource
def load_b2e_model():
    tokenizer = AutoTokenizer.from_pretrained(b2e_tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(b2e_model_path)
    model.to(device)
    return tokenizer, model

# Display a clean loading message inside an expander so it doesn't clutter the UI
with st.spinner("⚡ Initializing neural engine... Please hold on."):
    e2b_tokenizer, e2b_model = load_e2b_model()
    b2e_tokenizer, b2e_model = load_b2e_model()

# Translation logic core
def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output_tokens = model.generate(**inputs, max_length=128)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Offensive language loading & detection
@st.cache_data
def load_offensive_words(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f if line.strip()]
    except Exception:
        return []

def contains_offensive_language(text, offensive_words):
    if not offensive_words: 
        return False
    text = text.lower()
    for word in offensive_words:
        if re.search(r'\b' + re.escape(word) + r'\b', text):
            return True
    return False

offensive_words_list = load_offensive_words('./offensive_words.txt')

# Modern UI Structure: Using Tabs instead of Radio buttons for navigation
tab1, tab2 = st.tabs(["🇺🇸 English → 🇧🇩 Bangla", "🇧🇩 Bangla → 🇺🇸 English"])

# Layout setup for Execution
input_text = ""
direction = ""

with tab1:
    en_text = st.text_area("Source Text (English):", placeholder="Type or paste your English text here...", key="en_input", height=130)
    if en_text:
        input_text = en_text
        direction = "E2B"

with tab2:
    bn_text = st.text_area("Source Text (Bangla):", placeholder="এখানে আপনার বাংলা লেখাটি লিখুন বা পেস্ট করুন...", key="bn_input", height=130)
    if bn_text:
        input_text = bn_text
        direction = "B2E"

# Modern Action Control Area
if input_text.strip():
    has_offensive = contains_offensive_language(input_text, offensive_words_list)
    
    if has_offensive:
        # Show interactive modern warning block
        st.warning("⚠️ **Content Warning:** The system has detected language that might be considered offensive or inappropriate.")
        
        # User choice checkbox
        override_consent = st.checkbox("I understand and want to translate this content regardless.")
        
        if override_consent:
            if st.button("Translate Content Anyway 🔓", type="secondary"):
                with st.spinner("Processing sensitive translation..."):
                    if direction == "E2B":
                        result = translate_text(input_text, e2b_model, e2b_tokenizer)
                        header_label = "Translated Text (Bangla)"
                    else:
                        result = translate_text(input_text, b2e_model, b2e_tokenizer)
                        header_label = "Translated Text (English)"
                        
                st.markdown(f"""
                    <div class="output-card">
                        <div class="output-header">✅ {header_label}:</div>
                        <div class="output-text">{result}</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        # Standard Clean Action Button
        if st.button("Translate text ✨", type="primary", use_container_width=True):
            with st.spinner("Translating tokens..."):
                if direction == "E2B":
                    result = translate_text(input_text, e2b_model, e2b_tokenizer)
                    header_label = "Translated Text (Bangla)"
                else:
                    result = translate_text(input_text, b2e_model, b2e_tokenizer)
                    header_label = "Translated Text (English)"
            
            # HTML Card Injection for a custom premium UI finish
            st.markdown(f"""
                <div class="output-card">
                    <div class="output-header">✅ {header_label}:</div>
                    <div class="output-text">{result}</div>
                </div>
            """, unsafe_allow_html=True)
else:
    # Small placeholder layout when input is empty
    st.button("Translate text ✨", type="primary", disabled=True, use_container_width=True)
