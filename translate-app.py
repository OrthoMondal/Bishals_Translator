import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch
import os

# 1. Page Configuration & Adaptive Theme Styling
st.set_page_config(
    page_title="Bishals Translator", 
    page_icon="🌐", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Inject Modern UI styling using native Streamlit CSS variables
st.markdown("""
    <style>
    /* Modernize Headers using the theme's default text color */
    h1 {
        font-weight: 700 !important;
        color: var(--text-color) !important;
        letter-spacing: -0.5px;
    }
    
    /* Clean custom card for translation output that adapts to Light/Dark mode */
    .translation-card {
        background-color: var(--background-color);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid var(--secondary-background-color);
        margin-top: 15px;
    }
    
    .card-title {
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 600;
        margin-bottom: 8px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .card-body {
        color: var(--text-color);
        font-size: 1.15rem;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize short-term memory state variables if they don't exist yet
if 'show_warning' not in st.session_state:
    st.session_state.show_warning = False
if 'bypass_offensive' not in st.session_state:
    st.session_state.bypass_offensive = False

# Set the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths pointing to your GitHub repository folders
e2b_model_path = './E2B/checkpoint-4000'
e2b_tokenizer_path = './E2B/checkpoint-4000'
b2e_model_path = './B2E/checkpoint-4000'
b2e_tokenizer_path = './B2E/checkpoint-4000'

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

with st.spinner("⚡ Initializing neural translation engines..."):
    e2b_tokenizer, e2b_model = load_e2b_model()
    b2e_tokenizer, b2e_model = load_b2e_model()

def perform_translation(text, direction_choice):
    if direction_choice == 'English to Bangla':
        inputs = e2b_tokenizer(text, return_tensors="pt").to(device)
        output_tokens = e2b_model.generate(**inputs, max_length=128)
        return e2b_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    else:
        inputs = b2e_tokenizer(text, return_tensors="pt").to(device)
        output_tokens = b2e_model.generate(**inputs, max_length=128)
        return b2e_tokenizer.decode(output_tokens[0], skip_special_tokens=True)

@st.cache_data
def load_offensive_words(file_path):
    offensive_words_list = []
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                offensive_words_list.append(line.strip().lower())
    except Exception:
        return []
    return offensive_words_list

def contains_offensive_language(text, offensive_words):
    if not offensive_words: 
        return False
    text = text.lower()
    for word in offensive_words:
        if re.search(r'\b' + re.escape(word) + r'\b', text):
            return True
    return False

offensive_words_file_path = './offensive_words.txt'
offensive_words_list = load_offensive_words(offensive_words_file_path)

# App Interface
st.title("🌐 Bishals Translator")
st.caption("AI-powered seamless translation between English and Bangla")
st.markdown("---")

col1, col2 = st.columns([2, 1], gap="large")

with col2:
    st.subheader("Configuration")
    direction = st.radio(
        "Translation Direction:",
        ('English to Bangla', 'Bangla to English'),
        index=0
    )

with col1:
    st.subheader("Text Processing")
    text_to_translate = st.text_area(
        "Enter text to translate:", 
        placeholder="Type or paste your content here...", 
        height=140
    )
    
    trigger_translation = False

    if st.button("Translate Now ✨", type="primary", use_container_width=True):
        if text_to_translate.strip():
            st.session_state.show_warning = False
            st.session_state.bypass_offensive = False
            
            if contains_offensive_language(text_to_translate, offensive_words_list):
                st.session_state.show_warning = True
            else:
                trigger_translation = True
        else:
            st.info("Please enter some text to translate.")

    if st.session_state.show_warning and not st.session_state.bypass_offensive:
        st.warning("⚠️ Warning: The input text may contain potentially offensive or sensitive language.")
        
        if st.button("Translate Anyway 🔓", type="secondary", use_container_width=True):
            st.session_state.bypass_offensive = True
            trigger_translation = True

    if trigger_translation or st.session_state.bypass_offensive:
        with st.spinner("Processing translation pipeline..."):
            translated_result = perform_translation(text_to_translate, direction)
            target_lang = "Bangla" if direction == 'English to Bangla' else "English"
            
            st.markdown(f"""
                <div class="translation-card">
                    <div class="card-title">Translated Output ({target_lang})</div>
                    <div class="card-body">{translated_result}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.session_state.show_warning = False
            st.session_state.bypass_offensive = False
