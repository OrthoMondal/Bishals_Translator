import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch
import os

# Set up page config
st.set_page_config(page_title="Bishals Translator", page_icon="🌐", layout="centered")

# Set the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define relative paths pointing directly to your GitHub repository folders
e2b_model_path = './E2B/checkpoint-4000'
e2b_tokenizer_path = './E2B/checkpoint-4000'

b2e_model_path = './B2E/checkpoint-4000'
b2e_tokenizer_path = './B2E/checkpoint-4000'

# Load E2B tokenizer and model (Cached so they only load ONCE)
@st.cache_resource
def load_e2b_model():
    tokenizer = AutoTokenizer.from_pretrained(e2b_tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(e2b_model_path)
    model.to(device)
    return tokenizer, model

# Load B2E tokenizer and model
@st.cache_resource
def load_b2e_model():
    tokenizer = AutoTokenizer.from_pretrained(b2e_tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(b2e_model_path)
    model.to(device)
    return tokenizer, model

# Initialize models with a friendly loading spinner
with st.spinner("Loading models into memory... This might take a moment on first launch."):
    e2b_tokenizer, e2b_model = load_e2b_model()
    b2e_tokenizer, b2e_model = load_b2e_model()

# Translation functions
def translate_english_to_bangla(text):
    inputs = e2b_tokenizer(text, return_tensors="pt").to(device)
    output_tokens = e2b_model.generate(**inputs, max_length=128)
    translated_text = e2b_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return translated_text

def translate_bangla_to_english(text):
    inputs = b2e_tokenizer(text, return_tensors="pt").to(device)
    output_tokens = b2e_model.generate(**inputs, max_length=128)
    translated_text = b2e_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return translated_text

# Offensive language detection
@st.cache_data
def load_offensive_words(file_path):
    offensive_words_list = []
    if not os.path.exists(file_path):
        # Gracefully handle if the file isn't uploaded yet without crashing the whole app
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

# Looks for offensive_words.txt in the main directory of your GitHub repo
offensive_words_file_path = './offensive_words.txt'
offensive_words_list = load_offensive_words(offensive_words_file_path)

# Streamlit app interface
st.title("🌐 Bishals Translator")
st.write("Custom English ↔ Bangla Translation Platform")

direction = st.radio(
    "Select Translation Direction:",
    ('English to Bangla', 'Bangla to English'))

text_to_translate = st.text_area("Enter text to translate:", height=150)

if st.button("Translate", type="primary"):
    if text_to_translate.strip():
        # Check for offensive language
        if contains_offensive_language(text_to_translate, offensive_words_list):
            st.warning("Warning: The input text may contain offensive language and will not be translated.")
        else:
            with st.spinner("Translating..."):
                if direction == 'English to Bangla':
                    translated_text = translate_english_to_bangla(text_to_translate)
                    st.success("Translated Text (Bangla):")
                    st.text_area("Output:", value=translated_text, height=150, disabled=True)
                elif direction == 'Bangla to English':
                    translated_text = translate_bangla_to_english(text_to_translate)
                    st.success("Translated Text (English):")
                    st.text_area("Output:", value=translated_text, height=150, disabled=True)
    else:
        st.info("Please enter text to translate.")
