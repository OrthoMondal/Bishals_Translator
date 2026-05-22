import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch
import os

# 1. Page Configuration & Modern Setup
st.set_page_config(
    page_title="Bishals Translator", 
    page_icon="🌐", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize a session state variable to track if the user bypassed the warning
if "bypass_warning" not in st.session_state:
    st.session_state.bypass_warning = False

# Set the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths pointing to your GitHub folders
e2b_model_path = './E2B/checkpoint-4000'
e2b_tokenizer_path = './E2B/checkpoint-4000'
b2e_model_path = './B2E/checkpoint-4000'
b2e_tokenizer_path = './B2E/checkpoint-4000'

# 2. Cached Loading Functions
@st.cache_resource
def load_e2b_model():
    tokenizer = AutoTokenizer.from_pretrained(e2b_tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(e2b_model_path)
    model.to(device)
    return tokenizer, model

@st.cache_resource
def load_b2e_model():
    tokenizer = AutoTokenizer.from_pretrained(b2e_path) if 'b2e_path' in locals() else AutoTokenizer.from_pretrained(b2e_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(b2e_model_path)
    model.to(device)
    return tokenizer, model

# Display clean minimalist header loader
with st.spinner("⚡ Initializing neural networks... Please hold."):
    e2b_tokenizer, e2b_model = load_e2b_model()
    b2e_tokenizer, b2e_model = load_b2e_model()

# 3. Core Translation Logic
def translate_english_to_bangla(text):
    inputs = e2b_tokenizer(text, return_tensors="pt").to(device)
    output_tokens = e2b_model.generate(**inputs, max_length=128)
    return e2b_tokenizer.decode(output_tokens[0], skip_special_tokens=True)

def translate_bangla_to_english(text):
    inputs = b2e_tokenizer(text, return_tensors="pt").to(device)
    output_tokens = b2e_model.generate(**inputs, max_length=128)
    return b2e_tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# 4. Moderation Filter Logic
@st.cache_data
def load_offensive_words(file_path):
    offensive_words_list = []
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
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


# ==========================================
# 5. Modern UI Render Layout
# ==========================================

st.title("🌐 Bishals Translator")
st.caption("A clean, production-ready framework powered by your fine-tuned NLLB translation weights.")

# Clean spacing
st.write("---")

# Segmented controls or side-by-side layout for selectors
col1, col2 = st.columns([2, 1])

with col1:
    direction = st.segmented_control(
        "Translation Track",
        options=['English to Bangla', 'Bangla to English'],
        default='English to Bangla'
    ) or 'English to Bangla' # Fallback default

with col2:
    st.write("<br>", unsafe_allow_html=True) # Align with segmented controls
    # Clean indicator for runtime acceleration hardware
    if torch.cuda.is_available():
        st.markdown("🟢 `GPU Accelerated` ")
    else:
        st.markdown("⚪ `CPU Inference` ")

# Main text interface block
text_to_translate = st.text_area(
    "Source Text", 
    placeholder="Type or paste your content here...",
    height=160
)

# Modern Action Control Area
col_btn1, col_btn2 = st.columns([1, 4])
trigger_translation = False

with col_btn1:
    if st.button("Translate", type="primary", use_container_width=True):
        trigger_translation = True

# Process translation pipeline
if trigger_translation or st.session_state.bypass_warning:
    if text_to_translate.strip():
        # Step A: Perform Moderation Check
        is_offensive = contains_offensive_language(text_to_translate, offensive_words_list)
        
        # Step B: If offensive AND user has not clicked bypass yet, block and show verification toggle
        if is_offensive and not st.session_state.bypass_warning:
            st.error("⚠️ System Flag: The input text may violate standard conversational guidelines.")
            
            with st.container(border=True):
                st.markdown("💬 **Override Notice:** Do you still wish to complete this translation request for research or documentation purposes?")
                if st.button("Proceed Anyway", type="secondary"):
                    st.session_state.bypass_warning = True
                    st.rerun() # Refresh the page to apply bypass state changes
                    
        else:
            # Step C: Execution block (Runs normally, or if user explicitly chose to proceed)
            with st.spinner("Processing translation sequence..."):
                if direction == 'English to Bangla':
                    translated_text = translate_english_to_bangla(text_to_translate)
                    headline = "Target Output (Bangla)"
                else:
                    translated_text = translate_bangla_to_english(text_to_translate)
                    headline = "Target Output (English)"
            
            # Display target translation inside a nice custom border block
            st.success(f"✨ {headline}")
            st.text_area("", value=translated_text, height=160, disabled=True, label_visibility="collapsed")
            
            # Reset the override flag status silently for subsequent fresh queries
            st.session_state.bypass_warning = False
    else:
        st.toast("Please enter text before clicking translate.", icon="🚨")
