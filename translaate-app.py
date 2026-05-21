
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch # Import torch

# Import userdata for Google Colab, handle potential ImportError if not in Colab
try:
    from google.colab import userdata
except ImportError:
    userdata = None
    print("Warning: google.colab.userdata not available. Ensure offensive_words.txt is accessible if not using Colab.")

# Set the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define paths for E2B model and tokenizer
e2b_model_path = '/content/drive/MyDrive/E2B_Bishal_Final/NLLB_eng_ben_Run/E2B_finetuned_idiom_proverb_model/checkpoint-3680'
e2b_tokenizer_path = '/content/drive/MyDrive/E2B_Bishal_Final/NLLB_eng_ben_Run/E2B_finetuned_idiom_proverb_model/checkpoint-3680'

# Define paths for B2E model and tokenizer
b2e_model_path = '/content/drive/MyDrive/B2E_Bishal_Final/NLLB_ben_eng_Run/checkpoint-4000'
b2e_tokenizer_path = '/content/drive/MyDrive/B2E_Bishal_Final/NLLB_ben_eng_Run/checkpoint-4000'

# Load E2B tokenizer and model
@st.cache_resource
def load_e2b_model():
    tokenizer = AutoTokenizer.from_pretrained(e2b_tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(e2b_model_path)
    model.to(device) # Move model to the selected device
    return tokenizer, model

# Load B2E tokenizer and model
@st.cache_resource
def load_b2e_model():
    tokenizer = AutoTokenizer.from_pretrained(b2e_tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(b2e_model_path)
    model.to(device) # Move model to the selected device
    return tokenizer, model

e2b_tokenizer, e2b_model = load_e2b_model()
b2e_tokenizer, b2e_model = load_b2e_model()

# Translation functions
def translate_english_to_bangla(text):
  inputs = e2b_tokenizer(text, return_tensors="pt").to(device) # Move inputs to the selected device
  output_tokens = e2b_model.generate(**inputs, max_length=128) # Adjust max_length as needed
  translated_text = e2b_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
  return translated_text

def translate_bangla_to_english(text):
  inputs = b2e_tokenizer(text, return_tensors="pt").to(device) # Move inputs to the selected device
  output_tokens = b2e_model.generate(**inputs, max_length=128) # Adjust max_length as needed
  translated_text = b2e_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
  return translated_text

# Offensive language detection
@st.cache_data
def load_offensive_words(file_path):
    offensive_words_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                offensive_words_list.append(line.strip().lower())
    except FileNotFoundError:
        st.error(f"Error: Offensive words file not found at {{file_path}}. Offensive language checking will be skipped.")
        return []
    return offensive_words_list

def contains_offensive_language(text, offensive_words):
    if not offensive_words: # Skip check if list is empty due to file not found
        return False
    text = text.lower()
    for word in offensive_words:
        # Use regex to match whole words only
        if re.search(r'\b' + re.escape(word) + r'\b', text):
            return True
    return False

# Define the path to the offensive words file in Google Drive
offensive_words_file_path = '/content/drive/MyDrive/Colab Notebooks/offensive_words.txt'
offensive_words_list = load_offensive_words(offensive_words_file_path)


# Streamlit app interface
st.title("Bishals Translator")

direction = st.radio(
    "Select Translation Direction:",
    ('English to Bangla', 'Bangla to English'))

text_to_translate = st.text_area("Enter text to translate:")

if st.button("Translate"):
    if text_to_translate:
        # Check for offensive language
        if contains_offensive_language(text_to_translate, offensive_words_list):
            st.warning("Warning: The input text may contain offensive language and will not be translated.")
        else:
            translated_text = ""
            if direction == 'English to Bangla':
                translated_text = translate_english_to_bangla(text_to_translate)
                st.success("Translated Text (Bangla):")
                st.write(translated_text)
            elif direction == 'Bangla to English':
                translated_text = translate_bangla_to_english(text_to_translate)
                st.success("Translated Text (English):")
                st.write(translated_text)
    else:
        st.info("Please enter text to translate.")
