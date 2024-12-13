import streamlit as st
from translate import Translator
from datasets import load_dataset
import sacrebleu
from datasets import load_dataset
dataset = load_dataset("ted_multi", "en-fr", split="train[:1%]")

# Title
st.title("Language Translation Tool")

# Sidebar for language selection
st.sidebar.header("Settings")
source_lang = st.sidebar.selectbox("Source Language", ["en", "fr", "de", "es"])
target_lang = st.sidebar.selectbox("Target Language", ["fr", "en", "de", "es"])

# Input text
input_text = st.text_area("Enter text to translate:")

# Translate button
if st.button("Translate"):
    translator = Translator(source_lang=source_lang, target_lang=target_lang)
    translation = translator.translate(input_text)
    st.write("### Translation:")
    st.success(translation)

# Dataset for testing
st.sidebar.subheader("Dataset Options")
if st.sidebar.button("Load Dataset"):
    dataset = load_dataset("ted_multi", "en-fr", split="test[:5%]")
    st.write("### Sample Dataset:")
    st.write(dataset.select(range(5)).to_pandas())

# Evaluation
st.sidebar.subheader("Evaluate Model")
if st.sidebar.button("Evaluate"):
    references = [entry["translation"][target_lang] for entry in dataset]
    predictions = [translator.translate(entry["translation"][source_lang]) for entry in dataset]
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    st.write("### BLEU Score:")
    st.success(f"{bleu.score:.2f}")
