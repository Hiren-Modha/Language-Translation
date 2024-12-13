from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
dataset = load_dataset("ted_multi", "en-fr", split="train[:1%]")


class Translator:
    def __init__(self, source_lang="en", target_lang="fr"):
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

