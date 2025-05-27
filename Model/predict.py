from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import torch
import os
import pathlib
import requests
import zipfile
import gdown

MODEL_DIR = pathlib.Path(__file__).parent / "roberta_emotion"
MODEL_ZIP_PATH = MODEL_DIR.with_suffix(".zip")

GOOGLE_DRIVE_ID = "15w3ollE-efc0r2eGcBWmjjHTpHFUzpaz"

def download_model_if_needed():
    if not MODEL_DIR.exists():
        print("🔽 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id=15w3ollE-efc0r2eGcBWmjjHTpHFUzpaz"
        gdown.download(url, str(MODEL_ZIP_PATH), quiet=False)
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR.parent)
        MODEL_ZIP_PATH.unlink()

download_model_if_needed()

tokenizer = RobertaTokenizerFast.from_pretrained(str(MODEL_DIR))
model = RobertaForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    label_id = torch.argmax(outputs.logits, dim=1).item()
    label_map = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return label_map[label_id]

def get_kaggle_id():
    return "yagomoreno"
