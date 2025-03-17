from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer, pipeline
from typing import List
import torch
import threading
import spacy
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

phi2_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
phi2_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16)
phi2_tokenizer.pad_token = phi2_tokenizer.eos_token

bert_tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert")
bert_model = BertForSequenceClassification.from_pretrained("fine_tuned_bert")
bert_classifier = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)

with open("label_map.json", "r", encoding="utf-8") as file:
    label_map = json.load(file)

nlp = spacy.load("en_core_web_sm")

category_matching = pd.read_csv("category_matching_modified.csv")
sub_category_dict = category_matching.groupby("main_category")["sub_category"].unique().to_dict()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

def clean_output(text):
    if "**Product Description**:" in text:
        text = text.split("**Product Description**:")[1].strip()
    return text.strip()

def extract_key_features(description: str, max_features: int = 10) -> List[str]:
    """
    Args:
        description (str): Main text description.
        max_features (int): Số lượng đặc điểm tối đa (mặc định là 10).

    Returns:
        list: Danh sách các đặc điểm quan trọng.
    """
    doc = nlp(description)
    features = []

    for ent in doc.ents:
        if ent.label_ in {"PRODUCT", "ORG"}:
            features.append((ent.text, len(ent.text.split()), ent.start))

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        unwanted = {"which", "your", "and"}

        if not any(word in chunk_text.lower() for word in unwanted):
            if any(token.pos_ in ["NOUN", "PROPN"] for token in chunk):
                features.append((chunk_text, len(chunk_text.split()), chunk.start))

    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ == "NOUN":
            feature_phrase = f"{token.text} {token.head.text}"
            features.append((feature_phrase, len(feature_phrase.split()), token.i))

    features = sorted(features, key=lambda x: (-x[1], x[2]))

    feature_list = []
    seen = set()
    for feature, _, _ in features:
        if feature.lower() not in seen:
            feature_list.append(feature)
            seen.add(feature.lower())
        if len(feature_list) >= max_features:
            break

    return feature_list

def find_best_main_category(title):
    """Xác định danh mục chính bằng BERT."""
    prediction = bert_classifier(title)
    label_id = int(prediction[0]["label"].split("_")[-1])
    confidence = prediction[0]["score"]
    return label_map.get(label_id, "Other") if confidence >= 0.65 else "Other"

def find_best_subcategory(title, main_category):
    sub_categories = sub_category_dict.get(main_category, [])
    if len(sub_categories) == 0:
        return "Other"
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sub_categories)
    title_vector = vectorizer.transform([title])
    similarities = cosine_similarity(title_vector, tfidf_matrix)
    best_match_idx = similarities.argmax()
    best_match_score = similarities.max()
    return sub_categories[best_match_idx] if best_match_score > 0.6 else "Other"

def generate_response(user_input, main_category, sub_category, keywords):
    prompt = f""" 
    You are a professional description writer. Generate a high-quality product description.

    - **User input**: {user_input}
    - **Main Category**: {main_category}
    - **Sub Category**: {sub_category}
    - **Keywords**: {keywords}

    **Important**: The product description **must** include all the following keywords at least once: {keywords}.

    **Product Description**:
    [START]
    """.strip()

    input_ids = phi2_tokenizer(prompt, return_tensors="pt").input_ids.to(phi2_model.device)
    streamer = TextIteratorStreamer(phi2_tokenizer, skip_prompt=True, skip_special_tokens=True)

    thread = threading.Thread(
        target=lambda: phi2_model.generate(
            input_ids, 
            streamer=streamer, 
            max_new_tokens=250, 
            do_sample=True, 
            temperature=0.8, 
            top_k=40, 
            top_p=0.85,
            repetition_penalty=1.2,
            eos_token_id=phi2_tokenizer.eos_token_id
        )
    )
    thread.start()

    buffer = ""
    prev_length = 0

    for new_text in streamer:
        buffer += new_text
        cleaned_text = clean_output(buffer)
        if "[/END]" in cleaned_text:
            cleaned_text = cleaned_text.split("[/END]")[0]
            yield cleaned_text[prev_length:].strip() + " "
            break
        if "[END]" in cleaned_text:
            cleaned_text = cleaned_text.split("[END]")[0]
            yield cleaned_text[prev_length:].strip() + " "
            break

        new_part = cleaned_text[prev_length:].strip()
        prev_length = len(cleaned_text)

        if new_part:
            yield new_part + " "

@app.post("/answer")
async def answer(request: Request):
    data = await request.json()
    user_input = data.get("requirement", "")
    main_category = find_best_main_category(user_input)
    sub_category = find_best_subcategory(user_input, main_category)
    keywords = ", ".join(extract_key_features(user_input))
    print(keywords)
    return StreamingResponse(generate_response(user_input, main_category, sub_category, keywords), media_type="text/plain")