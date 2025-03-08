from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer, BartForConditionalGeneration, BartTokenizer, BertForSequenceClassification, BertTokenizer, pipeline
from typing import List
import spacy
import torch
import asyncio
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import threading

app = FastAPI()
bart_tokenizer = BartTokenizer.from_pretrained("fine_tuned_bart_v1")
bart_model = BartForConditionalGeneration.from_pretrained("fine_tuned_bart_v1")
bert_tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert")
bert_model = BertForSequenceClassification.from_pretrained("fine_tuned_bert")
bert_classifier = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

with open("label_map.json", "r", encoding="utf-8") as file:
    label_map = json.load(file)

nlp = spacy.load("en_core_web_sm")

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
    
    for chunk in doc.noun_chunks:
        unwanted = {"which", "your", "and"} 
        chunk_text = chunk.text.strip()
        if not any(word in chunk_text.lower() for word in unwanted):
            if any(token.pos_ in ["NOUN", "PROPN"] for token in chunk):
                if chunk_text in description:
                    features.append((chunk_text, len(chunk_text.split()), chunk.start))
    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ == "NOUN":
            feature_phrase = f"{token.text} {token.head.text}"
            if feature_phrase in description: 
                features.append((feature_phrase, len(feature_phrase.split()), token.i))
    
    # Sắp xếp theo độ quan trọng:
    # 1. Độ dài cụm (dài hơn = quan trọng hơn)
    # 2. Vị trí trong văn bản (sớm hơn = quan trọng hơn)
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

category_matching = pd.read_csv("category_matching_modified.csv")
sub_category_dict = category_matching.groupby("main_category")["sub_category"].unique().to_dict()

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

def find_best_main_category(title):
    prediction = bert_classifier(title)
    label_id = int(prediction[0]["label"].split("_")[-1])
    confidence = prediction[0]["score"]
    if confidence >= 0.8:
        return label_map.get(label_id, "Other")
    else:
        return "Other"

def generate_prompt(user_input: str) -> str:
    main_category = find_best_main_category(user_input)
    sub_category = find_best_subcategory(user_input, main_category)
    features = ", ".join(process_description(user_input)["features"])
    prompt = (
        f"User input: {user_input}\n"
        f"Main category: {main_category}\n"
        f"Sub category: {sub_category}\n"
        f"Features: {features}"
    )
    return prompt

def process_description(description: str) -> dict:
    """
    Xử lý mô tả sản phẩm để trích xuất các đặc điểm quan trọng.
    
    Args:
        description (str): Mô tả sản phẩm.
    
    Returns:
        dict: danh sách đặc điểm.
    """
    features = extract_key_features(description)
    return {"features": features}

async def generate_bart_response(prompt: str):
    input_ids = bart_tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")

    streamer = TextIteratorStreamer(bart_tokenizer, skip_special_tokens=True)
    
    thread = threading.Thread(target=bart_model.generate, kwargs={"input_ids": input_ids, "streamer": streamer, "num_beams": 1, "max_length": 1024})
    thread.start()

    for chunk in streamer:
        yield chunk
        await asyncio.sleep(0) 

@app.post("/answer")
async def answer(request: Request):
    data = await request.json()
    user_input = data.get("requirement", "")
    prompt = generate_prompt(user_input)
    return StreamingResponse(generate_bart_response(prompt=prompt), media_type="text/plain")