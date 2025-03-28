"""
This script tests:
1. That your Together.ai API key works and returns expected results (LLM test)
2. That FAISS vectorstore + embeddings are working correctly (DB test)

Setup Instructions:
- Set your Together API key as the environment variable `API_KEY`
- Install all required packages using:
    pip install -r requirements.txt

Usage:
Just run the script:
    python test_homework.py
"""

from together import Together
from langchain_community.embeddings import TensorflowHubEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import json
import os
import shutil

API_KEY = os.getenv("API_KEY")

# Should work decently with this one
EMBEDDING_MODEL = "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual/versions/2"

def test_client():

    client = Together(api_key=API_KEY)
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the capital of Britain?"}
    ]

    response = client.chat.completions.create(
        messages=messages,
        temperature=0.5,
        top_k=100,
        top_p=0.9,
        repetition_penalty=1.05,
        model=model_name
    ).model_dump_json()

    response_text = json.loads(response)["choices"][0]['message']['content'] 
    # print('LLM response: ', response_text)
    assert "london" in response_text.lower()
    print('test_client passed!')

def test_db():
    
    db_path = 'test_catalogue.db'
    score_threshold = 0.001

    embedding_function = TensorflowHubEmbeddings(model_url=EMBEDDING_MODEL)
    dummy_catalog = pd.DataFrame([{'item': 1, 'description': 'test string'}, 
                                    {'item': 2, 'description': 'London is a big city'}, 
                                    {'item': 3, 'description': 'Paris is the capital of France'}, 
                                    {'item': 4, 'description': 'A bird is flying'},
                                    {'item': 5, 'description': 'faiss is a cool database'}])
    
    test_queries = ['testing is cool', 'Paris is huge', 'a tiger in the trees', 'how to set up a vector database?']

    if not os.path.exists(db_path):
        docs = dummy_catalog['description'].str.lower().tolist()
        
        faiss_vectorstore = FAISS.from_texts(
            texts=docs,
            embedding=embedding_function,
            metadatas=[{"source": 1}] * len(docs)
        )
        faiss_vectorstore.save_local(db_path)
        
    faiss_vectorstore = FAISS.load_local(db_path, embedding_function, allow_dangerous_deserialization=True)
    
    relevant_items_final = []
    
    for query in test_queries:
        query = query.lower()
        relevant_items = []
        new_items = faiss_vectorstore._similarity_search_with_relevance_scores(query, k=2)
        relevant_items.extend(new_items)
        relevant_items = [[c[0].page_content, c[1]] for c in relevant_items if c[1] > score_threshold]
        relevant_items = sorted(relevant_items, key=lambda x: x[1], reverse=True)
        relevant_items = [c[0] for c in relevant_items]
        relevant_items_final.append(relevant_items)

    # print('DB retrieval test: ', relevant_items_final)
    assert relevant_items_final == [['test string', 'faiss is a cool database'], 
                                    ['paris is the capital of france', 'london is a big city'], 
                                    ['a bird is flying'], 
                                    ['faiss is a cool database']], """Double-check the db setup. 
                                    If you used another embeddings model and your relevant_items_final seems to make some sense, ignore this assertion."""
    shutil.rmtree(db_path)

    print("DB test passed!")


test_client()
test_db()