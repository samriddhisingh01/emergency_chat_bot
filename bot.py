from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.local_llm import LocalLLM

app = Flask(__name__)

# Load datasets
def load_datasets():
    with open('data/intents.json') as f:
        intents = json.load(f)['intents']
    with open('data/emergency_numbers_india.json') as f:
        emergency_numbers = json.load(f)
    hospitals_df = pd.read_csv('data/hospitals.csv')
    return intents, emergency_numbers, hospitals_df

# Prepare NLP components
def initialize_nlp(intents):
    patterns = []
    tags = []
    responses = {}
    for intent in intents:
        tags.extend([intent['tag']] * len(intent['patterns']))
        patterns.extend(intent['patterns'])
        responses[intent['tag']] = intent['responses'][0]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(patterns)
    return vectorizer, X, tags, responses

intents, emergency_numbers, hospitals_df = load_datasets()
vectorizer, X, tags, responses = initialize_nlp(intents)
llm = LocalLLM(model_id="gpt2", max_new_tokens=100)

def find_emergency_number(query):
    query = query.lower()
    for service in emergency_numbers:
        if any(kw in query for kw in service['service'].lower().split()):
            return f"{service['service']}: {service['number']}"
    # Common fallbacks
    if "fire" in query:
        return "Fire: 101"
    if "police" in query:
        return "Police: 100"
    if "ambulance" in query:
        return "Ambulance: 102"
    if "emergency" in query:
        return "National Emergency Number: 112"
    return None

def find_hospitals(location=None):
    if location:
        mask = (hospitals_df['State'].str.lower() == location.lower()) | \
               (hospitals_df['District'].str.lower() == location.lower())
        results = hospitals_df[mask]
    else:
        results = hospitals_df
    return results.head(5)

def detect_intent(query):
    user_vec = vectorizer.transform([query])
    sims = cosine_similarity(user_vec, X)
    idx = sims.argmax()
    return tags[idx] if sims[0, idx] > 0.25 else None

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '').strip()
    if not query:
        return jsonify({'response': 'Please enter a valid query'})

    # 1. Emergency numbers
    emergency_response = find_emergency_number(query)
    if emergency_response:
        return jsonify({'response': emergency_response})

    # 2. Hospital lookup
    if 'hospital' in query.lower():
        location = None
        if ' in ' in query.lower():
            location = query.lower().split(' in ')[-1].strip()
        hospitals = find_hospitals(location)
        hospital_list = "\n".join(
            [f"{row['Hospital_Name']} ({row['District']}) - 📞{row['Emergency_Num']}" 
             for _, row in hospitals.iterrows()]
        ) if not hospitals.empty else "No hospitals found"
        return jsonify({'response': hospital_list})

    # 3. Intent detection
    intent = detect_intent(query)
    if intent and intent in responses:
        return jsonify({'response': responses[intent]})

    # 4. LLM fallback
    llm_response = llm.generate(f"Emergency context: {query}")
    return jsonify({'response': llm_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
