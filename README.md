# 🚑 Emergency Assistance Chatbot

## 📌 Overview
The **Emergency Assistance Chatbot** is a **Flask-based AI chatbot** that can:
- Provide **emergency helpline numbers** in India
- Suggest **nearest hospitals** with contact info
- Give **first-aid guidance** for various medical situations
- Answer **general emergency-related queries** using a local LLM
- Fallback to an AI model when no predefined intent matches

It uses:
- **Intent classification** with `TF-IDF` and `cosine similarity`
- **Structured knowledge base** (`intents.json` for first-aid, `emergency_numbers_india.json` for helplines, `hospitals.csv` for hospital data)
- **Local LLM integration** for open-ended queries

---

## 🚀 Features
- **🩺 First Aid Help** — Based on `intents.json` patterns and responses
- **📞 Helpline Numbers** — Direct retrieval using keyword-matched queries
- **🏥 Hospital Locator** — Filter hospitals by *state* or *district*, with emergency contact number
- **🤖 Local AI Fallback** — Uses a local LLM (`gpt2` in example) to answer related questions
- **💬 Web-based Chat Interface** — Flask app serving an HTML front-end
- **🔍 NLP-driven Intent Detection** — `TF-IDF` + `cosine similarity` pattern matching

---


