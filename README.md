# 🎙️ Intervalyze: Mock Interview Evaluator

## 🧠 Overview

**Intervalyze** is a Flask-based web application designed to help users improve their interview skills by analyzing audio responses. It evaluates both **emotional tone** and **content accuracy**, providing **real-time feedback** using fine-tuned AI models.

- Achieves **96% accuracy** in emotion detection using **Wav2Vec2** on the TESS dataset  
- Delivers **semantic similarity scores** using **Sentence Transformers**  
- Offers feedback like: _“You sound calm and composed, and your answer is excellent.”_

---

## ✨ Features

- 🔐 **User Authentication**: Secure login and signup using `Flask-Login` with password hashing  
- 🔊 **Audio Analysis**: Accepts live recordings or uploaded audio files for evaluation  
- 😌 **Emotion Detection**: Fine-tuned Wav2Vec2 model detects emotions like happiness, calmness, anxiety, and apathy  
- 🧠 **Content Evaluation**: Uses `bert-base-nli-mean-tokens` to compare responses with expected answers via cosine similarity  
- 📝 **Feedback Delivery**: Real-time feedback combining emotion and semantic accuracy  
- 📜 **Response History**: View past responses in a table with emotion, similarity score, and timestamps  
- 🎨 **Frontend Interface**: Animated login, question selection, and history pages styled with custom HTML/CSS  

---

## 🛠️ Installation

### ✅ Prerequisites

- Python 3.8+
- SQLite (bundled with Python)
- Microphone or audio recording capability (for live recording)

### ⚙️ Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/intervalyze.git
   cd intervalyze

   
2. **Create and activate a virtual environment:**

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies::**

    ```
    pip install -r requirements.txt


4. Set up the database:

The SQLite database intervalyze.db is automatically created when the app is first run.

Download required models:

facebook/wav2vec2-large-960h-lv60-self – for transcription

Fine-tuned Wav2Vec2 model – for emotion classification (downloaded automatically)

bert-base-nli-mean-tokens – for semantic similarity (downloaded automatically)
