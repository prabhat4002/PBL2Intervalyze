from flask import Flask, request, render_template
import os
from emotion import predict_emotion      # Import emotion prediction
from transcribe import transcribe_audio  # Import transcription
from similarity import compute_similarity  # Import text similarity

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return render_template('index.html', error="No file selected")
        
        # Save the uploaded file temporarily
        audio_path = "temp_audio.wav"
        audio_file.save(audio_path)

        # Get emotion, transcription, and similarity
        predicted_emotion, probabilities = predict_emotion(audio_path)
        transcription = transcribe_audio(audio_path)
        similarity_score = compute_similarity(transcription)
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # Pass all data to the template
        return render_template('index.html', emotion=predicted_emotion, probabilities=probabilities, 
                             emotion_labels=['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy'], 
                             transcription=transcription, similarity_score=similarity_score)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)