import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import librosa

# Load the processor and emotion model once
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
model = Wav2Vec2ForSequenceClassification.from_pretrained('./emotion_model')
model.eval()

# Emotion labels from your Kaggle dataset
emotion_labels = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy']

def predict_emotion(audio_path):
    """
    Predict emotion from an audio file.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        tuple: (predicted emotion label, probability array)
    """
    audio, sample_rate = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_values = input_values.to(device)
    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        predicted_label = emotion_labels[predicted_class]
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return predicted_label, probs