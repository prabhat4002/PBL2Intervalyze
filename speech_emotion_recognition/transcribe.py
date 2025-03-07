import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa

# Load the processor and model from the saved local folder
save_path = './saved_transcription_model'
transcription_processor = Wav2Vec2Processor.from_pretrained(save_path)
transcription_model = Wav2Vec2ForCTC.from_pretrained(save_path)
transcription_model.eval()

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using Wav2Vec2ForCTC.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        str: Transcribed text.
    """
    audio, sample_rate = librosa.load(audio_path, sr=16000)
    inputs = transcription_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transcription_model.to(device)
    input_values = input_values.to(device)
    with torch.no_grad():
        logits = transcription_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = transcription_processor.batch_decode(predicted_ids)[0]
    return transcription