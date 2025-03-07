from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load the transcription processor and model
model_name = 'facebook/wav2vec2-base-960h'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Save to a local folder
save_path = './saved_transcription_model'
processor.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"Transcription model saved to {save_path}")