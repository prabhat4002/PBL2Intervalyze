import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the model and tokenizer from the saved local folder
save_path = './saved_similarity_model'
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModel.from_pretrained(save_path)
model.eval()

# Pre-saved reference text
SAVED_TEXT = "elephant is an animal"

def compute_similarity(user_text):
    """
    Compute cosine similarity between user text and pre-saved text.
    Args:
        user_text (str): Text to compare (e.g., from transcription).
    Returns:
        float: Cosine similarity score.
    """
    sentences = [SAVED_TEXT, user_text]
    tokens = {'input_ids': [], 'attention_mask': []}
    for sentence in sentences:
        new_tokens = tokenizer.encode_plus(
            sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt'
        )
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state
    attention = tokens['attention_mask']
    mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / counts
    mean_pooled_np = mean_pooled.detach().numpy()
    cosine_sim = cosine_similarity([mean_pooled_np[0]], [mean_pooled_np[1]])
    return cosine_sim[0][0]