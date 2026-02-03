from sentence_transformers import SentenceTransformer
import torch

CLIP_MODEL_PATH = "openai/clip-vit-large-patch14"
device = "cuda" if torch.cuda.is_available() else "cpu"


clip_encoder = SentenceTransformer(CLIP_MODEL_PATH, device=device)

print(clip_encoder)