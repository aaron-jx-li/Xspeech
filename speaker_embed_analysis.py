import torch
import soundfile as sf
import soxr
from speaker_embed import *
from preprocess_audio import *
from speaker_model import *
import os

SPEAKER_EMBED_PATH = "./data/vctk/speaker_embed_no_split/"
embed0 = torch.load(os.path.join(SPEAKER_EMBED_PATH, "p225/p225_001.pt"))
#embed1 = torch.load(os.path.join(SPEAKER_EMBED_PATH, "p227/p227_002.pt"))
similarities = []
for dir_name in os.listdir(SPEAKER_EMBED_PATH):
    if dir_name.startswith('.'):
        continue
    speaker_dir = os.path.join(SPEAKER_EMBED_PATH, dir_name)
    speaker_sim = 0
    for filename in os.listdir(speaker_dir):
        if filename == "p225_001.pt":
            continue
        embed1 = torch.load(os.path.join(speaker_dir, filename))
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        speaker_sim += cos(torch.from_numpy(embed0), torch.from_numpy(embed1))
    similarities.append((speaker_sim / len(os.listdir(speaker_dir)), dir_name))



print(similarities)
print(sum([sim[0] for sim in similarities]) / len(similarities))