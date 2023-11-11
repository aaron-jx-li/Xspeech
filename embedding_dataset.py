import torch
import soundfile as sf
import soxr
from speaker_embed import *
from preprocess_audio import *
from speaker_model import *
import os
from classifier import *
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from tqdm import tqdm

class EmbeddingDataset(Dataset):
    def __init__(self, content_path, speaker_path):
        self.content_path = content_path
        self.speaker_path = speaker_path
        content_paths = list(os.listdir(self.content_path))
        
        self.speakers = []
        self.speaker_to_label = {}
        self.num_speakers = 0
        for s in os.listdir(self.speaker_path):
            if s.startswith("."):
                continue
            if s not in self.speaker_to_label.keys():
                self.speaker_to_label[s] = self.num_speakers
                self.num_speakers += 1
        self.labels = []
        self.contents = []
        for filename in content_paths:
            if not filename.endswith(".hid"):
                continue
            local_path = filename
            speaker_id = local_path.split("_")[0]
            snippet_id = local_path.split(".")[0]
            try:
                content = torch.load(f'{self.content_path}{local_path}')
                speaker = torch.tensor(torch.load(f'{self.speaker_path}{speaker_id}/{snippet_id}.pt'))
                self.contents.append(content)
                self.speakers.append(speaker)
            except:
                continue
            self.labels.append(self.speaker_to_label[speaker_id])
        assert len(self.contents) == len(self.speakers)


    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        content = self.contents[idx]
        dim = 1 if len(content.shape) == 3 else 0
        content = torch.mean(content, dim=dim)
        speaker = self.speakers[idx]
        embed = torch.cat((content, speaker), dim=-1)
        label = self.labels[idx]
        y = torch.zeros(self.num_speakers)
        y[label] = 1
        return embed, y