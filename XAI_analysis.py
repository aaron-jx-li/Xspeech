import torch
import numpy as np
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
from lime.lime_text import LimeTextExplainer
from embedding_dataset import EmbeddingDataset
import shap

shap_values = np.load('./explanations/shap_gradient_0_1.npy')
content_score = np.mean(shap_values[:, :, :768])
speaker_score = np.mean(shap_values[:, :, 768:])
print(content_score, speaker_score)