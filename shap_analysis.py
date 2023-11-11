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

CONTENT_PATH = "./data/vctk/hubert-base-1w/ninth_layer_hidden/"
SPEAKER_PATH = "./data/vctk/speaker_embed_no_split/"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = SimpleClassifier(1024, 20)
model.load_state_dict(torch.load("./models/classifier_0.9803.pt"))
model.eval()

dataset = EmbeddingDataset(CONTENT_PATH, SPEAKER_PATH)
label_dict = dataset.speaker_to_label

split = 0.8
split_idx = int(len(dataset) * split)
train_dataset = Subset(dataset, torch.arange(split_idx))
test_dataset = Subset(dataset, torch.arange(split_idx, len(dataset)))
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

model = model.to(device)

def predict(x, model):
    x = x.to(device)
    model = model.to(device)
    pred = model(x)

    return pred.detach().cpu().numpy()

# x: (batch_size, 1024) y: (batch_size, 20)
x, y = next(iter(test_dataloader))
x = x.to(device)
y = y.to(device)
pred = predict(x[0], model)

to_explain = x[[0, 1]]
explainer = shap.GradientExplainer(model, x, batch_size=32, local_smoothing=0)

shap_values = explainer.shap_values(x, ranked_outputs=None)
print(len(shap_values), shap_values[0].shape)
shap_values = np.array(shap_values)
np.save('./explanations/shap_gradient_0_1.npy', shap_values)