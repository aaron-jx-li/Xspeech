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

'''
num_errors = 0
for i, (x, y) in tqdm(enumerate(test_dataloader)):
    x = x.to(device)
    y = y.to(device)
    pred = model(x)
    print(pred)
    pred = torch.argmax(pred, dim=1)
    y = torch.argmax(y, dim=1)
    for k in range(len(pred)):
        if pred[k] != y[k]:
            num_errors += 1
acc = 1 - num_errors / len(test_dataset)
acc = float("%.4f" % acc)
print(f'Number of errors: {num_errors}')
print(f'Test Accuracy: {acc}')
'''

def predict(x, model):
    x = x.to(device)
    model = model.to(device)
    pred = model(x)

    return pred.detach().cpu().numpy()

# x: (batch_size, 1024) y: (batch_size, 20)
x, y = next(iter(test_dataloader))
pred = predict(x[0], model)

explainer = LimeTextExplainer(class_names=dataset.speaker_to_label.keys())
explanation = explainer.explain_instance(x[0], 
                                         predict, # classification function
                                         num_features=6, 
                                         labels=[0, 1]
                                         )    

print(explanation)