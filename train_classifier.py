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
from embedding_dataset import EmbeddingDataset

CONTENT_PATH = "./data/vctk/hubert-base-1w/ninth_layer_hidden/"
SPEAKER_PATH = "./data/vctk/speaker_embed_no_split/"
device = "cuda:0" if torch.cuda.is_available() else "cpu"


dataset = EmbeddingDataset(CONTENT_PATH, SPEAKER_PATH)
print(f'Dataset size: {len(dataset)}')
split = 0.8
split_idx = int(len(dataset) * split)
train_dataset = Subset(dataset, torch.arange(split_idx))
test_dataset = Subset(dataset, torch.arange(split_idx, len(dataset)))
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
criterion = torch.nn.CrossEntropyLoss().to(device)
model = SimpleClassifier(1024, 20).to(device)
optimizer = torch.optim.Adam(model.parameters())
num_epochs = 10
accs = []
for j in range(num_epochs):
    total_training_loss = 0
    model.train()
    for i, (x, y) in tqdm(enumerate(train_dataloader)):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_training_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {j} Training loss: {total_training_loss / i}')

    model.eval()
    num_errors = 0
    for i, (x, y) in tqdm(enumerate(test_dataloader)):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred = torch.argmax(pred, dim=1)
        y = torch.argmax(y, dim=1)
        for k in range(len(pred)):
            if pred[k] != y[k]:
                num_errors += 1
    acc = 1 - num_errors / len(test_dataset)
    acc = float("%.4f" % acc)
    accs.append(acc)
    print(f'Number of errors: {num_errors}')
    print(f'Epoch {j} Test Accuracy: {acc}')


torch.save(model.state_dict(), f'./models/classifier_{acc}.pt')


# shape (num_frames, 768)
'''
content = torch.load(EMBED_PATH + "p225_003.hid")
content = torch.mean(content, dim=0)
# shape (256,)
speaker = torch.tensor(torch.load(os.path.join(SPEAKER_EMBED_PATH, "p225/p225_001.pt")))
# shape (1024,)
embed = torch.cat((content, speaker))

model = SimpleClassifier(embed.shape[0], 20)
result = model(embed)
print(result)
'''