import torch
import soundfile as sf
import soxr
from speaker_embed import *
from preprocess_audio import *
from speaker_model import *
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_PATH = "./data/vctk/VCTK_subset_20/wav/"
MODEL_PATH = "./encoder.pt"
SPEAKER_EMBED_PATH = "./data/vctk/speaker_embed_no_split/"
# model = torch.load(MODEL_PATH)
# x, y: numpy.array
for dir_name in os.listdir(DATA_PATH):
    if dir_name.startswith('.'):
        continue
    data_dir = os.path.join(DATA_PATH, dir_name)
    speaker_dir = os.path.join(SPEAKER_EMBED_PATH, dir_name)
    if not os.path.exists(speaker_dir):
        os.mkdir(speaker_dir)
    for filename in os.listdir(data_dir):
        wav_path = os.path.join(data_dir, filename)
        embed_path = os.path.join(speaker_dir, filename).replace(".wav", ".pt")

        x, sr = sf.read(wav_path)

        y = soxr.resample(
            x,          # 1D(mono) or 2D(frames, channels) array input
            48000,      # input samplerate
            16000       # target samplerate
        )

        # Extract speaker embeddings

        load_model(MODEL_PATH, device=DEVICE)
        # (num_frames, 40)
        frames = wav_to_mel_spectrogram(y)
        # (256,)
        print(frames[None, ...].shape)
        embed = embed_frames_batch(frames[None, ...])[0]
        #torch.save(embed, embed_path)
        print("Done: ", embed_path)
        #print(embed.shape)
    break
        #embed_utterance(wav, using_partials=False, return_partials=False)
