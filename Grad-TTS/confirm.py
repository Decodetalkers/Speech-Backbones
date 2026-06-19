import json
from hifi_gan.env import AttrDict

from hifi_gan.env import AttrDict
from hifi_gan.models import Generator as HiFiGAN
from scipy.io.wavfile import write

import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

# Load a pretrained HIFIGAN Vocoder
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_models/tts-hifigan-ljspeech",
    run_opts={"device": "cuda:0"},
)

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTS
from data import TextMelDataset, TextMelBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols
train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale


if __name__ == "__main__":
    print("Initializing data loaders...")

    train_dataset = TextMelDataset(
        train_filelist_path,
        cmudict_path,
        add_blank,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
    )
    test_batch = train_dataset.sample_test_batch(size=4)

    for i, item in enumerate(test_batch):
        mel = item["y"]
        audio = hifi_gan.decode_batch(mel.cpu())

        torchaudio.save("waveform_reconstructed.wav", audio.cpu(), 22050)

