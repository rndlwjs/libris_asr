import os
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import Sampler
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.module import LightningModule
from torchmetrics import functional as FM
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
import sentencepiece as spm
from transform import TextTransform, preprocess
from DS2.ds2 import DeepSpeech2ASR
from pytorch_lightning.utilities.model_summary import LayerSummary

### Dataset, Dataloader
class LibriSpeechDataset(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, type="train", download=False):
        if type=='train':
            super().__init__(root='/home/rndlwjs/conformer/data', url='train-clean-100', download=False)
            self.aug = True

        self.TextTransform = TextTransform()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)

    def __getitem__(self, i):
        wav, sr, transcript, speaker, chapter, utterance = super().__getitem__(i)

        text = preprocess(transcript)
        labels = self.TextTransform.text_to_int(text=text)

        #audio = self.transform(wav.squeeze(0)).transpose(0,1)
        audio = self.transform(wav).squeeze(0).transpose(0,1)
        #print(audio.shape); exit()
        label = torch.LongTensor(labels)
        return audio, label

    def __len__(self):
        return super().__len__()

def _collate_fn(batch, pad_id=0):
    # batch give lists of _getitem_, [audio1, label1], [audio2, label2], ...
    spectrograms = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    batch_size = len(spectrograms)

    # Pad batch to length of longest sample
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3) ##DS
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    #audio_len = torch.LongTensor([spectrograms.size(1)] * batch_size)
    #label_len = torch.LongTensor([labels.size(1) - 1] * batch_size)
    # from DS2
    audio_len = torch.LongTensor([spectrograms.shape[0]//2] * batch_size)
    label_len = torch.LongTensor([len(labels)] * batch_size)

    return spectrograms, labels, audio_len, label_len


train_dataset = LibriSpeechDataset()
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=_collate_fn,)
val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn,)[:10]
test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,)

###
epochs = 10

### Training
seed_everything(42, workers=True)
model = DeepSpeech2ASR()
trainer = Trainer(precision=16, max_epochs=epochs, accelerator="gpu", devices=[0, 2])
trainer.fit(model, train_dataloader, val_dataloader)
#trainer.test(model, dataloaders=test_dataloader, ckpt_path="/home/rndlwjs/qhdd14/hdd14/kyujin/241125_asr_project/libris_asr/main/lightning_logs/version_0/checkpoints/epoch=38-step=17394.ckpt")
