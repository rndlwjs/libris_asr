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
from lightning.pytorch import loggers as pl_loggers
import sentencepiece as spm
from transform import TextTransform, preprocess
#from DS2.ds2 import DeepSpeech2ASR
from Ebranchformer.ebranchformer import EbranchformerASR
from pytorch_lightning.utilities.model_summary import LayerSummary

### Dataset, Dataloader
class LibriSpeechDataset(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, type="train", download=False):
        if type=='train':
            super().__init__(root='/home/rndlwjs/conformer/data', url='train-clean-100', download=False)
            self.aug = True
        else:
            super().__init__(root='/home/rndlwjs/conformer/data', url='test-clean', download=False)
            self.aug = False

        self.TextTransform = TextTransform()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
        self.SpecAug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )

    def __getitem__(self, i):
        wav, sr, transcript, speaker, chapter, utterance = super().__getitem__(i)

        audio = self.transform(wav.squeeze(0)).transpose(0,1)
        if self.aug == True:
            audio = self.SpecAug(audio)
        transcript = preprocess(transcript)
        labels = self.TextTransform.text_to_int(text=transcript)
        label = torch.LongTensor(labels)
        return audio, label

    def __len__(self):
        return super().__len__()

def _collate_fn(batch):
    # batch give lists of _getitem_, [audio1, label1], [audio2, label2], ...
    spectrograms = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    #input_lengths = torch.LongTensor([i[0].size(0)//2 for i in batch])
    input_lengths = torch.LongTensor([i[0].size(0) for i in batch])
    label_lengths = torch.LongTensor([len(i[1]) for i in batch])
    
    # Pad batch to length of longest sample
    #spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3) ##DS [B, 1, 128, T]
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).transpose(1, 2) ##ebranch [B, T, 80]
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

train_dataset = LibriSpeechDataset(type="train")
test_dataset = LibriSpeechDataset(type="test")
test_dataset = torch.utils.data.Subset(test_dataset, [0])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=_collate_fn,)
val_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn,)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=_collate_fn,)

###
epochs = 70

checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor="train_loss",
    mode="min",
    dirpath="logs/ebranch/",
    filename="libris100-{epoch:02d}-{val_loss:.2f}",
)

### Training
seed_everything(42, workers=True)
model = DeepSpeech2ASR()

tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
trainer = Trainer(precision="32-true", 
                    max_epochs=epochs, 
                    accelerator="gpu", 
                    logger=tb_logger,
                    callbacks=[checkpoint_callback],
                    devices=[2])

trainer.fit(model, train_dataloader, val_dataloader)
#trainer.test(model, dataloaders=test_dataloader, ckpt_path="~/qhdd14/hdd14/kyujin/241125_asr_project/libris_asr/logs/libris100-epoch=69-val_loss=0.00.ckpt")
