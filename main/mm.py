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
from models.model import ConformerLSTMModel
from branchformer.model import BranchformerLSTMModel
from pytorch_lightning.utilities.model_summary import LayerSummary

### Make Tokenizer
if not os.path.exists("tokenizer.vocab"):
    #### collect transcripts
    train_set = torchaudio.datasets.LIBRISPEECH("/home/rndlwjs/conformer/data", url='train-clean-100', download=False)
    test_set = torchaudio.datasets.LIBRISPEECH("/home/rndlwjs/conformer/data", url='test-clean', download=False)

    print("Iterating train_set")
    train_set = [transcript for wav, sr, transcript, speaker, chapter, utterance in train_set]
    print("Iterating test_set")
    all_text = [transcript for wav, sr, transcript, speaker, chapter, utterance in test_set]
    all_text.extend(train_set)

    with open("spm_input.txt", 'w') as f:
        for t in all_text:
            f.write(f"{t}\n")

    #### tokenization
    input_file = 'spm_input.txt'
    model_name = 'tokenizer'
    model_type = 'unigram'
    vocab_size = 5000

    print("Creating Tokenizer file")
    cmd = f"--input={input_file} --model_prefix={model_name} --vocab_size={vocab_size} " \
            f"--model_type={model_type} --user_defined_symbols=<blank>"
    spm.SentencePieceTrainer.Train(cmd)
    print("Finish training Tokenizer")

### Load Tokenizer
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")
sos = sp.PieceToId("<s>")
pad = sp.PieceToId("<pad>")
eos = sp.PieceToId("</s>")
blank = sp.PieceToId("<blank>")


### Dataset, Dataloader
class LibriSpeechDataset(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, type="train", download=False):
        if type=='train':
            super().__init__(root='/home/rndlwjs/conformer/data', url='test-clean', download=False)
            self.aug = True

        self.TextTransform = TextTransform()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)

    def __getitem__(self, i):
        wav, sr, transcript, speaker, chapter, utterance = super().__getitem__(i)
        #text = " ".join(sp.EncodeAsPieces(transcript))
        #labels = sp.EncodeAsIds(transcript) #" ".join([str(item) for item in sp.EncodeAsIds(transcript)])
        text = preprocess(transcript)
        labels = self.TextTransform.text_to_int(text=text)

        audio = self.transform(wav.squeeze(0)).transpose(0,1)
        label = torch.LongTensor(labels)
        return audio, label

    def __len__(self):
        return super().__len__()


train_dataset = LibriSpeechDataset()

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,)

empty = []

from tqdm import tqdm

for i,j in tqdm(train_dataloader):
    empty.append(j.size(1))

print(max(empty))

#test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,)