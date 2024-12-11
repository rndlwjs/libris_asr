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
            super().__init__(root='/home/rndlwjs/conformer/data', url='train-clean-100', download=False)
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

def _collate_fn(batch, pad_id=0):
    # batch give lists of _getitem_, [audio1, label1], [audio2, label2], ...
    spectrograms = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    batch_size = len(spectrograms)

    # Pad batch to length of longest sample
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    audio_len = torch.LongTensor([spectrograms.size(1)] * batch_size)
    label_len = torch.LongTensor([labels.size(1) - 1] * batch_size)

    return spectrograms, labels, audio_len, label_len


train_dataset = LibriSpeechDataset()
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=_collate_fn,)
val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn,)
test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,)

#for n, i in enumerate(train_dataloader):
#    audio, label, audio_len, label_len = i
#    print(audio_len.shape, label_len.shape)
#    if n == 5: break

### Hyperparameters
epochs        = 70

### Training
seed_everything(42, workers=True)
model = ConformerLSTMModel()

param = LayerSummary(model).num_parameters / 1000000
print("The size of the model is: ", round(param, 2))

trainer = Trainer(precision=16, max_epochs=epochs)
#trainer.fit(model, train_dataloader)
trainer.test(model, dataloaders=test_dataloader, ckpt_path="/home/rndlwjs/qhdd14/hdd14/kyujin/241125_asr_project/libris_asr/main/lightning_logs/version_0/checkpoints/epoch=38-step=17394.ckpt")