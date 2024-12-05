import torch
import torchaudio

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")
number = sp.PieceToId("<s>")
number1 = sp.PieceToId("<pad>")
number2 = sp.PieceToId("</s>")
number3 = sp.PieceToId("<blank>")

class LibriSpeechDataset(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, type="train", download=False):
        if type=='train':
            super().__init__(root='/home/rndlwjs/conformer/data', url='train-clean-100', download=False)
            self.aug = True
        if type=='test':
            super().__init__(root='/home/rndlwjs/conformer/data', url='test-clean', download=False)

        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)

    def __getitem__(self, i):
        wav, sr, transcript, speaker, chapter, utterance = super().__getitem__(i)
        text = " ".join(sp.EncodeAsPieces(transcript))
        label = " ".join([str(item) for item in sp.EncodeAsIds(transcript)])

        return self.transform(wav), text, label

    def __len__(self):
        return super().__len__()

a = LibriSpeechDataset()
print(a[0])