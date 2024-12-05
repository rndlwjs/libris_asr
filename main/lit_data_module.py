import os
import wget
import tarfile
import logging
import shutil
import torchaudio
import sentencepiece as spm
import pytorch_lightning as pl
from typing import Union, List, Tuple, Optional
from omegaconf import DictConfig
from torch.utils.data import DataLoader


## collect transcripts
train_set = torchaudio.datasets.LIBRISPEECH("/home/rndlwjs/conformer/data", url='train-clean-100', download=False)
test_set = torchaudio.datasets.LIBRISPEECH("/home/rndlwjs/conformer/data", url='test-clean', download=False)

train_set = [transcript for wav, sr, transcript, speaker, chapter, utterance in train_set]  
all_text = [transcript for wav, sr, transcript, speaker, chapter, utterance in test_set]
all_text.extend(train_set)

print(len(train_set)); exit()

## tokenization
input_file = 'spm_input.txt'
model_name = 'tokenizer'
model_type = 'unigram'
vocab_size = 5000

with open("spm_input.txt", 'w') as f:
    for t in all_text:
        f.write(f"{t}\n")

cmd = f"--input={input_file} --model_prefix={model_name} --vocab_size={vocab_size} " \
        f"--model_type={model_type} --user_defined_symbols=<blank>"
spm.SentencePieceTrainer.Train(cmd)

## generate manifest file
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")
number = sp.PieceToId("<s>")
number1 = sp.PieceToId("<pad>")
number2 = sp.PieceToId("</s>")
number3 = sp.PieceToId("<blank>")
print(number, number1, number2, number3)
exit()

#p = "i think i am hungry".upper()
#print(sp.EncodeAsPieces(p))
#print(" ".join([str(item) for item in sp.EncodeAsIds(p)]))

with open(f"train.txt", 'w') as f:
    for wav, sr, transcript, speaker, chapter, utterance in test_set:
        text = " ".join(sp.EncodeAsPieces(transcript))
        label = " ".join([str(item) for item in sp.EncodeAsIds(transcript)])

        f.write('%s_%s\t%s\t%s\n' % (speaker, utterance, text, label))

class Vocabulary(object):
    """
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, *args, **kwargs):
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.blank_id = None
        self.vocab_size = None

    def __len__(self):
        return self.vocab_size

    def label_to_string(self, labels):
        raise NotImplementedError

class LibriSpeechVocabulary(Vocabulary):
    """
    Converts label to string for librispeech dataset.

    Args:
        model_path (str): path of sentencepiece model
        vocab_size (int): size of vocab
    """
    def __init__(self, model_path: str, vocab_size: int):
        super(LibriSpeechVocabulary, self).__init__()
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("Please install sentencepiece: `pip install sentencepiece`")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.pad_id = self.sp.PieceToId("<pad>")
        self.sos_id = self.sp.PieceToId("<bos>")
        self.eos_id = self.sp.PieceToId("<eos>")
        self.blank_id = self.sp.PieceToId("<blank>")
        self.vocab_size = vocab_size

    def label_to_string(self, labels):
        if len(labels.shape) == 1:
            return self.sp.DecodeIds([l.item() for l in labels])

        elif len(labels.shape) == 2:
            sentences = list()

            for label in labels:
                sentence = self.sp.DecodeIds([l for l in label])
                sentences.append(sentence)
            return sentences
        else:
            raise ValueError("Unsupported label's shape")

vocab = LibriSpeechVocabulary("tokenizer.model", vocab_size)

## setup

with open("./train.txt", "r") as f:
    transcripts = [i.split("\t")[2] for i in f.readlines()]

print(transcripts[:5])



import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            audio_paths: list,
            transcripts: list,
            apply_spec_augment: bool = False,
            sos_id: int = 1,
            eos_id: int = 2,
            sample_rate: int = 16000,
            num_mels: int = 80,
            frame_length: float = 25.0,
            frame_shift: float = 10.0,
            freq_mask_para: int = 27,
            time_mask_num: int = 4,
            freq_mask_num: int = 2,
    ) -> None:
        super(AudioDataset, self).__init__()
        self.transcripts = list(transcripts)
        self.spec_augment_flags = [False] * len(self.transcripts)
        self.dataset_size = len(self.transcripts)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.num_mels = 80
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
        self.n_fft = int(round(sample_rate * 0.001 * frame_length))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift))

class MelSpectrogramDataset(AudioDataset):
    """ Dataset for mel-spectrogram & transcript matching """
    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        melspectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_mels=self.num_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        return melspectrogram

data_ = MelSpectrogramDataset

class liDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

'''
def _parse_manifest_file(manifest_file_path: str) -> Tuple[list, list]:
    """ Parsing manifest file """
    audio_paths = list()
    transcripts = list()

    with open(manifest_file_path) as f:
        for idx, line in enumerate(f.readlines()):
            audio_path, _, transcript = line.split('\t')
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts


class LightningLibriSpeechDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning Data Module for LibriSpeech Dataset.

    Args:
        configs (DictConfig): configuraion set

    Attributes:
        dataset_path (str): path of librispeech dataset
        apply_spec_augment (bool): flag indication whether to apply spec augment or not
        max_epochs (int): the number of max epochs
        batch_size (int): the size of batch samples
        num_workers (int): the number of cpu workers
        sample_rate (int): sampling rate of audio
        num_mels (int): the number of mfc coefficients to retain.
        frame_length (float): frame length for spectrogram (ms)
        frame_shift (float): length of hop between STFT (short time fourier transform) windows.
        freq_mask_para (int): hyper Parameter for freq masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make
    """
    librispeech_parts = [
        'dev-clean',
        'test-clean',
        'dev-other',
        'test-other',
        'train-clean-100',
        'train-clean-360',
        'train-other-500',
    ]

    def __init__(self, configs: DictConfig) -> None:
        super(LightningLibriSpeechDataModule, self).__init__()
        self.dataset_path = configs.dataset_path
        self.librispeech_dir = 'LibriSpeech'
        self.manifest_paths = [
            f"{configs.dataset_path}/train-960.txt",
            f"{configs.dataset_path}/dev-clean.txt",
            f"{configs.dataset_path}/dev-other.txt",
            f"{configs.dataset_path}/test-clean.txt",
            f"{configs.dataset_path}/test-other.txt",
        ]
        self.dataset = dict()
        self.batch_size = configs.batch_size
        self.apply_spec_augment = configs.apply_spec_augment
        self.max_epochs = configs.max_epochs
        self.batch_size = configs.batch_size
        self.num_workers = configs.num_workers
        self.sample_rate = configs.sample_rate
        self.num_mels = configs.num_mels
        self.frame_length = configs.frame_length
        self.frame_shift = configs.frame_shift
        self.freq_mask_para = configs.freq_mask_para
        self.time_mask_num = configs.time_mask_num
        self.freq_mask_num = configs.freq_mask_num
        self.logger = logging.getLogger(__name__)

        if configs.feature_extract_method == 'spectrogram':
            self.audio_dataset = SpectrogramDataset
        elif configs.feature_extract_method == 'melspectrogram':
            self.audio_dataset = MelSpectrogramDataset
        elif configs.feature_extract_method == 'mfcc':
            self.audio_dataset = MFCCDataset
        elif configs.feature_extract_method == 'fbank':
            self.audio_dataset = FBankDataset
        else:
            raise ValueError(f"Unsupported `feature_extract_method`: {configs.feature_extract_method}")

    def _download_librispeech(self) -> None:
        """
        Download librispeech dataset.
            - train-960(train-clean-100, train-clean-360, train-other-500)
            - dev-clean
            - dev-other
            - test-clean
            - test-other
        """
        base_url = "http://www.openslr.org/resources/12"
        train_dir = "train-960"

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        for part in self.librispeech_parts:
            self.logger.info(f"Librispeech-{part} download..")
            url = f"{base_url}/{part}.tar.gz"
            wget.download(url, self.dataset_path)

            self.logger.info(f"Un-tarring archive {self.dataset_path}/{part}.tar.gz")
            tar = tarfile.open(f"{self.dataset_path}/{part}.tar.gz", mode="r:gz")
            tar.extractall()
            tar.close()
            os.remove(f"{self.dataset_path}/{part}.tar.gz")

        self.logger.info("Merge all train packs into one")

        if not os.path.exists(os.path.join(self.dataset_path, self.librispeech_dir)):
            os.mkdir(os.path.join(self.dataset_path, self.librispeech_dir))
        if not os.path.exists(os.path.join(self.dataset_path, self.librispeech_dir, train_dir)):
            os.mkdir(os.path.join(self.dataset_path, self.librispeech_dir, train_dir))

        for part in self.librispeech_parts[:-3]:    # dev, test
            shutil.move(
                os.path.join(self.librispeech_dir, part),
                os.path.join(self.dataset_path, self.librispeech_dir, part),
            )

        for part in self.librispeech_parts[-3:]:    # train
            path = os.path.join(self.librispeech_dir, part)
            subfolders = os.listdir(path)
            for subfolder in subfolders:
                shutil.move(
                    os.path.join(path, subfolder),
                    os.path.join(self.dataset_path, self.librispeech_dir, train_dir, subfolder),
                )

    def _generate_manifest_files(self, vocab_size: int) -> None:
        """
        Generate manifest files.
        Format: {audio_path}\t{transcript}\t{numerical_label}

        Args:
            vocab_size (int): size of subword vocab

        Returns:
            None
        """
        self.logger.info("Generate Manifest Files..")
        transcripts_collection = collect_transcripts(
            os.path.join(self.dataset_path, self.librispeech_dir),
            self.librispeech_dir,
        )
        prepare_tokenizer(transcripts_collection[0], vocab_size)

        for idx, part in enumerate(['train-960', 'dev-clean', 'dev-other', 'test-clean', 'test-other']):
            generate_manifest_file(self.dataset_path, part, transcripts_collection[idx])

    def prepare_data(self, download: bool = False, vocab_size: int = 5000) -> Vocabulary:
        """
        Prepare librispeech data

        Args:
            download (bool): if True, download librispeech dataset
            vocab_size (int): size of subword vocab

        Returns:
            None
        """
        if download:
            self._download_librispeech()
        self._generate_manifest_files(vocab_size)
        return LibriSpeechVocabulary("tokenizer.model", vocab_size)

    def setup(self, stage: Optional[str] = None, vocab: Vocabulary = None) -> None:
        """ Split dataset into train, valid, and test. """
        splits = ['train', 'val-clean', 'val-other', 'test-clean', 'test-other']

        for idx, (path, split) in enumerate(zip(self.manifest_paths, splits)):
            audio_paths, transcripts = _parse_manifest_file(path)
            self.dataset[split] = self.audio_dataset(
                dataset_path=self.dataset_path,
                audio_paths=audio_paths,
                transcripts=transcripts,
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
                apply_spec_augment=self.apply_spec_augment if idx == 0 else False,
                sample_rate=self.sample_rate,
                num_mels=self.num_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                freq_mask_para=self.freq_mask_para,
                freq_mask_num=self.freq_mask_num,
                time_mask_num=self.time_mask_num,
            )

    def train_dataloader(self) -> DataLoader:
        train_sampler = BucketingSampler(self.dataset['train'], batch_size=self.batch_size)
        return AudioDataLoader(
            dataset=self.dataset['train'],
            num_workers=self.num_workers,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        val_clean_sampler = BucketingSampler(self.dataset['val-clean'], batch_size=self.batch_size)
        val_other_sampler = BucketingSampler(self.dataset['val-other'], batch_size=self.batch_size)
        return [
            AudioDataLoader(
                dataset=self.dataset['val-clean'],
                num_workers=self.num_workers,
                batch_sampler=val_clean_sampler,
            ),
            AudioDataLoader(
                dataset=self.dataset['val-other'],
                num_workers=self.num_workers,
                batch_sampler=val_other_sampler,
            ),
        ]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_clean_sampler = BucketingSampler(self.dataset['test-clean'], batch_size=self.batch_size)
        test_other_sampler = BucketingSampler(self.dataset['test-other'], batch_size=self.batch_size)
        return [
            AudioDataLoader(
                dataset=self.dataset['test-clean'],
                num_workers=self.num_workers,
                batch_sampler=test_clean_sampler,
            ),
            AudioDataLoader(
                dataset=self.dataset['test-other'],
                num_workers=self.num_workers,
                batch_sampler=test_other_sampler,
            ),
        ]
'''