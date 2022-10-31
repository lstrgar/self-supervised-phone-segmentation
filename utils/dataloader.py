from copy import deepcopy
import torch
from torch.utils.data import Dataset
from os.path import join
from boltons.fileutils import iter_find_files
import torchaudio
import math
from functools import partial
from torch.utils.data import DataLoader
import numpy as np
import random

LAYERS = [(10,5,0), (3,2,0), (3,2,0), (3,2,0), (3,2,0), (2,2,0), (2,2,0)]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_fn_padd(batch):
    wavs = [t[0] for t in batch]
    sr = [t[1] for t in batch]
    seg_raw = [t[2] for t in batch]
    seg_aligned = [t[3] for t in batch]
    phonemes = [t[4] for t in batch]
    bin_labels = [t[5] for t in batch]
    lengths = [t[6] for t in batch]
    fnames = [t[7] for t in batch]
    padded_wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0)
    padded_bin_labels = torch.nn.utils.rnn.pad_sequence(bin_labels, batch_first=True, padding_value=0)
    return padded_wavs, seg_aligned, padded_bin_labels, phonemes, lengths, fnames

def spectral_size(wav_len, layers):
    for kernel, stride, padding in layers:
        wav_len = math.floor((wav_len + 2*padding - 1*(kernel-1) - 1)/stride + 1)
    return wav_len

def construct_mask(lengths, device):
    lengths = torch.tensor(lengths)
    max_len = lengths.max()
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask.to(device)

def get_subset(dataset, percent):
    A_split = int(len(dataset) * percent)
    B_split = len(dataset) - A_split
    dataset, _ = torch.utils.data.random_split(dataset, [A_split, B_split])
    return dataset

def get_dloaders(cfg, layers, logger, g=None):
    if cfg.data == "timit":
        train, val, test = TrainTestDataset.get_datasets(
            path=cfg.timit_path, 
            val_ratio=cfg.val_ratio, 
            train_percent=cfg.train_percent, 
            layers=layers, 
        )
    elif cfg.data == "buckeye":
        train, val, test = TrainValTestDataset.get_datasets(
            path=cfg.buckeye_path,
            layers=layers, 
            train_percent=cfg.train_percent,
        )
    else:
        raise ValueError("Provided dataset not supported")

    logger.info("Train set size: {}".format(len(train)))
    logger.info("Val set size: {}".format(len(val)))
    logger.info("Test set size: {}".format(len(test)))
    
    trainloader = DataLoader(
        train, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn_padd
    )
    valloader = DataLoader(
        val, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn_padd
    )
    testloader = DataLoader(
        test, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn_padd
    )
    return trainloader, valloader, testloader

class WavPhnDataset(Dataset):
    def __init__(self, path, layers=LAYERS, files=None):
        self.path = path
        self.layers = layers
        self.spectral_size = partial(spectral_size, layers=layers)
        if files is None:
            self.data = list(iter_find_files(self.path, "*.wav"))
        else:
            self.data = files
        self.files = files
        super(WavPhnDataset, self).__init__()

    @staticmethod
    def get_datasets(path):
        raise NotImplementedError

    def process_file(self, wav_path):
        phn_path = wav_path.replace("wav", "phn")
        audio, sr = torchaudio.load(wav_path)
        audio = audio[0]
        audio_len = len(audio)
        spectral_len = self.spectral_size(audio_len)
        len_ratio = (audio_len / spectral_len)

        with open(phn_path, "r") as f:
            lines = f.readlines()
            lines = list(map(lambda line: line.split(" "), lines))
            scaled_times = torch.FloatTensor(list(map(lambda line: (int(int(line[0]) / len_ratio), int(int(line[1]) / len_ratio)), lines)))[1:-1]
            times = torch.FloatTensor(list(map(lambda line: (int(line[0]), int(line[1])), lines)))[1:-1]
            phonemes = list(map(lambda line: line[2].strip(), lines))[1:-1]
            bin_labels = torch.zeros(spectral_len).float()
            bin_labels[torch.tensor([scaled_times[0][0].item(), scaled_times[0][1].item()] + [s[1].item() for s in scaled_times[1:]], dtype=int)] = 1.0

        return audio, sr, times.tolist(), scaled_times.tolist(), bin_labels, phonemes, wav_path

    def __getitem__(self, idx):
        audio, sr, seg, seg_scaled, bin_labels, phonemes, fname = self.process_file(self.data[idx])
        return audio, sr, seg, seg_scaled, phonemes, bin_labels, self.spectral_size(len(audio)), fname

    def __len__(self):
        return len(self.data)


class TrainTestDataset(WavPhnDataset):
    def __init__(self, path, layers=LAYERS, files=None):
        super(TrainTestDataset, self).__init__(path, layers, files)

    @staticmethod
    def get_datasets(path, val_ratio=0.1, train_percent=1.0, layers=LAYERS, files=None):
        train_dataset = TrainTestDataset(join(path, 'train'), layers, files)
        test_dataset  = TrainTestDataset(join(path, 'test'), layers, files)
        train_len   = len(train_dataset)
        train_split = int(train_len * (1 - val_ratio))
        val_split   = train_len - train_split
        train_holdout = int(train_split * (1 - train_percent))
        train_split -= train_holdout
        dataset_copy = deepcopy(train_dataset)
        train_dataset, val_dataset, _ = torch.utils.data.random_split(train_dataset, [train_split, val_split, train_holdout])
        train_dataset.dataset = dataset_copy
        train_dataset.path = join(path, 'train')
        val_dataset.path = join(path, 'train')
        return train_dataset, val_dataset, test_dataset


class TrainValTestDataset(WavPhnDataset):
    def __init__(self, path, layers=LAYERS, files=None):
        super(TrainValTestDataset, self).__init__(path, layers, files)

    @staticmethod
    def get_datasets(path, layers=LAYERS, files=None, train_percent=1.0):
        train_dataset = TrainValTestDataset(join(path, 'train'), layers=layers, files=files)
        if train_percent != 1.0:
            train_dataset = get_subset(train_dataset, train_percent)
            train_dataset.path = join(path, 'train')
        val_dataset = TrainValTestDataset(join(path, 'val'), layers=layers, files=files)
        test_dataset = TrainValTestDataset(join(path, 'test'), layers=layers, files=files)

        return train_dataset, val_dataset, test_dataset