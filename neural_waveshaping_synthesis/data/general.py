import os

import gin
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data.sampler import Sampler
import random

class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, batch_size: int, split: str = "train", load_to_memory: bool = True):
        super().__init__()
        # split = "train"
        self.load_to_memory = load_to_memory
        self.batch_size = batch_size
        self.ctr = 0
        self.curr_content = None
        self.split_path = os.path.join(path, split)
        self.data_list = [
            f.replace("audio_", "")
            for f in os.listdir(os.path.join(self.split_path, "audio"))
            if f[-4:] == ".npy"
        ]
        self.data_list = sorted(self.data_list)
        if load_to_memory:
            self.audio = [
                np.load(os.path.join(self.split_path, "audio", "audio_%s" % name))
                for name in self.data_list
            ]
            self.control = [
                np.load(os.path.join(self.split_path, "control", "control_%s" % name))
                for name in self.data_list
            ]

        self.data_mean = np.load(os.path.join(path, "data_mean.npy"))
        self.data_std = np.load(os.path.join(path, "data_std.npy"))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # idx = 10
        name = self.data_list[idx]
        # print(f"In get_item: {name}")
        # print(f"idx: {idx}")
        #
        # content = int(name.split()[-1][0])
        # preset = name[:3]
        # print(f"content: {content}")
        # print(f"preset: {preset}")
        #
        # if self.ctr == 0:
        #     # accept new content
        #     print(f"Accepting new content: {content}")
        #     self.curr_content = content
        # else:
        #     # accept cur_content
        #     print(f"Going with curr_content: {self.curr_content}")
        #     name = list(name)
        #     name[-7] = str(content)
        #     name = "".join(name)
        #     print(f"New name: {name}")

        if self.load_to_memory:
            audio = self.audio[idx]
            control = self.control[idx]
        else:
            audio_name = "audio_%s" % name
            control_name = "control_%s" % name

            audio = np.load(os.path.join(self.split_path, "audio", audio_name))
            control = np.load(os.path.join(self.split_path, "control", control_name))
        denormalised_control = (control * self.data_std) + self.data_mean

        # self.ctr += 1
        # if self.ctr == self.batch_size:
        #     self.ctr = 0
        return {
            "audio": audio,
            "f0": denormalised_control[0:1, :],
            "amp": denormalised_control[1:2, :],
            "control": control,
            "name": os.path.splitext(os.path.basename(name))[0],
        }


class MyBatchSampler(Sampler):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.content_enc = [1,2,3,4]
        self.batches = []
        num_batches = 320//self.batch_size
        for i in range(num_batches):
            cnt = random.choice(self.content_enc)
            presets = random.sample(range(80), self.batch_size)
            batch = [p*4+cnt-1 for p in presets]
            self.batches.append(batch)
        # print(self.batches)
    def __iter__(self):
        for b in self.batches:
            yield b

    def __len__(self):
        return len(self.batches)


@gin.configurable
class GeneralDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root: str,
            batch_size: int = 16,
            load_to_memory: bool = True,
            **dataloader_args
    ):
        super().__init__()
        self.data_dir = data_root
        self.batch_size = batch_size
        self.dataloader_args = dataloader_args
        self.load_to_memory = load_to_memory

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        print(f"CALL TO SETUP: {stage}")

        if stage == "fit":
            self.urmp_train = GeneralDataset(self.data_dir, batch_size=self.batch_size, split="train",
                                             load_to_memory=self.load_to_memory)
            print(f"length of train ds: {len(self.urmp_train)}")
            self.urmp_val = GeneralDataset(self.data_dir, batch_size=self.batch_size, split="val",
                                        load_to_memory=self.load_to_memory)
            print(f"length of val ds: {len(self.urmp_val)}")
            self.urmp_all = GeneralDataset(self.data_dir, batch_size=self.batch_size, split="all",
                                           load_to_memory=self.load_to_memory)
            print(f"length of all ds: {len(self.urmp_all)}")
        elif stage == "test" or stage is None:
            self.urmp_test = GeneralDataset(self.data_dir, batch_size=self.batch_size, split="test",
                                            load_to_memory=self.load_to_memory)

    def _make_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset, batch_sampler= MyBatchSampler(self.batch_size),#self.dataloader_args
        )

    def train_dataloader(self):
        return self._make_dataloader(self.urmp_all)

    def all_dataloader(self):
        return self._make_dataloader(self.urmp_all)

    def trainval_dataloader(self):
        return self._make_dataloader(self.urmp_train)

    def val_dataloader(self):
        return self._make_dataloader(self.urmp_all)

    def test_dataloader(self):
        return self._make_dataloader(self.urmp_test)
