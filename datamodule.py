import torch
import pandas as pd
import pytorch_lightning as pl

from dataset import PhonemeDataset
from utils import get_audio_data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class PhonemeDataModule(pl.LightningDataModule):
    def __init__(
            self,
            desc_table: pd.DataFrame,
            dataset_dir_path: str,
            batch_size: int = 256,
            train_size: float = 0.8,
            fraction: float = 0.5,
            transform: torch.nn.Module | torch.nn.Sequential | None = None,
            overlapping: bool = True,
            frame_length: int = 1024
        ):
        super().__init__()
        self.desc_table = desc_table
        self.batch_size = batch_size
        self.train_size = train_size
        self.fraction = fraction
        self.transform = transform
        
        self.dataset_dir_path = dataset_dir_path
        self.overlapping = overlapping
        self.frame_length = frame_length

    def setup(self, stage: str):
        if stage == 'fit':
            self.desc_table = self.desc_table.sample(frac=self.fraction).reset_index(drop='index')
            X = self.desc_table.index
            y = self.desc_table.class_index
            train_indicies, val_indicies, _, _ = train_test_split(
                X,
                y,
                train_size=self.train_size,
                stratify=y,
                shuffle=True
            )

            self.train_dataset = PhonemeDataset(
                audio_data=get_audio_data(
                    desc_table=self.desc_table.iloc[train_indicies],
                    dir_path = self.dataset_dir_path,
                    overlapping_frames=True,
                    frame_length=self.frame_length
                ),
                transform=self.transform
            )

            self.val_dataset = PhonemeDataset(
                audio_data=get_audio_data(
                    desc_table=self.desc_table.iloc[val_indicies],
                    dir_path = self.dataset_dir_path,
                    overlapping_frames=True,
                    frame_length=self.frame_length
                ),
                transform=self.transform
            )
        elif stage == 'predict':
            self.predict_dataset = PhonemeDataset(
                audio_data=get_audio_data(
                    desc_table=self.desc_table,
                    dir_path = self.dataset_dir_path,
                    overlapping_frames=True,
                    frame_length=self.frame_length
                ),
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True

            )

    def val_dataloader(self):
        return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

    def predict_dataloader(self):
        return DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
