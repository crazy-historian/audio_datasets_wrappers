import tqdm
import pandas as pd
import numpy as np
import textgrid
import torchaudio
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset

from dataclasses import dataclass, astuple
from typing import Optional, Union, Any


@dataclass
class PhonemeData:
    data: Union[bytes, np.ndarray, torch.Tensor]
    label: str
    label_index: int
    frame_rate: int
    sample_width: int

    def __iter__(self):
        return iter(astuple(self))

class PhonemeLabeler:
    def __init__(self, phoneme_classes: dict[str, list]):
        self.phoneme_classes = phoneme_classes

    def __getitem__(self, phoneme_label: str) -> str:
        for phoneme_class, phoneme_labels in self.phoneme_classes.items():
            if phoneme_label in phoneme_labels:
                return phoneme_class
        else:
            return 'others'
    
    def get_index_of_phoneme(self, phoneme_label: str):
        return list(self.phoneme_classes.keys()).index(phoneme_label)

class PhonemeDataset(Dataset):
    def __init__(
            self,
            audio_data: list[PhonemeData],
            transform: torch.nn.Module | torch.nn.Sequential | None = None
        ) -> None:
        super().__init__()
        self.audio_data = audio_data
        self.transform = transform
    
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, index: int) -> Any:
        # TODO: pass default transform = nn.Indentity()
        if self.transform:
            audio_data = self.audio_data[index]
            audio_data.data = self.transform(audio_data.data)
            return audio_data, self.audio_data[index].label_index
        return self.audio_data[index].data, self.audio_data[index].label_index
