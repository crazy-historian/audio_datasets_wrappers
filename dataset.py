import pandas as pd
import numpy as np
import textgrid
import torchaudio
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from torch.utils.data import Dataset

import torch.nn.functional as F
from dataclasses import dataclass, astuple
from typing import Optional, Union, List, Callable


@dataclass
class AudioData:
    data: Union[bytes, np.ndarray, torch.Tensor]
    label: str
    frame_rate: int
    sample_width: int

    def __iter__(self):
        return iter(astuple(self))


class PhonemeLabeler:
    def __init__(self, phoneme_classes: Optional[dict[str, list]] = None, mode: Optional[str] = 'default'):
        self.mode = mode
        self.phoneme_classes = phoneme_classes

    def __getitem__(self, phoneme_label: str) -> str:
        if self.mode == 'default':
            return phoneme_label
        else:
            for phoneme_class, phoneme_labels in self.phoneme_classes.items():
                if phoneme_label in phoneme_labels:
                    return phoneme_class
            else:
                return phoneme_label


class AudioDataset(Dataset, ABC):
    @abstractmethod
    def _prepare_description(self, *args, **kwargs) -> pd.DataFrame:
        ...

    @abstractmethod
    def _filter_description_table(self, *args, **kwargs) -> pd.DataFrame:
        ...

    @abstractmethod
    def _get_audio_fragments(self, *args, **kwargs) -> list[AudioData]:
        ...

    def _load_audio_fragment(self, audio_fragment: pd.Series, root_dir: str) -> AudioData:
        metadata = torchaudio.info(Path(root_dir, audio_fragment.audio_file_path))
        frame_rate = int(metadata.sample_rate)
        sample_width = metadata.bits_per_sample
        t0 = round(audio_fragment.t0 * frame_rate)
        t1 = round(audio_fragment.t1 * frame_rate)

        data, _ = torchaudio.load(Path(root_dir, audio_fragment.audio_file_path))
        data = data[:, t0:t1]

        if self.padding_length != 0:
            new_shape = self.padding_length - data.shape[1]
            data = F.pad(data, (0, new_shape), 'constant', 0.0)

        
        return AudioData(
            data=data,
            label=audio_fragment.phone_name,
            frame_rate=frame_rate,
            sample_width=sample_width
        )