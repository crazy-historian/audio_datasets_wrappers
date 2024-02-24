import tqdm
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
from typing import Optional, Union, Any


@dataclass
class AudioData:
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


class AudioDataset(Dataset):
    def __init__(
            self,
            audio_data: list[AudioData],
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


def get_audio_data(
        desc_table: pd.DataFrame,
        dir_path: str,
        phoneme_labeler: PhonemeLabeler,
        overlapping_frames: bool = True,
        frame_length: int | None = 1024,
        padding_length: int | None = None
    ) -> list[AudioData]:
    audio_data = list()
    for _, row in tqdm.tqdm(desc_table.iterrows(), total=desc_table.shape[0]):
        metadata = torchaudio.info(Path(dir_path, row.audio_file_path))
        frame_rate = int(metadata.sample_rate)
        sample_width = metadata.bits_per_sample
        t0 = round(row.t0 * frame_rate)
        t1 = round(row.t1 * frame_rate)

        data, _ = torchaudio.load(Path(dir_path, row.audio_file_path))
        data = data[:, t0:t1]
        
        if overlapping_frames is False:
            audio_data.append(AudioData(
                data=data,
                label=
                row.phone_class
                ,
                label_index=phoneme_labeler.get_index_of_phoneme(
                    row.phone_class
                    ),
                frame_rate=frame_rate,
                sample_width=sample_width
            ))
        elif padding_length is not None:
            new_shape = padding_length - data.shape[1]
            data = F.pad(data, (0, new_shape), 'constant', 0.0)
            audio_data.append(AudioData(
                data=data,
                label=row.phone_class,
                label_index=phoneme_labeler.get_index_of_phoneme(
                    row.phone_class
                    ),
                frame_rate=frame_rate,
                sample_width=sample_width
            ))
        else:
            i = 0
            frames = list()
            for _ in range(data.shape[1] // (frame_length // 2) - 1):
                frames.append(AudioData(
                    data=data[:, i: i + frame_length],
                    label=row.phone_class,
                    label_index=phoneme_labeler.get_index_of_phoneme(
                        row.phone_class
                        ),
                    frame_rate=frame_rate,
                    sample_width=sample_width 
                ))
                i += frame_length // 2
            else:
                new_shape = frame_length - data[:, i:].shape[1]
                frames.append(AudioData(
                    data=F.pad(data[:, i:], (0, new_shape), 'constant', 0.0),
                    label=row.phone_class,
                    label_index=phoneme_labeler.get_index_of_phoneme(
                        row.phone_class
                        ),
                    frame_rate=frame_rate,
                    sample_width=sample_width 
                ))
            audio_data.extend(frames)

    return audio_data