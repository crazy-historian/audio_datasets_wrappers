import pandas as pd
import textgrid

from pathlib import Path
from dataset import AudioDataset, AudioData, AudioFragment, PhonemeLabeler
from typing import (
    Union,
    Optional,
    Callable,
    List
)

class ArcticDataset(AudioDataset):
    """ ARCTIC L2 dataset """

    def __init__(self,
                 root_dir: str,
                 description_file_path: Optional[str] = None,
                 test_fraction: float = 0.2,
                 data_percentage: float = 0.5,
                 usage='train',
                 padding: int = 0,
                 transform: Callable = None,
                 phone_codes: Union[List[str], str] = None,
                 gender: Optional[str] = None,
                 first_language: Optional[List[str]] = None,
                 phoneme_labeler=PhonemeLabeler()):
        self.padding = padding
        self.root_dir = root_dir
        self.description_file_path = description_file_path

        self.transform = transform
        self.phoneme_labeler = phoneme_labeler

        self.phone_codes = phone_codes
        self.speaker_description = {
            'ABA': ['Arabic', 'M'],
            'SKA': ['Arabic', 'F'],
            'YBAA': ['Arabic', 'M'],
            'ZHAA': ['Arabic', 'F'],
            'BWC': ['Mandarin', 'M'],
            'LXC': ['Mandarin', 'F'],
            'NCC': ['Mandarin', 'F'],
            'TXHC': ['Mandarin', 'M'],
            'ASI': ['Hindi', 'M'],
            'RRBI': ['Hindi', 'M'],
            'SVBI': ['Hindi', 'F'],
            'TNI': ['Hindi', 'F'],
            'HJK': ['Korean', 'F'],
            'HKK': ['Korean', 'M'],
            'YDCK': ['Korean', 'F'],
            'YKWK': ['Korean', 'M'],
            'EBVS': ['Spanish', 'M'],
            'ERMS': ['Spanish', 'M'],
            'MBMPS': ['Spanish', 'F'],
            'NJS': ['Spanish', 'F'],
            'HQTV': ['Vietnamese', 'M'],
            'PNV': ['Vietnamese', 'F'],
            'THV': ['Vietnamese', 'F'],
            'TLV': ['Vietnamese', 'M']
        }
        self.description_table = self._prepare_description(test_fraction)
        self.description_table = self._filter_description_table(data_percentage, test_fraction, usage, gender, first_language)
        self.audio_fragments = self._get_audio_fragments()

    def _prepare_description(self, test_fraction: float) -> pd.DataFrame:
        if self.description_file_path is not None and Path(self.description_file_path).is_file():
            return pd.read_csv(self.description_file_path)
        else:
            table = list()
            for speaker_dir in Path(self.root_dir).iterdir():
                for textgrid_file, wav_file in zip(
                        sorted(Path(speaker_dir, 'textgrid').iterdir()),
                        sorted(Path(speaker_dir, 'wav').iterdir())
                    ): 
                    table_rows = list()
                    for interval in textgrid.TextGrid.fromFile(textgrid_file)[1]:
                        table_rows.append([
                            speaker_dir.stem,
                            self.speaker_description[speaker_dir.stem][0],
                            self.speaker_description[speaker_dir.stem][1],
                            str(textgrid_file),
                            str(wav_file),
                            interval.mark,
                            interval.minTime,
                            interval.maxTime
                        ])
                    table.extend(table_rows)
                # break

            df = pd.DataFrame(data=table, columns=[
                'nickname', 'l1', 'gender', 'labels_file_path',
                'wav_file_path', 'phone_name', 't0', 't1'])
            df['usage'] = None
            df.to_csv('arctic_description.csv', index=False)

            return df

    def _filter_description_table(self, data_percentage: float, test_fraction:float, usage: str, gender: Optional[str], first_language: Optional[str]) -> pd.DataFrame:
        self.description_table.loc[self.description_table.sample(frac=test_fraction).index.to_list(), 'usage'] = 'test'
        self.description_table = self.description_table.loc[self.description_table['usage'] == usage]
        self.description_table = self.description_table.sample(frac=data_percentage)

        if gender is not None:
            self.description_table = self.description_table.loc[self.description_table['gender'] == gender]

        if first_language is not None:
            dialects = self.description_table['l1'].isin(first_language)
            self.description_table = self.description_table[dialects]

        return self.description_table
    
    def _get_audio_fragments(self, *args, **kwargs) -> list[AudioData]:
        fragments = list()
        for _, row in self.description_table.iterrows():
            fragments.append(self._load_audio_fragment(row))
            break
        return fragments

    def __len__(self) -> int:
        return len(self.audio_fragments)

    def __getitem__(self, item: int) -> AudioData:
        if self.transform:
            audio_data = self.audio_fragments[item]
            audio_data.data = self.transform(audio_data.data)
            return audio_data
        return self.audio_fragments[item]