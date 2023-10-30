import pandas as pd
import textgrid
import tqdm

from pathlib import Path
from audio_datasets_wrappers.dataset import AudioDataset, AudioData, PhonemeLabeler
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
                 os_slash: str = '\\',
                 description_file_path: Optional[str] = None,
                 test_fraction: float = 0.2,
                 percentage: float = 0.5,
                 usage='train',
                 padding_length: int = 0,
                 by_frame: bool = True,
                 frame_length: int = 1024,
                 transform: Callable = None,
                 phone_codes: Union[List[str], str] = None,
                 gender: Optional[str] = None,
                 first_language: Optional[List[str]] = None,
                 phoneme_labeler=PhonemeLabeler()):
        self.padding_length = padding_length
        self.root_dir = root_dir
        self.os_slash = os_slash
        self.description_file_path = description_file_path

        self.usage = usage
        self.gender = gender
        self.first_language = first_language
        self.percentage = percentage

        self.by_frame = by_frame
        self.frame_length = frame_length

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
        self.description_table = self._filter_description_table(percentage, phone_codes, usage, gender, first_language)
        self.audio_fragments = list()

    def _prepare_description(self, test_fraction) -> pd.DataFrame:
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
                            '/'.join(map(str, str(textgrid_file).split('\\')[-3:])),
                            '/'.join(map(str, str(wav_file).split('\\')[-3:])),
                            interval.mark,
                            self.phoneme_labeler[interval.mark],
                            interval.minTime,
                            interval.maxTime
                        ])
                    table.extend(table_rows)

            df = pd.DataFrame(data=table, columns=[
                'dir_id',
                'l1',
                'gender',
                'labels_file_path',
                'audio_file_path',
                'phone_name',
                'phone_class',
                't0',
                't1']
                )
            df['usage'] = 'train'
            df.loc[df.sample(frac=test_fraction).index.to_list(), 'usage'] = 'test'
            df.to_csv('arctic_description.csv', index=False)

            return df

    def _filter_description_table(self, percentage: float, phone_classes: List[str], usage: str, gender: Optional[str], first_language: Optional[str]) -> pd.DataFrame:
        
        self.description_table = self.description_table.loc[self.description_table['usage'] == usage]

        if gender is not None:
            self.description_table = self.description_table.loc[self.description_table['gender'] == gender]

        if first_language is not None:
            dialects = self.description_table['l1'].isin(first_language)
            self.description_table = self.description_table[dialects]

        if phone_classes is not None:
            self.description_table = self.description_table.loc[self.description_table['phone_class'].isin(phone_classes)]

        if percentage is not None:
            self.description_table = self.description_table.sample(frac=percentage)

        return self.description_table
    
    def _get_audio_fragments(self, *args, **kwargs) -> list[AudioData]:
        fragments = list()
        for _, row in tqdm.tqdm(self.description_table.iterrows(), total=self.description_table.shape[0]):
            fragments.extend(self._load_audio_fragment(row, self.root_dir))
        return fragments
    
    def cut_and_load_phonemes(self):
        self.audio_fragments = self._get_audio_fragments()
        print(f'Number of fragments with phonemes: {len(self.audio_fragments)}')
    
    def info(self, pie_radius: float = 1.5):
        print(
            'ACRTIC DATASET DESCRIPTION\n'

            f'Usage: {self.usage}.\n'
            f'Specific gender: {self.gender}.\n'
            f'Specific L1: {self.first_language}.\n'
            f'Percentage: {self.percentage * 100}% of all data.\n'
            f'Number of phonemes: {self.description_table.shape[0]}.\n'
            f'By frame: {self.by_frame}.\n'
            f'Frame_length: {self.frame_length}'
        )
                
        self.description_table['phone_class'].value_counts(normalize=True).plot.pie(
            radius=pie_radius,
            autopct='%1.1f%%'
        )

    def __len__(self) -> int:
        return len(self.audio_fragments)

    def __getitem__(self, item: int) -> AudioData:
        if self.transform:
            audio_data = self.audio_fragments[item]
            audio_data.data = self.transform(audio_data.data)
            return audio_data
        return self.audio_fragments[item]