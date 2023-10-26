import textgrid
import pandas as pd

from pathlib import Path
from dataset import AudioDataset, AudioData, PhonemeLabeler
from typing import (
    Union,
    Optional,
    Callable,
    List
)


class LibriSpeechDataset(AudioDataset):
    """  """

    def __init__(self,
                 data_dir: str,
                 description_file_path: Optional[str] = None,
                 data_type: str = 'dev-clean',
                 usage: str = 'train',
                 padding: int = 0,
                 data_percentage: Optional[float] = None,
                 transform: Optional[Callable] = None,
                 phone_codes: Union[List[str], str] = None,
                 gender: Optional[str] = None,
                 dialect: Optional[List[str]] = None,
                 phoneme_labeler: PhonemeLabeler = PhonemeLabeler()
                 ):
        super().__init__()
        self.padding = padding
        self.data_dir = data_dir
        self.description_file_path = description_file_path
        self.data_percentage = data_percentage
        self.phone_codes = phone_codes
        self.transform = transform
        self.phoneme_labeler = phoneme_labeler

        self.description_table = self._prepare_description()
        # self.description_table = self._filter_description_table()
        # self.audio_fragments = self._get_audio_fragments()

    def _prepare_description(self):
        if self.description_file_path is not None and Path(self.description_file_path).is_file():
            return pd.read_csv(self.description_file_path)
        else:
            table = list()

            for directory in Path(self.data_dir).iterdir():
                data_type, usage = str(directory.stem).split('-')
                print(data_type, usage)
                for sub_directory in directory.iterdir():
                    for textgrid_file, flac_file in zip(
                        sorted(sub_directory.glob('*/*.TextGrid')),
                        sorted(sub_directory.glob('*/*.flac'))
                        ):
                        table_rows = list()
                        for interval in textgrid.TextGrid.fromFile(textgrid_file)[1]:
                            table_rows.append([
                                data_type, 
                                usage, 
                                # speaker id (?)
                                # speaker sex (?)
                                # speech task id (?)
                                '/'.join(map(str, str(textgrid_file).split('\\')[-4:])),
                               '/'.join(map(str, str(flac_file).split('\\')[-4:])),
                                interval.mark,
                                interval.minTime,
                                interval.maxTime
                            ])
                        table.extend(table_rows)
            
            df = pd.DataFrame(
                data=table, columns=[
                    'data_kind',
                    'quality',
                     # speaker id (?)
                    # speaker sex (?)
                    # speech task id (?)
                    'textgrid_file_path',
                    'flac_file_path',
                    'phone_name',
                    't0',
                    't1'
                ]
            )
            df.to_csv('libri_speech_description.csv', index=False)

            return df
    
    def _filter_description_table(self, data_type, usage):
        self.description_table = self.description_table.loc[self.description_table.usage == 'usage']
        self.description_table = self.description_table.loc[self.description_table.data_type == 'data_type']
        self.description_table = self.description_table.sample(frac=self.data_percentage)
            
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