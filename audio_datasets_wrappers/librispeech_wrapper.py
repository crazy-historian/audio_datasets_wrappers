import tqdm
import textgrid
import pandas as pd

from pathlib import Path
from audio_datasets_wrappers.dataset import AudioDataset, AudioData, PhonemeLabeler
from typing import (
    Union,
    Optional,
    Callable,
    List
)


class LibriSpeechDataset(AudioDataset):
    """  """

    def __init__(self,
                 root_dir: str,
                 os_slash: str = '\\',
                 description_file_path: Optional[str] = None,
                 data_kind: Optional[str] = 'dev',
                 quality: Optional[str] = 'clean',
                 padding_length: int = 0,
                 by_frame: bool = True,
                 frame_length: int = 1024,
                 percentage: Optional[float] = None,
                 transform: Optional[Callable] = None,
                 phone_codes: Union[List[str], str] = None,
                 phoneme_labeler: PhonemeLabeler = PhonemeLabeler()
                 ):
        super().__init__()
        self.padding_length = padding_length
        self.root_dir = root_dir

        self.by_frame = by_frame
        self.frame_length = frame_length

        self.data_kind = data_kind
        self.quality = quality

        self.os_slash = os_slash
        self.description_file_path = description_file_path
        self.percentage = percentage
        self.phone_codes = phone_codes
        self.transform = transform
        self.phoneme_labeler = phoneme_labeler

        self.description_table = self._prepare_description()
        self.description_table = self._filter_description_table(data_kind, quality, phone_codes, percentage)
        self.audio_fragments = list()

    def _prepare_description(self):
        if self.description_file_path is not None and Path(self.description_file_path).is_file():
            return pd.read_csv(self.description_file_path)
        else:
            table = list()

            for directory in Path(self.root_dir).iterdir():
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
                                self.phoneme_labeler[interval.mark],
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
                    'audio_file_path',
                    'phone_name',
                    'phone_class',
                    't0',
                    't1'
                ]
            )
            df.to_csv('libri_speech_description.csv', index=False)

            return df
    
    def _filter_description_table(self, data_kind, quality,phone_classes, percentage):
        if data_kind is not None:
            self.description_table = self.description_table.loc[self.description_table.data_kind == data_kind]

        if quality is not None:
            self.description_table = self.description_table.loc[self.description_table.quality == quality]

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

    def info(self, pie_radius: float = 1.5):
        print(
            'LIBRISPEECH DATASET DESCRIPTION\n'

            f'Specific data kind: {self.data_kind}.\n'
            f'Specific quality: {self.quality}.\n'
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
    
    def cut_and_load_phonemes(self):
        self.audio_fragments = self._get_audio_fragments()
        print(f'Number of fragments with phonemes: {len(self.audio_fragments)}')

    def __getitem__(self, item: int) -> AudioData:
        if self.transform:
            audio_data = self.audio_fragments[item]
            audio_data.data = self.transform(audio_data.data)
            return audio_data
        return self.audio_fragments[item]