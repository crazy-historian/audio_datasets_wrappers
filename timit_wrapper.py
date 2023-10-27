import pandas as pd

from pathlib import Path
from dataset import AudioDataset, AudioData, PhonemeLabeler
from typing import (
    Union,
    Optional,
    Callable,
    List
)


class TIMITDataset(AudioDataset):
    """
    The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus
    
    """

    def __init__(self,
                 root_dir: str,
                 description_file_path: Optional[str] = None,
                 usage: str = 'TRAIN',
                 os_slash: str = '\\',
                 padding_length: int = 0,
                 by_frame: bool = True,
                 frame_length: int = 1024,
                 percentage: Optional[float] = None,
                 transform: Optional[Callable] = None,
                 phone_codes: Union[List[str], str] = None,
                 gender: Optional[str] = None,
                 dialect: Optional[List[str]] = None,
                 phoneme_labeler: PhonemeLabeler = PhonemeLabeler()
                 ):
        super().__init__()
        self.padding_length = padding_length
        self.timit_constant = 15987
        self.root_dir = root_dir
        self.by_frame = by_frame
        self.frame_length = frame_length
        self.description_file_path = description_file_path
        self.os_slash = os_slash
        self.phone_codes = phone_codes
        self.transform = transform
        self.phoneme_labeler = phoneme_labeler

        self.description_table = self._prepare_description()
        self.description_table = self._filter_description_table(usage, phone_codes, percentage, gender, dialect)
        self.audio_fragments = self._get_audio_fragments()

    def _prepare_description(self):
        if self.description_file_path is not None and Path(self.description_file_path).is_file():
            return pd.read_csv(self.description_file_path)
        else:
            dialects = {'DR1': 'New England', 'DR2': 'Northern', 'DR3': 'North Midland', 'DR4': 'South Midland',
                        'DR5': 'Southern', 'DR6': 'New York City', 'DR7': 'Western', 'DR8': 'Army Brat'}

            table = list()

            for allignment_file, audio_file in zip(
                sorted(Path(r'D:\voice_datasets\timit\TIMIT_2\data').glob('*/*/*/*.PHN')),
                sorted(Path(r'D:\voice_datasets\timit\TIMIT_2\data').glob('*/*/*/*.WAV.wav'))
                ):
                with open(allignment_file) as labels:
                    usage, dialect, dictor_id, filename = str(allignment_file).split(self.os_slash)[-4:]
                    for label in labels:
                        print(label)
                        label = label.split()
                        phone_name = label[2].upper()
                        start = round(int(label[0]) / self.timit_constant, 3)
                        end = round(int(label[1]) / self.timit_constant, 3)

                        table.append([
                            phone_name,                         # ARPABET code
                            self.phoneme_labeler[phone_name],   # class of phonemes
                            usage,                              # TEST or TRAIN
                            dictor_id,
                            dictor_id[0],
                            dialects[dialect],
                            '/'.join(map(str, [usage, dialect, dictor_id, filename])),
                            '/'.join(map(str, str(audio_file).split(self.os_slash)[-4:])),
                            start,
                            end
                        ])

            df = pd.DataFrame(data=table, columns=[
                'phone_name',
                'phone_class',
                'usage',
                'speaker_id',
                'speaker_gender',
                'dialect',
                'allignment_file_path',
                'audio_file_path',
                't0',
                't1']
                )
            df.to_csv('timit_description.csv', index=False)

            return df

    def _filter_description_table(self,
                                  usage: str,
                                  phone_classes: Optional[List[str]],
                                  percentage: Optional[float],
                                  gender: Optional[str],
                                  dialect: Optional[List[str]]) -> pd.DataFrame:
        self.description_table = self.description_table.loc[self.description_table['usage'] == usage]


        if percentage is not None:
            self.description_table = self.description_table.sample(frac=percentage)

        if phone_classes is not None:
            self.description_table = self.description_table.loc[self.description_table['phone_class'].isin(phone_classes)]

        if gender is not None:
            self.description_table = self.description_table.loc[self.description_table['gender'] == gender]

        if dialect is not None:
            dialects = self.description_table['dialect'].isin(dialect)
            self.description_table = self.description_table[dialects]

        return self.description_table

    def _get_audio_fragments(self, *args, **kwargs) -> list[AudioData]:
        fragments = list()
        for _, row in self.description_table.iterrows():
            fragments.extend(self._load_audio_fragment(row, self.root_dir))
        return fragments

    def __len__(self) -> int:
        return len(self.description_table)

    def __getitem__(self, item: int) -> AudioData:
        if self.transform:
            audio_data = self.audio_fragments[item]
            audio_data.data = self.transform(audio_data.data)
            return audio_data
        return self.audio_fragments[item]