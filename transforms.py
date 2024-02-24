import re
import json
import pandas as pd
import platform
import textgrid

from pathlib import Path
from dataset import PhonemeLabeler

TIMIT_CONSTANT = 15987

COLUMNS = [
    'phone_name',
    'phone_class',
    'usage',
    'speaker_id',
    'gender',
    'dialect',
    'allignment_file_path',
    'audio_file_path',
    't0',
    't1'
]

with open(file='timit_dialects.json', mode='r') as file:
    TIMIT_DIALECTS = json.load(file)

with open(file='arctic_speakers.json', mode='r') as file:
    ARCTIC_SPEAKERS = json.load(file)

def remove_digits(phoneme_name: str) -> str:
    return re.sub(r'[0-9]+', '', phoneme_name)

def create_timit_discription_table(dir_path: str, phoneme_labeler: PhonemeLabeler) -> pd.DataFrame:
    columns = [
        'phone_name',
        'phone_class',
        'usage',
        'speaker_id',
        'gender',
        'dialect',
        'allignment_file_path',
        'audio_file_path',
        't0',
        't1'
    ]
    data = list()

    slash = '\\' if platform.system() == 'Windows' else '/'

    for allignment_file, audio_file in zip(
        sorted(Path(dir_path).glob('*/*/*/*.PHN')),
        sorted(Path(dir_path).glob('*/*/*/*.WAV.wav'))
        ):
        with open(allignment_file) as labels:
            usage, dialect, dictor_id, filename = str(allignment_file).split(slash)[-4:]
            for label in labels:
                label = label.split()
                phone_name = label[2].upper()
                start = round(int(label[0]) / TIMIT_CONSTANT, 3)
                end = round(int(label[1]) / TIMIT_CONSTANT, 3)

                data.append([
                    phone_name,                         # ARPABET code
                    phoneme_labeler[phone_name],   # class of phonemes
                    usage,                              # TEST or TRAIN
                    dictor_id,
                    dictor_id[0],
                    TIMIT_DIALECTS[dialect],
                    '/'.join(map(str, [usage, dialect, dictor_id, filename])),
                    '/'.join(map(str, str(audio_file).split(slash)[-4:])),
                    start,
                    end
                ])
    
    return pd.DataFrame(data=data, columns=COLUMNS)

def create_arctic_discription_table(dir_path: str, phoneme_labeler: PhonemeLabeler) -> pd.DataFrame:

    data = list()

    slash = '\\' if platform.system() == 'Windows' else '/'

    for speaker_dir in Path(dir_path).iterdir():
        for textgrid_file, wav_file in zip(
                sorted(Path(speaker_dir, 'textgrid').iterdir()),
                sorted(Path(speaker_dir, 'wav').iterdir())
            ): 
            table_rows = list()
            for interval in textgrid.TextGrid.fromFile(textgrid_file)[1]:
                phoneme_name = remove_digits(interval.mark)
                table_rows.append([
                    phoneme_name,
                    phoneme_labeler[phoneme_name],
                    None,
                    speaker_dir.stem,
                    ARCTIC_SPEAKERS[speaker_dir.stem]['gender'],
                    ARCTIC_SPEAKERS[speaker_dir.stem]['country'],
                    '/'.join(map(str, str(textgrid_file).split(slash)[-3:])),
                    '/'.join(map(str, str(wav_file).split(slash)[-3:])),
                    interval.minTime,
                    interval.maxTime
                ])
                
            data.extend(table_rows)
    
        return pd.DataFrame(data=data, columns=COLUMNS)
    
def create_librispeech_description_table(dir_path: str, phoneme_labeler: PhonemeLabeler) -> pd.DataFrame:
    columns = [
        'phone_name',
        'phone_class',
        'usage',
        'speaker_id',
        'gender',
        'dialect',
        'allignment_file_path',
        'audio_file_path',
        't0',
        't1'
    ]
    data = list()

    slash = '\\' if platform.system() == 'Windows' else '/'

    for directory in Path(dir_path).iterdir():
        data_type, usage = str(directory.stem).split('-')
        for sub_directory in directory.iterdir():
            for textgrid_file, flac_file in zip(
                sorted(sub_directory.glob('*/*.TextGrid')),
                sorted(sub_directory.glob('*/*.flac'))
                ):
                table_rows = list()
                for interval in textgrid.TextGrid.fromFile(textgrid_file)[1]:
                    table_rows.append([
                        interval.mark,
                        phoneme_labeler[interval.mark],
                        usage, 
                        None, # speaker id (?)
                        None, # speaker sex (?),
                        None, # speaker dialect?
                        '/'.join(map(str, str(textgrid_file).split(slash)[-4:])),
                        '/'.join(map(str, str(flac_file).split(slash)[-4:])),
                        interval.minTime,
                        interval.maxTime
                    ])
                data.extend(table_rows)
    
    return pd.DataFrame(data=data, columns=COLUMNS)