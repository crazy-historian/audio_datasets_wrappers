import re
import json
import tqdm
import pandas as pd
import platform
import textgrid
import torchaudio
import torch.nn.functional as F

from pathlib import Path
from dataset import PhonemeLabeler, PhonemeData

TIMIT_CONSTANT = 15987

COLUMNS = [
    'phone_name',
    'phone_class',
    'class_index',
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
                phoneme_name = label[2].upper()
                start = round(int(label[0]) / TIMIT_CONSTANT, 3)
                end = round(int(label[1]) / TIMIT_CONSTANT, 3)

                data.append([
                    phoneme_name,                         # ARPABET code
                    phoneme_labeler[phoneme_name],   # class of phonemes
                    phoneme_labeler.get_index_of_phoneme(
                        phoneme_labeler[phoneme_name]
                        ),
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
                    phoneme_labeler.get_index_of_phoneme(
                        phoneme_labeler[phoneme_name]
                        ),
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
                        phoneme_labeler.get_index_of_phoneme(
                            phoneme_labeler[interval.mark]
                        ),
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



def get_audio_data(
        desc_table: pd.DataFrame,
        dir_path: str,
        overlapping_frames: bool = True,
        frame_length: int | None = 1024,
        padding_length: int | None = None
    ) -> list[PhonemeData]:
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
            audio_data.append(PhonemeData(
                data=data,
                label=row.phone_class,
                label_index=row.class_index,
                frame_rate=frame_rate,
                sample_width=sample_width
            ))
        elif padding_length is not None:
            new_shape = padding_length - data.shape[1]
            data = F.pad(data, (0, new_shape), 'constant', 0.0)
            audio_data.append(PhonemeData(
                data=data,
                label=row.phone_class,
                label_index=row.class_index,
                frame_rate=frame_rate,
                sample_width=sample_width
            ))
        else:
            i = 0
            frames = list()
            for _ in range(data.shape[1] // (frame_length // 2) - 1):
                frames.append(PhonemeData(
                    data=data[:, i: i + frame_length],
                    label=row.phone_class,
                    label_index=row.class_index,
                    frame_rate=frame_rate,
                    sample_width=sample_width 
                ))
                i += frame_length // 2
            else:
                new_shape = frame_length - data[:, i:].shape[1]
                frames.append(PhonemeData(
                    data=F.pad(data[:, i:], (0, new_shape), 'constant', 0.0),
                    label=row.phone_class,
                    label_index=row.class_index,
                    frame_rate=frame_rate,
                    sample_width=sample_width 
                ))
            audio_data.extend(frames)

    return audio_data