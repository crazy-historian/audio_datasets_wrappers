from pathlib import Path
from setuptools import setup

setup(
    name='audio_datasets_wrappers',
    description='custom wrappers for audio datasets',
    version='0.0.2',
    license='',
    url='https://github.com/crazy-historian/audio_datasets_wrappers',
    author='Maxim Zaitsev',
    author_email='zaitsev808@mail.ru',

    packages=['audio_datasets_wrappers'],
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchaudio',
        'textgrid'
    ]
)