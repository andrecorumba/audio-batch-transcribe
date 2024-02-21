# audio-batch-transcribe

`audio-batch-transcribe` is a Python tool designed for batch transcription of audio files using state-of-the-art speech recognition models. This project is especially useful for processing large datasets of audio files efficiently.

## Features

- Utilizes the `transformers` and `torch` libraries for leveraging powerful speech recognition models like `openai/whisper-large-v3`.
- Includes functionality for handling audio files, processing them in batches, and storing transcriptions in a SQLite database for easy access and management.
- Supports various audio formats, with a focus on 'opus' files, providing a streamlined process from audio file input to text transcription output.

## Installation

Ensure you have Python 3.12 or newer. Clone this repository, and install dependencies using Poetry:

```shell
poetry install
```

## Usage

1. Place your audio files in the `audios` folder.
2. Run the script to start the transcription process. Transcriptions and any errors encountered will be stored in a SQLite database named `audios_transcriptions.db`.

```shell
python transcribe.py
```

## Dependencies

- Python ^3.12
- pandas ^2.2.0
- transformers ^4.37.2
- accelerate ^0.27.2
- torch ^2.2.0
- datasets ^2.17.1
- librosa ^0.10.1
- soundfile ^0.12.1
- setuptools ^69.1.0

## Authors

Andre Rocha - @andrecorumba

## License

This project is licensed under the MIT License.