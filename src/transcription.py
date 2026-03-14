import whisper
import warnings
import os
from src.utils import setup_ffmpeg

warnings.filterwarnings("ignore")

# Ensure FFmpeg is available
setup_ffmpeg()

class AudioTranscriber:
    def __init__(self, model_name="tiny.en"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, file_path):
        result = self.model.transcribe(file_path, word_timestamps=True)
        return {
            "text": result["text"].strip(),
            "segments": result["segments"]
        }
