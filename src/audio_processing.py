import librosa
import numpy as np
import warnings
import os
from src.utils import setup_ffmpeg

warnings.filterwarnings("ignore")

# Ensure FFmpeg is available
setup_ffmpeg()

class AudioAnalyzer:
    def __init__(self, sr=16000):
        self.sr = sr

    def analyze(self, file_path):
        # Enforcing sr=16000 for strict memory constraints
        y, sr = librosa.load(file_path, sr=self.sr)
        
        # Find active speech intervals
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        
        active_samples = sum([end - start for start, end in non_silent_intervals])
        active_duration = active_samples / sr
        
        # Calculate pauses > 1.0 seconds
        pauses = []
        for i in range(1, len(non_silent_intervals)):
            prev_end = non_silent_intervals[i-1][1]
            curr_start = non_silent_intervals[i][0]
            silence_dur = (curr_start - prev_end) / sr
            if silence_dur > 1.0:
                pauses.append(silence_dur)
                
        # Pitch variation for engagement
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_variation = np.nanstd(f0) if np.any(~np.isnan(f0)) else 0
        
        return {
            "active_duration": active_duration,
            "pause_count": len(pauses),
            "pitch_variation": float(pitch_variation)
        }
