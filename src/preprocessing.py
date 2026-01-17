"""
Module de prétraitement audio amélioré
"""

import numpy as np
import librosa
import noisereduce as nr
from typing import Tuple, Optional


class AudioPreprocessor:
    """Prétraitement audio robuste avec débruitage"""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    def load_audio(self, filepath: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Charge un fichier audio"""
        if sr is None:
            sr = self.target_sr
        y, sr = librosa.load(filepath, sr=sr)
        return y, sr
    
    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Normalisation RMS"""
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            return y * (0.1 / rms)
        return y
    
    def remove_silence(self, y: np.ndarray, sr: int, 
                      top_db: int = 20) -> Tuple[np.ndarray, tuple]:
        """Suppression des silences"""
        y_trimmed, index = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed, index
    
    def denoise(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Débruitage avec noisereduce"""
        try:
            return nr.reduce_noise(y=y, sr=sr, stationary=True)
        except:
            return y
    
    def preemphasis(self, y: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """Pré-accentuation pour amplifier hautes fréquences"""
        return np.append(y[0], y[1:] - coef * y[:-1])
    
    def augment_audio(self, y: np.ndarray, sr: int, 
                     aug_type: str = 'pitch_shift') -> np.ndarray:
        """Augmentation de données"""
        if aug_type == 'pitch_shift':
            return librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        elif aug_type == 'time_stretch':
            return librosa.effects.time_stretch(y, rate=1.1)
        elif aug_type == 'noise':
            noise = np.random.randn(len(y)) * 0.005
            return y + noise
        return y
    
    def preprocess_pipeline(self, filepath: str, 
                           denoise: bool = True,
                           preemphasis: bool = True) -> Tuple[np.ndarray, int]:
        """Pipeline complet de prétraitement"""
        # Charger
        y, sr = self.load_audio(filepath)
        
        # Normaliser
        y = self.normalize_audio(y)
        
        # Supprimer silence
        y, _ = self.remove_silence(y, sr)
        
        # Débruiter (optionnel)
        if denoise:
            y = self.denoise(y, sr)
        
        # Pré-accentuation (optionnel)
        if preemphasis:
            y = self.preemphasis(y)
        
        return y, sr