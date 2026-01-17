"""
Extraction de features avancée
"""

import numpy as np
import librosa
from typing import Dict, Tuple


class AdvancedFeatureExtractor:
    """Extraction de features vocales complètes"""
    
    def __init__(self, n_mfcc: int = 20, n_mels: int = 128):
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
    
    def extract_mfcc_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        MFCC avec deltas et delta-deltas
        """
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Combiner
        features = np.vstack([mfcc, delta, delta2])
        return features
    
    def extract_mel_spectrogram(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Mel-spectrogramme (utile pour deep learning)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Features spectrales complètes
        """
        features = {}
        
        # Centroïde spectral
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # Bande passante
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        # Contraste spectral
        features['spectral_contrast'] = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Roll-off
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Zero crossing rate
        features['zcr'] = librosa.feature.zero_crossing_rate(y)
        
        # Flatness
        features['spectral_flatness'] = librosa.feature.spectral_flatness(y=y)
        
        return features
    
    def extract_pitch_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Features de pitch (fondamental pour le timbre vocal)
        """
        features = {}
        
        # F0 (fréquence fondamentale) via autocorrélation
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        features['f0'] = f0.reshape(1, -1)
        
        # Chroma features (contenu tonal)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features['chroma'] = chroma
        
        return features
    
    def extract_prosody_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Features prosodiques (rythme, énergie)
        """
        features = {}
        
        # Énergie RMS
        rms = librosa.feature.rms(y=y)
        features['rms_energy'] = rms
        
        # Tempogram (rythme)
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        features['tempogram'] = tempogram
        
        return features
    
    def aggregate_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Agrégation statistique avancée
        
        Pour chaque feature temporelle, calcule:
        - Moyenne
        - Écart-type
        - Médiane
        - Min/Max
        - Percentiles (25%, 75%)
        """
        aggregated = []
        
        for key, feature_array in features_dict.items():
            # S'assurer que c'est 2D
            if feature_array.ndim == 1:
                feature_array = feature_array.reshape(1, -1)
            
            # Statistiques par feature
            mean = np.mean(feature_array, axis=1)
            std = np.std(feature_array, axis=1)
            median = np.median(feature_array, axis=1)
            min_val = np.min(feature_array, axis=1)
            max_val = np.max(feature_array, axis=1)
            q25 = np.percentile(feature_array, 25, axis=1)
            q75 = np.percentile(feature_array, 75, axis=1)
            
            # Concaténer
            stats = np.concatenate([mean, std, median, min_val, max_val, q25, q75])
            aggregated.append(stats)
        
        return np.concatenate(aggregated)
    
    def extract_complete_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Pipeline complet d'extraction
        """
        all_features = {}
        
        # MFCC
        mfcc_features = self.extract_mfcc_features(y, sr)
        all_features['mfcc'] = mfcc_features
        
        # Spectral
        spectral = self.extract_spectral_features(y, sr)
        all_features.update(spectral)
        
        # Pitch
        pitch = self.extract_pitch_features(y, sr)
        all_features.update(pitch)
        
        # Prosody
        prosody = self.extract_prosody_features(y, sr)
        all_features.update(prosody)
        
        # Agréger
        feature_vector = self.aggregate_features(all_features)
        
        return feature_vector