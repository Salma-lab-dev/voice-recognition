"""
Configuration for diarization system
Tune these parameters to improve accuracy
"""

class DiarizationConfig:
    """
    Configuration for speaker diarization
    """
    
    # Voice Activity Detection (VAD)
    VAD_MIN_SEGMENT_DURATION = 0.3  # Minimum segment length (seconds)
    VAD_ENERGY_PERCENTILE = 20      # Lower = more sensitive to quiet speech
    VAD_TOP_DB = 20                 # Silence threshold (higher = more aggressive)
    
    # Clustering
    CLUSTERING_METHOD = 'ward'       # 'ward', 'average', 'complete'
    CLUSTERING_METRIC = 'euclidean'  # For non-ward methods
    
    # Auto-detection of speakers
    MIN_SPEAKERS = 2
    MAX_SPEAKERS = 8
    SILHOUETTE_THRESHOLD = 0.15      # Minimum silhouette score to accept
    
    # Speaker identification
    USE_CONSENSUS_VOTING = True      # Use majority vote per cluster
    MIN_CONFIDENCE = 0.3             # Minimum confidence to accept prediction
    
    # Post-processing
    MERGE_GAP_THRESHOLD = 0.5        # Merge segments closer than this (seconds)
    MIN_CLUSTER_SIZE = 2             # Minimum segments per speaker
    
    # Feature extraction
    N_MFCC = 20
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # Model thresholds
    PROBABILITY_THRESHOLD = 0.5      # Minimum probability for speaker ID


class OptimizedDiarizationConfig(DiarizationConfig):
    """
    Optimized settings for better accuracy
    """
    
    # More conservative VAD
    VAD_MIN_SEGMENT_DURATION = 0.5   # Longer minimum
    VAD_ENERGY_PERCENTILE = 25       # Less sensitive
    VAD_TOP_DB = 25                  # Higher threshold
    
    # Stricter clustering
    SILHOUETTE_THRESHOLD = 0.20
    MIN_CLUSTER_SIZE = 3
    
    # Higher confidence requirements
    MIN_CONFIDENCE = 0.4
    PROBABILITY_THRESHOLD = 0.6


class ConservativeDiarizationConfig(DiarizationConfig):
    """
    Conservative settings - tends to detect fewer speakers
    """
    
    VAD_MIN_SEGMENT_DURATION = 0.7
    VAD_ENERGY_PERCENTILE = 30
    VAD_TOP_DB = 30
    
    MAX_SPEAKERS = 5
    SILHOUETTE_THRESHOLD = 0.25
    MIN_CLUSTER_SIZE = 4
    
    MERGE_GAP_THRESHOLD = 1.0


class AggressiveDiarizationConfig(DiarizationConfig):
    """
    Aggressive settings - detects more speakers and shorter segments
    """
    
    VAD_MIN_SEGMENT_DURATION = 0.2
    VAD_ENERGY_PERCENTILE = 15
    VAD_TOP_DB = 15
    
    MAX_SPEAKERS = 10
    SILHOUETTE_THRESHOLD = 0.10
    MIN_CLUSTER_SIZE = 1
    
    MERGE_GAP_THRESHOLD = 0.3


# Default configuration
DEFAULT_CONFIG = OptimizedDiarizationConfig()


def get_config(profile: str = 'optimized'):
    """
    Get configuration profile
    
    Args:
        profile: 'default', 'optimized', 'conservative', 'aggressive'
    
    Returns:
        Configuration object
    """
    profiles = {
        'default': DiarizationConfig(),
        'optimized': OptimizedDiarizationConfig(),
        'conservative': ConservativeDiarizationConfig(),
        'aggressive': AggressiveDiarizationConfig()
    }
    
    return profiles.get(profile, DEFAULT_CONFIG)