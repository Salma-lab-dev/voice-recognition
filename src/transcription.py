"""
Module de transcription automatique avec Whisper
"""

import numpy as np
import whisper
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class AudioTranscriber:
    """
    Transcription automatique avec OpenAI Whisper
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large'
        """
        print(f"🔄 Chargement du modèle Whisper ({model_size})...")
        self.model = whisper.load_model(model_size, device='cpu')
        print("✅ Modèle chargé!")
    
    def transcribe(self, audio_path: str, language: str = "en") -> Dict:
        """
        Transcrit un fichier audio
        
        Returns:
            Dict avec texte, segments, et langue détectée
        """
        print(f"🎤 Transcription de: {audio_path}")
        
        # Transcrire
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            verbose=False
        )
        
        return {
            'text': result['text'],
            'segments': result['segments'],
            'language': result['language']
        }
    
    def transcribe_with_timestamps(self, audio_path: str, 
                                   language: str = "en") -> List[Dict]:
        """
        Transcription avec timestamps pour chaque segment
        
        Returns:
            Liste de segments avec start, end, texte
        """
        result = self.transcribe(audio_path, language)
        
        segments_with_time = []
        for seg in result['segments']:
            segments_with_time.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip(),
                'confidence': seg.get('no_speech_prob', 0)
            })
        
        return segments_with_time
    
    def align_transcription_with_diarization(self, 
                                            transcription_segments: List[Dict],
                                            diarization_segments: List[Dict]) -> List[Dict]:
        """
        Aligne transcription avec diarisation
        
        Args:
            transcription_segments: Segments de transcription avec timestamps
            diarization_segments: Segments de diarisation avec speaker_id
        
        Returns:
            Segments combinés avec texte et speaker_id
        """
        aligned_segments = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg['start']
            trans_end = trans_seg['end']
            trans_text = trans_seg['text']
            
            # Trouver le locuteur correspondant (overlap maximal)
            best_speaker = None
            max_overlap = 0
            
            for diar_seg in diarization_segments:
                diar_start = diar_seg['start']
                diar_end = diar_seg['end']
                
                # Calculer overlap
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = diar_seg.get('speaker_id', 'UNKNOWN')
            
            aligned_segments.append({
                'start': trans_start,
                'end': trans_end,
                'text': trans_text,
                'speaker_id': best_speaker if best_speaker else 'UNKNOWN',
                'duration': trans_end - trans_start
            })
        
        return aligned_segments
    
    def format_transcript(self, segments: List[Dict]) -> str:
        """
        Formate la transcription pour affichage
        
        Returns:
            Texte formaté avec speakers et timestamps
        """
        formatted = []
        
        for seg in segments:
            speaker = seg.get('speaker_id', 'UNKNOWN')
            start = seg['start']
            end = seg['end']
            text = seg['text']
            
            time_str = f"[{start:.2f}s - {end:.2f}s]"
            formatted.append(f"{speaker} {time_str}: {text}")
        
        return "\n".join(formatted)