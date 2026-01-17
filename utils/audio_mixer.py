"""
Utilitaire pour mélanger des audios LibriSpeech
"""

import os
import random
import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple


class AudioMixer:
    """
    Crée des fichiers audio mixés avec plusieurs locuteurs
    pour tester la diarisation
    """
    
    def __init__(self, librispeech_path: str = "LibriSpeech/dev-clean"):
        self.librispeech_path = librispeech_path
    
    def get_speaker_files(self, speaker_id: str, max_files: int = 5) -> List[str]:
        """Récupère les fichiers d'un locuteur"""
        speaker_path = os.path.join(self.librispeech_path, speaker_id)
        
        if not os.path.exists(speaker_path):
            return []
        
        audio_files = []
        
        for chapter in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter)
            
            if not os.path.isdir(chapter_path):
                continue
            
            for file in os.listdir(chapter_path):
                if file.endswith('.flac'):
                    audio_files.append(os.path.join(chapter_path, file))
                    
                    if len(audio_files) >= max_files:
                        return audio_files
        
        return audio_files
    
    def create_mixed_conversation(self, 
                                 speaker_ids: List[str],
                                 output_path: str,
                                 files_per_speaker: int = 3,
                                 gap_duration: float = 0.5,
                                 sr: int = 16000) -> Tuple[np.ndarray, List[dict]]:
        """
        Crée une conversation mixée alternant entre locuteurs
        
        Args:
            speaker_ids: Liste des IDs de locuteurs
            output_path: Chemin de sortie
            files_per_speaker: Nombre de fichiers par locuteur
            gap_duration: Durée du silence entre segments (secondes)
            sr: Sample rate
        
        Returns:
            (audio_mixé, segments_info)
        """
        print(f"🎙️ Création conversation mixée avec {len(speaker_ids)} locuteurs...")
        
        # Collecter les fichiers
        speaker_files = {}
        for speaker_id in speaker_ids:
            files = self.get_speaker_files(speaker_id, files_per_speaker)
            if files:
                speaker_files[speaker_id] = files
                print(f"  {speaker_id}: {len(files)} fichiers")
        
        if not speaker_files:
            raise ValueError("Aucun fichier audio trouvé!")
        
        # Créer la séquence alternée
        mixed_audio = []
        segments_info = []
        current_time = 0.0
        
        gap_samples = int(gap_duration * sr)
        gap_audio = np.zeros(gap_samples)
        
        # Alterner entre locuteurs
        max_turns = max(len(files) for files in speaker_files.values())
        
        for turn in range(max_turns):
            for speaker_id in speaker_ids:
                if speaker_id not in speaker_files:
                    continue
                
                files = speaker_files[speaker_id]
                if turn >= len(files):
                    continue
                
                # Charger segment
                file_path = files[turn]
                y, _ = librosa.load(file_path, sr=sr)
                
                # Ajouter gap avant
                if len(mixed_audio) > 0:
                    mixed_audio.append(gap_audio)
                    current_time += gap_duration
                
                # Info du segment
                segment_start = current_time
                segment_duration = len(y) / sr
                segment_end = segment_start + segment_duration
                
                segments_info.append({
                    'speaker_id': speaker_id,
                    'start': segment_start,
                    'end': segment_end,
                    'duration': segment_duration,
                    'file': os.path.basename(file_path)
                })
                
                # Ajouter audio
                mixed_audio.append(y)
                current_time = segment_end
        
        # Concaténer tout
        final_audio = np.concatenate(mixed_audio)
        
        # Sauvegarder
        sf.write(output_path, final_audio, sr)
        print(f"✅ Audio mixé sauvegardé: {output_path}")
        print(f"   Durée totale: {len(final_audio)/sr:.2f}s")
        print(f"   Segments: {len(segments_info)}")
        
        return final_audio, segments_info
    
    def create_overlapping_conversation(self,
                                       speaker_ids: List[str],
                                       output_path: str,
                                       duration: float = 30.0,
                                       sr: int = 16000) -> Tuple[np.ndarray, List[dict]]:
        """
        Crée une conversation avec segments qui se chevauchent
        (plus réaliste mais plus difficile)
        """
        print(f"🎙️ Création conversation avec overlaps...")
        
        # Collecter fichiers
        speaker_files = {}
        for speaker_id in speaker_ids:
            files = self.get_speaker_files(speaker_id, max_files=10)
            if files:
                speaker_files[speaker_id] = files
        
        # Créer canvas audio
        total_samples = int(duration * sr)
        mixed_audio = np.zeros(total_samples)
        
        segments_info = []
        
        for speaker_id, files in speaker_files.items():
            # Nombre de segments pour ce locuteur
            num_segments = random.randint(2, 4)
            
            for i in range(num_segments):
                if not files:
                    break
                
                # Choisir un fichier
                file_path = random.choice(files)
                y, _ = librosa.load(file_path, sr=sr)
                
                # Position aléatoire
                max_start = max(0, total_samples - len(y))
                if max_start <= 0:
                    continue
                
                start_sample = random.randint(0, max_start)
                end_sample = start_sample + len(y)
                
                # Ajouter avec fade
                segment_audio = y.copy()
                fade_len = int(0.1 * sr)  # 100ms fade
                
                if len(segment_audio) > fade_len:
                    # Fade in
                    segment_audio[:fade_len] *= np.linspace(0, 1, fade_len)
                    # Fade out
                    segment_audio[-fade_len:] *= np.linspace(1, 0, fade_len)
                
                # Mixer
                mixed_audio[start_sample:end_sample] += segment_audio * 0.7
                
                # Info
                segments_info.append({
                    'speaker_id': speaker_id,
                    'start': start_sample / sr,
                    'end': end_sample / sr,
                    'duration': len(y) / sr
                })
        
        # Normaliser
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
        
        # Sauvegarder
        sf.write(output_path, mixed_audio, sr)
        print(f"✅ Audio avec overlaps sauvegardé: {output_path}")
        
        return mixed_audio, segments_info


def create_test_audios(output_dir: str = "static/uploads/test_audios"):
    """
    Crée quelques fichiers de test
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mixer = AudioMixer()
    
    # Liste de locuteurs disponibles
    librispeech_path = "LibriSpeech/dev-clean"
    if os.path.exists(librispeech_path):
        speakers = sorted(os.listdir(librispeech_path))[:5]
    else:
        print("⚠️ LibriSpeech non trouvé!")
        return
    
    # Test 1: Conversation simple (2 locuteurs)
    try:
        mixer.create_mixed_conversation(
            speaker_ids=speakers[:2],
            output_path=os.path.join(output_dir, "conversation_2speakers.wav"),
            files_per_speaker=3
        )
    except Exception as e:
        print(f"Erreur test 1: {e}")
    
    # Test 2: Conversation complexe (3 locuteurs)
    try:
        mixer.create_mixed_conversation(
            speaker_ids=speakers[:3],
            output_path=os.path.join(output_dir, "conversation_3speakers.wav"),
            files_per_speaker=2
        )
    except Exception as e:
        print(f"Erreur test 2: {e}")
    
    print("✅ Fichiers de test créés!")


if __name__ == "__main__":
    create_test_audios()