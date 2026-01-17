"""
Create test audio with LONGER segments per speaker
This ensures VAD doesn't split into tiny pieces
"""

import os
import numpy as np
import librosa
import soundfile as sf
import random
from typing import List, Tuple


def concatenate_speaker_files(speaker_id: str, 
                              librispeech_path: str = "LibriSpeech/dev-clean",
                              num_files: int = 5,
                              target_duration: float = 10.0) -> np.ndarray:
    """
    Concatenate multiple files from a speaker to create LONGER segments
    
    Args:
        speaker_id: Speaker ID
        num_files: Number of files to concatenate
        target_duration: Target duration in seconds
    
    Returns:
        Concatenated audio array
    """
    speaker_path = os.path.join(librispeech_path, speaker_id)
    
    # Get all audio files
    audio_files = []
    for root, dirs, files in os.walk(speaker_path):
        for file in files:
            if file.endswith('.flac'):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        raise ValueError(f"No files found for speaker {speaker_id}")
    
    # Randomly select files
    random.shuffle(audio_files)
    selected_files = audio_files[:num_files]
    
    # Load and concatenate
    segments = []
    total_duration = 0
    
    for file_path in selected_files:
        audio, sr = librosa.load(file_path, sr=16000)
        segments.append(audio)
        total_duration += len(audio) / sr
        
        if total_duration >= target_duration:
            break
    
    # Concatenate with small gaps
    result = []
    gap = np.zeros(int(0.1 * 16000))  # 100ms gap between files
    
    for i, segment in enumerate(segments):
        result.append(segment)
        if i < len(segments) - 1:
            result.append(gap)
    
    return np.concatenate(result)


def create_long_segment_conversation(speaker_ids: List[str],
                                     output_path: str,
                                     segments_per_speaker: int = 3,
                                     segment_duration: float = 8.0,
                                     pause_between: float = 1.0):
    """
    Create conversation where each speaker talks for 8+ seconds
    
    Args:
        speaker_ids: List of speaker IDs
        output_path: Output file path
        segments_per_speaker: How many times each speaker talks
        segment_duration: Target duration per segment (seconds)
        pause_between: Pause between speakers (seconds)
    """
    print(f"\n{'='*60}")
    print(f"Creating LONG SEGMENT conversation")
    print(f"{'='*60}")
    print(f"Speakers: {speaker_ids}")
    print(f"Segments per speaker: {segments_per_speaker}")
    print(f"Target duration per segment: {segment_duration}s")
    print()
    
    conversation = []
    metadata = {
        'speakers': speaker_ids,
        'segments': [],
        'total_duration': 0
    }
    
    current_time = 0.0
    pause_samples = int(pause_between * 16000)
    
    # Alternate speakers
    for turn in range(segments_per_speaker):
        for speaker_id in speaker_ids:
            print(f"Creating segment {turn+1}/{segments_per_speaker} for speaker {speaker_id}...")
            
            # Create long segment by concatenating multiple files
            segment = concatenate_speaker_files(
                speaker_id,
                num_files=5,  # Use 5 files to get ~8s
                target_duration=segment_duration
            )
            
            segment_duration_actual = len(segment) / 16000
            
            # Add to conversation
            conversation.append(segment)
            
            # Add metadata
            metadata['segments'].append({
                'speaker_id': speaker_id,
                'start': current_time,
                'end': current_time + segment_duration_actual,
                'duration': segment_duration_actual
            })
            
            print(f"  ✅ Added {segment_duration_actual:.2f}s segment at {current_time:.2f}s")
            
            current_time += segment_duration_actual
            
            # Add pause
            if pause_samples > 0:
                conversation.append(np.zeros(pause_samples))
                current_time += pause_between
    
    # Concatenate all
    final_audio = np.concatenate(conversation)
    metadata['total_duration'] = len(final_audio) / 16000
    
    # Save audio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, final_audio, 16000)
    
    # Save metadata
    import json
    metadata_path = output_path.replace('.wav', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS!")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Total duration: {metadata['total_duration']:.2f}s")
    print(f"Total segments: {len(metadata['segments'])}")
    print(f"\nSegment breakdown:")
    for seg in metadata['segments']:
        print(f"  {seg['speaker_id']}: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)")
    
    return final_audio, metadata


if __name__ == "__main__":
    # Create test audio with LONG segments
    audio, metadata = create_long_segment_conversation(
        speaker_ids=['1462', '1272'],
        output_path='static/uploads/test_audios/conversation_LONG_SEGMENTS.wav',
        segments_per_speaker=3,
        segment_duration=8.0,
        pause_between=1.0
    )
    
    print("\n" + "="*60)
    print("Now test with:")
    print("python debug_diarization.py static/uploads/test_audios/conversation_LONG_SEGMENTS.wav 2")
    print("="*60)