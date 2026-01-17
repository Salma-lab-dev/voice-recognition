"""
Module de visualisations pour l'application
"""

import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour Flask
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import os
from datetime import datetime

sns.set_style("whitegrid")


def create_timeline_plot(segments: List[Dict], output_dir: str = 'static/results') -> str:
    """
    Crée une timeline de la diarisation
    """
    if not segments:
        return None
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Extraire speakers uniques
    speakers = sorted(set(s.get('speaker_id', 'UNKNOWN') for s in segments))
    speaker_colors = plt.cm.tab10(np.linspace(0, 1, len(speakers)))
    color_map = {speaker: color for speaker, color in zip(speakers, speaker_colors)}
    
    # Tracer chaque segment
    for seg in segments:
        speaker = seg.get('speaker_id', 'UNKNOWN')
        start = seg['start']
        end = seg['end']
        duration = end - start
        
        y_pos = speakers.index(speaker)
        
        ax.barh(
            y_pos,
            duration,
            left=start,
            height=0.8,
            color=color_map[speaker],
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Ajouter confidence si disponible
        if 'confidence' in seg and seg['confidence'] > 0:
            mid_point = start + duration / 2
            ax.text(
                mid_point, y_pos,
                f"{seg['confidence']:.0%}",
                ha='center', va='center',
                fontsize=8, fontweight='bold'
            )
    
    # Configuration
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speaker', fontsize=12, fontweight='bold')
    ax.set_title('Diarization Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map[speaker], label=speaker, alpha=0.7)
        for speaker in speakers
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Sauvegarder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"timeline_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_sentiment_chart(segments: List[Dict], output_dir: str = 'static/results') -> str:
    """
    Crée un graphique de sentiment au fil du temps
    """
    if not segments or 'sentiment' not in segments[0]:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Extraire données
    times = [(seg['start'] + seg['end']) / 2 for seg in segments]
    sentiments = [seg.get('sentiment', 'neutral') for seg in segments]
    emotions = [seg.get('emotion', 'neutral') for seg in segments]
    speakers = [seg.get('speaker_id', 'UNKNOWN') for seg in segments]
    
    # Mapping sentiment to numeric value
    sentiment_values = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }
    sentiment_nums = [sentiment_values.get(s, 0) for s in sentiments]
    
    # 1. Sentiment over time
    unique_speakers = sorted(set(speakers))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_speakers)))
    color_map = {speaker: color for speaker, color in zip(unique_speakers, colors)}
    
    for speaker in unique_speakers:
        speaker_times = [t for t, sp in zip(times, speakers) if sp == speaker]
        speaker_sentiments = [s for s, sp in zip(sentiment_nums, speakers) if sp == speaker]
        
        ax1.scatter(
            speaker_times,
            speaker_sentiments,
            c=[color_map[speaker]],
            s=100,
            alpha=0.6,
            label=speaker,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Trend line overall
    if len(times) > 1:
        z = np.polyfit(times, sentiment_nums, 2)
        p = np.poly1d(z)
        time_smooth = np.linspace(min(times), max(times), 100)
        ax1.plot(time_smooth, p(time_smooth), 'r--', alpha=0.5, linewidth=2, label='Trend')
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sentiment', fontsize=12, fontweight='bold')
    ax1.set_title('Sentiment Evolution', fontsize=14, fontweight='bold')
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(['Negative', 'Neutral', 'Positive'])
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Emotion distribution
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    emotions_sorted = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    emotion_names = [e[0] for e in emotions_sorted]
    emotion_vals = [e[1] for e in emotions_sorted]
    
    bars = ax2.bar(
        emotion_names,
        emotion_vals,
        color=plt.cm.Pastel1(np.linspace(0, 1, len(emotion_names))),
        edgecolor='black',
        linewidth=1
    )
    
    # Ajouter valeurs sur barres
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center', va='bottom',
            fontweight='bold'
        )
    
    ax2.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Emotion Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Sauvegarder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentiment_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_statistics_chart(speaker_stats: Dict, output_dir: str = 'static/results') -> str:
    """
    Crée un graphique des statistiques par locuteur
    """
    if not speaker_stats:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    speakers = list(speaker_stats.keys())
    
    # 1. Speaking time
    total_times = [speaker_stats[sp]['total_time'] for sp in speakers]
    
    axes[0, 0].barh(
        speakers,
        total_times,
        color=plt.cm.Blues(np.linspace(0.4, 0.8, len(speakers))),
        edgecolor='black'
    )
    axes[0, 0].set_xlabel('Time (seconds)', fontweight='bold')
    axes[0, 0].set_title('Speaking Time per Speaker', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Add values
    for i, (speaker, time) in enumerate(zip(speakers, total_times)):
        axes[0, 0].text(time, i, f' {time:.1f}s', va='center', fontweight='bold')
    
    # 2. Time percentage
    percentages = [speaker_stats[sp].get('percentage', 0) for sp in speakers]
    
    axes[0, 1].pie(
        percentages,
        labels=speakers,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Set3(np.linspace(0, 1, len(speakers)))
    )
    axes[0, 1].set_title('Speaking Time Distribution', fontsize=14, fontweight='bold')
    
    # 3. Number of segments
    num_segments = [speaker_stats[sp]['num_segments'] for sp in speakers]
    
    bars = axes[1, 0].bar(
        speakers,
        num_segments,
        color=plt.cm.Greens(np.linspace(0.4, 0.8, len(speakers))),
        edgecolor='black'
    )
    axes[1, 0].set_ylabel('Number of Segments', fontweight='bold')
    axes[1, 0].set_title('Number of Interventions', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center', va='bottom',
            fontweight='bold'
        )
    
    # 4. Average confidence
    if 'confidence' in speaker_stats[speakers[0]]:
        confidences = [speaker_stats[sp].get('confidence', 0) * 100 for sp in speakers]
        
        axes[1, 1].barh(
            speakers,
            confidences,
            color=plt.cm.Oranges(np.linspace(0.4, 0.8, len(speakers))),
            edgecolor='black'
        )
        axes[1, 1].set_xlabel('Confidence (%)', fontweight='bold')
        axes[1, 1].set_title('Average Identification Confidence', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlim([0, 100])
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        for i, (speaker, conf) in enumerate(zip(speakers, confidences)):
            axes[1, 1].text(conf, i, f' {conf:.1f}%', va='center', fontweight='bold')
    else:
        axes[1, 1].axis('off')
        axes[1, 1].text(
            0.5, 0.5, 'Confidence not available',
            ha='center', va='center',
            fontsize=12
        )
    
    plt.tight_layout()
    
    # Sauvegarder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"statistics_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename