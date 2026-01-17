"""
Application Flask - Système d'Analyse Vocale Avancé
CORRECTION: Meilleur chargement du modèle
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import numpy as np
from datetime import datetime
import traceback

# Imports des modules personnalisés
from src.preprocessing import AudioPreprocessor
from src.feature_extraction import AdvancedFeatureExtractor
from src.model_training import EnsembleSpeakerRecognition
from src.diarization import AutoSpeakerDiarizer
from src.transcription import AudioTranscriber
from src.sentiment_analysis import SentimentAnalyzer
from utils.visualizations import create_timeline_plot, create_sentiment_chart, create_statistics_chart

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Créer dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# ✅ FIX: Meilleur chargement du modèle
def load_speaker_model():
    """Charge le modèle avec gestion d'erreurs améliorée"""
    model_path = 'models/speaker_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ ERREUR: Modèle non trouvé à {model_path}")
        print("\n💡 SOLUTION:")
        print("   1. Assurez-vous d'avoir entraîné le modèle:")
        print("      python train.py")
        print("   2. Vérifiez que le fichier models/speaker_model.pkl existe")
        print("\n⚠️ L'application fonctionnera mais utilisera des clusters génériques\n")
        return None
    
    try:
        print(f"🔄 Chargement du modèle depuis {model_path}...")
        
        # ✅ Utiliser la méthode statique qui retourne l'instance
        model = EnsembleSpeakerRecognition.load_model(model_path)
        
        # Vérifier que le modèle est valide
        if model is None:
            raise ValueError("Le modèle chargé est None")
        
        if not hasattr(model, 'model') or model.model is None:
            raise ValueError("Le modèle n'a pas d'attribut 'model' valide")
        
        print("✅ Modèle chargé avec succès!")
        
        # Afficher des infos sur le modèle
        if hasattr(model, 'label_encoder'):
            classes = model.label_encoder.classes_
            print(f"   Locuteurs entraînés: {list(classes)}")
            print(f"   Nombre de locuteurs: {len(classes)}")
        
        if hasattr(model, 'ensemble_type'):
            print(f"   Type d'ensemble: {model.ensemble_type}")
        
        return model
        
    except Exception as e:
        print(f"❌ ERREUR lors du chargement du modèle:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n💡 SOLUTIONS POSSIBLES:")
        print("   1. Réentraîner le modèle: python train.py")
        print("   2. Vérifier la compatibilité des versions de bibliothèques")
        print("   3. Supprimer models/speaker_model.pkl et réentraîner")
        print("\n⚠️ L'application fonctionnera mais utilisera des clusters génériques\n")
        return None


# Charger le modèle au démarrage
print("\n" + "="*60)
print("🚀 INITIALISATION DU SYSTÈME")
print("="*60)

speaker_model = load_speaker_model()

# Initialiser les composants
diarizer = AutoSpeakerDiarizer(speaker_model=speaker_model)
transcriber = None  # Chargé à la demande
sentiment_analyzer = SentimentAnalyzer()

print("="*60 + "\n")


@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process audio file"""
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['audio_file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save the file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Complete audio analysis"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Analysis options
        do_transcription = data.get('transcription', True)
        do_sentiment = data.get('sentiment', True)
        
        # ✅ Convertir num_speakers en int
        num_speakers = data.get('num_speakers')
        if num_speakers is not None:
            try:
                num_speakers = int(num_speakers)
                if num_speakers < 1:
                    return jsonify({'error': 'num_speakers doit être >= 1'}), 400
            except (ValueError, TypeError):
                return jsonify({'error': f'num_speakers invalide: {num_speakers}'}), 400
        
        results = {}
        
        # 1. DIARIZATION
        print("🎤 Diarization in progress...")
        
        # ✅ Vérifier si le modèle est chargé
        if speaker_model is None:
            print("⚠️ ATTENTION: Pas de modèle chargé - utilisation de clusters génériques")
        
        segments = diarizer.diarize(
            filepath,
            num_speakers=num_speakers,
            min_segment_duration=0.5
        )
        
        # Merge close segments
        segments = diarizer.merge_close_segments(segments, gap_threshold=0.5)
        
        results['diarization'] = segments
        results['model_loaded'] = speaker_model is not None
        results['num_speakers_detected'] = len(set(s.get('speaker_id', 'UNKNOWN') for s in segments))
        
        # Speaker statistics
        speaker_stats = diarizer.get_speaker_statistics(segments)
        results['speaker_statistics'] = speaker_stats
        
        # 2. TRANSCRIPTION
        if do_transcription:
            print("📝 Transcription in progress...")
            global transcriber
            
            if transcriber is None:
                transcriber = AudioTranscriber(model_size='base')
            
            trans_segments = transcriber.transcribe_with_timestamps(filepath)
            
            # Align with diarization
            aligned_segments = transcriber.align_transcription_with_diarization(
                trans_segments, segments
            )
            
            results['transcription'] = aligned_segments
            results['full_transcript'] = transcriber.format_transcript(aligned_segments)
            
            # 3. SENTIMENT ANALYSIS
            if do_sentiment:
                print("😊 Sentiment analysis...")
                analyzed_segments = sentiment_analyzer.analyze_segments(aligned_segments)
                results['sentiment_analysis'] = analyzed_segments
                
                # Sentiment stats per speaker
                sentiment_stats = sentiment_analyzer.get_speaker_sentiment_stats(analyzed_segments)
                results['sentiment_statistics'] = sentiment_stats
        
        # 4. VISUALIZATIONS
        print("📊 Creating visualizations...")
        
        # Timeline
        timeline_path = create_timeline_plot(segments, output_dir='static/results')
        results['timeline_plot'] = timeline_path
        
        # Sentiment charts
        if do_sentiment and 'sentiment_analysis' in results:
            sentiment_chart_path = create_sentiment_chart(
                results['sentiment_analysis'],
                output_dir='static/results'
            )
            results['sentiment_chart'] = sentiment_chart_path
        
        # Statistics
        stats_chart_path = create_statistics_chart(
            speaker_stats,
            output_dir='static/results'
        )
        results['statistics_chart'] = stats_chart_path
        
        print("✅ Analysis complete!")
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/get_results/<filename>')
def get_results(filename):
    """Retrieve analysis results"""
    try:
        results_path = os.path.join('static/results', f'{filename}_results.json')
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({'error': 'Results not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download_transcript/<filename>')
def download_transcript(filename):
    """Download transcription"""
    try:
        transcript_path = os.path.join('static/results', f'{filename}_transcript.txt')
        
        if os.path.exists(transcript_path):
            return send_file(transcript_path, as_attachment=True)
        else:
            return jsonify({'error': 'Transcription not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Service health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': speaker_model is not None,
        'transcriber_loaded': transcriber is not None
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ADVANCED VOICE ANALYSIS SYSTEM")
    print("="*60)
    print(f"📂 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🤖 Model: {'✅ Loaded' if speaker_model else '❌ Not loaded (using generic clusters)'}")
    
    if speaker_model is None:
        print("\n⚠️ WARNING: Speaker identification model not loaded!")
        print("   The system will use generic cluster labels (SPEAKER_00, SPEAKER_01)")
        print("   instead of real LibriSpeech IDs (1272, 2035, etc.)")
        print("\n💡 To fix: Run 'python train.py' to train the model")
    
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)