"""
Script d'entraînement du modèle de reconnaissance du locuteur
Amélioration du mini-projet avec ensemble models
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import AudioPreprocessor
from src.feature_extraction import AdvancedFeatureExtractor
from src.model_training import EnsembleSpeakerRecognition


def download_librispeech():
    """Télécharge LibriSpeech dev-clean si nécessaire"""
    import urllib.request
    import tarfile
    
    if not os.path.exists("LibriSpeech"):
        print("Téléchargement de LibriSpeech...")
        url = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
        filename = "dev-clean.tar.gz"
        
        urllib.request.urlretrieve(url, filename)
        
        print("📦 Extraction...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
        
        os.remove(filename)
        print(" Dataset téléchargé!")
    else:
        print(" Dataset déjà présent")


def load_dataset(base_path: str = "LibriSpeech/dev-clean",
                n_speakers: int = 10,
                max_files_per_speaker: int = 60,
                use_augmentation: bool = True):
    """
    Charge et extrait les features du dataset
    
    Args:
        base_path: Chemin vers LibriSpeech
        n_speakers: Nombre de locuteurs
        max_files_per_speaker: Nombre max de fichiers par locuteur
        use_augmentation: Utiliser l'augmentation de données
    
    Returns:
        X, y, speaker_ids
    """
    print(f"📂 Chargement du dataset ({n_speakers} locuteurs)...")
    
    preprocessor = AudioPreprocessor()
    feature_extractor = AdvancedFeatureExtractor()
    
    X = []
    y = []
    speaker_ids = []
    
    speakers = sorted(os.listdir(base_path))[:n_speakers]
    
    for speaker_id in tqdm(speakers, desc="Traitement des locuteurs"):
        speaker_path = os.path.join(base_path, speaker_id)
        
        if not os.path.isdir(speaker_path):
            continue
        
        speaker_ids.append(speaker_id)
        file_count = 0
        
        for chapter in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter)
            
            if not os.path.isdir(chapter_path):
                continue
            
            for audio_file in os.listdir(chapter_path):
                if audio_file.endswith('.flac') and file_count < max_files_per_speaker:
                    audio_path = os.path.join(chapter_path, audio_file)
                    
                    try:
                        # Prétraiter
                        y_audio, sr = preprocessor.preprocess_pipeline(
                            audio_path,
                            denoise=False,
                            preemphasis=True
                        )
                        
                        # Extraire features
                        features = feature_extractor.extract_complete_features(y_audio, sr)
                        
                        X.append(features)
                        y.append(speaker_id)
                        file_count += 1
                        
                        # Augmentation de données
                        if use_augmentation and file_count < max_files_per_speaker:
                            # Pitch shift
                            y_aug = preprocessor.augment_audio(y_audio, sr, 'pitch_shift')
                            features_aug = feature_extractor.extract_complete_features(y_aug, sr)
                            X.append(features_aug)
                            y.append(speaker_id)
                            file_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ Erreur {audio_file}: {e}")
                        continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Dataset chargé:")
    print(f"   Échantillons: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Locuteurs: {len(speaker_ids)}")
    
    return X, y, speaker_ids


def plot_training_results(results: dict, speaker_ids: list):
    """Visualise les résultats d'entraînement"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Matrice de confusion
    cm = results['confusion_matrix']
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=speaker_ids,
        yticklabels=speaker_ids,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Prédiction')
    axes[0, 0].set_ylabel('Vérité terrain')
    
    # 2. Accuracy par classe
    report = results['classification_report']
    classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
    f1_scores = [report[c]['f1-score'] for c in classes]
    
    axes[0, 1].barh(range(len(classes)), f1_scores, color='skyblue')
    axes[0, 1].set_yticks(range(len(classes)))
    axes[0, 1].set_yticklabels(classes)
    axes[0, 1].set_xlabel('F1-Score')
    axes[0, 1].set_title('Performance par Locuteur', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlim([0, 1])
    
    # 3. Métriques globales
    metrics = {
        'Test Accuracy': results['test_accuracy'],
        'CV Accuracy': results['cv_accuracy_mean'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-Score': report['weighted avg']['f1-score']
    }
    
    axes[1, 0].bar(metrics.keys(), metrics.values(), color='lightgreen')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Métriques Globales', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Confusion détaillée (top erreurs)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_norm, 0)  # Ignorer diagonale
    
    # Trouver top confusions
    top_confusions = []
    for i in range(len(speaker_ids)):
        for j in range(len(speaker_ids)):
            if i != j and cm[i, j] > 0:
                top_confusions.append({
                    'pair': f"{speaker_ids[i]}→{speaker_ids[j]}",
                    'count': cm[i, j],
                    'rate': cm_norm[i, j]
                })
    
    top_confusions.sort(key=lambda x: x['count'], reverse=True)
    top_5 = top_confusions[:5]
    
    if top_5:
        pairs = [c['pair'] for c in top_5]
        counts = [c['count'] for c in top_5]
        
        axes[1, 1].barh(range(len(pairs)), counts, color='salmon')
        axes[1, 1].set_yticks(range(len(pairs)))
        axes[1, 1].set_yticklabels(pairs)
        axes[1, 1].set_xlabel('Nombre de confusions')
        axes[1, 1].set_title('Top 5 Confusions', fontsize=14, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Aucune confusion!', 
                       ha='center', va='center', fontsize=16)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print(" Graphiques sauvegardés: training_results.png")
    plt.show()


def compare_models(X, y):
    """Compare différents types d'ensemble"""
    
    print("\n" + "="*60)
    print("COMPARAISON DES MODÈLES")
    print("="*60)
    
    model_types = ['voting', 'stacking', 'single']
    results_comparison = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Modèle: {model_type.upper()}")
        print(f"{'='*60}")
        
        model = EnsembleSpeakerRecognition(ensemble_type=model_type)
        results = model.train(X, y, test_size=0.2)
        
        results_comparison[model_type] = {
            'test_accuracy': results['test_accuracy'],
            'cv_accuracy': results['cv_accuracy_mean'],
            'cv_std': results['cv_accuracy_std']
        }
        
        print(f"\nRésultats {model_type}:")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    
    # Trouver le meilleur
    best_model_type = max(results_comparison.items(), 
                         key=lambda x: x[1]['test_accuracy'])[0]
    
    print(f"\n🏆 Meilleur modèle: {best_model_type.upper()}")
    
    return best_model_type, results_comparison


def main():
    """Pipeline complet d'entraînement"""
    
    print("\n" + "="*60)
    print("SYSTÈME AVANCÉ DE RECONNAISSANCE DU LOCUTEUR")
    print("Amélioration avec Ensemble Models")
    print("="*60 + "\n")
    
    # Créer dossiers
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Télécharger données
    download_librispeech()
    
    # Charger dataset
    X, y, speaker_ids = load_dataset(
        n_speakers=10,
        max_files_per_speaker=60,
        use_augmentation=True
    )
    
    # Comparer modèles
    best_model_type, comparison = compare_models(X, y)
    
    # Entraîner le meilleur modèle final
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT DU MODÈLE FINAL ({best_model_type.upper()})")
    print(f"{'='*60}\n")
    
    final_model = EnsembleSpeakerRecognition(ensemble_type=best_model_type)
    final_results = final_model.train(X, y, test_size=0.2)
    
    # Sauvegarder
    final_model.save_model('models/speaker_model.pkl')
    
    # Visualiser
    plot_training_results(final_results, speaker_ids)
    
    # Rapport final
    print("\n" + "="*60)
    print("RAPPORT FINAL")
    print("="*60)
    print(f"\n Modèle: {best_model_type}")
    print(f"Test Accuracy: {final_results['test_accuracy']:.2%}")
    print(f" CV Accuracy: {final_results['cv_accuracy_mean']:.2%} ± {final_results['cv_accuracy_std']:.2%}")
    print(f"\n Modèle sauvegardé: models/speaker_model.pkl")
    print(f"Graphiques: training_results.png")
    
    


if __name__ == "__main__":
    main()