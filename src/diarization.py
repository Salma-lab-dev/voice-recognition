"""
Système de diarisation automatique avec identification des locuteurs réels
VERSION AMÉLIORÉE - Corrections pour meilleure performance 2 speakers
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.signal import medfilt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class AutoSpeakerDiarizer:
    """
    Diarisation automatique avec identification des locuteurs réels
    
    AMÉLIORATIONS:
    - Meilleure détection automatique du nombre de locuteurs
    - Normalisation cohérente avec l'entraînement
    - VAD plus robuste avec seuils adaptatifs
    - Post-processing pour nettoyer les prédictions
    """
    
    def __init__(self, speaker_model=None, verbose=True):
        """
        Args:
            speaker_model: Modèle d'ensemble pré-entraîné
            verbose: Affichage détaillé
        """
        self.segments = []
        self.speaker_model = speaker_model
        self.verbose = verbose
        
        from src.feature_extraction import AdvancedFeatureExtractor
        self.feature_extractor = AdvancedFeatureExtractor()
        
    def extract_voice_segments(self, y: np.ndarray, sr: int,
                              min_segment_duration: float = 0.4,
                              energy_percentile: int = 20) -> List[Dict]:
        """
        Détection d'activité vocale AMÉLIORÉE avec seuils plus permissifs
        """
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)    # 10ms
        
        # 1. Énergie RMS
        rms = librosa.feature.rms(
            y=y, 
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # 2. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y, 
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # 3. Spectral centroid (nouveau - aide pour la voix)
        spec_cent = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=frame_length,
            hop_length=hop_length
        )[0]
        
        # Normalisation
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-8)
        zcr_norm = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-8)
        spec_norm = (spec_cent - np.min(spec_cent)) / (np.max(spec_cent) - np.min(spec_cent) + 1e-8)
        
        # Seuils adaptatifs PLUS PERMISSIFS
        energy_threshold = np.percentile(rms_norm, energy_percentile)
        zcr_max = np.percentile(zcr_norm, 80)  # Augmenté de 75 à 80
        spec_min = np.percentile(spec_norm, 15)  # Nouveau critère
        
        # Détection combinée avec 3 critères
        vad = (
            (rms_norm > energy_threshold) & 
            (zcr_norm < zcr_max) &
            (spec_norm > spec_min)
        )
        
        # Lissage plus agressif pour réduire les coupures
        vad_smooth = medfilt(vad.astype(float), kernel_size=7) > 0.5
        
        # Extraction des segments
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(vad_smooth):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
                end_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
                duration = end_time - start_time
                
                if duration >= min_segment_duration:
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segment_audio = y[start_sample:end_sample]
                    
                    segments.append({
                        'start': float(start_time),
                        'end': float(end_time),
                        'duration': float(duration),
                        'audio': segment_audio
                    })
                
                in_speech = False
        
        # Dernier segment
        if in_speech:
            end_time = len(y) / sr
            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            duration = end_time - start_time
            
            if duration >= min_segment_duration:
                start_sample = int(start_time * sr)
                segment_audio = y[start_sample:]
                
                segments.append({
                    'start': float(start_time),
                    'end': float(end_time),
                    'duration': float(duration),
                    'audio': segment_audio
                })
        
        return segments
    
    def extract_speaker_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extraction des features avec gestion d'erreur robuste"""
        try:
            features = self.feature_extractor.extract_complete_features(audio, sr)
            
            if len(features) != 3290:
                if self.verbose:
                    print(f"⚠️ Dimension incorrecte: {len(features)} (attendu 3290)")
                return None
            
            return features
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Erreur extraction: {e}")
            return None
    
    def auto_detect_num_speakers(self, features_matrix: np.ndarray, 
                                 max_speakers: int = 8,
                                 expected_speakers: int = 2) -> int:
        """
        Détection automatique AMÉLIORÉE du nombre de locuteurs
        
        Utilise plusieurs métriques et favorise le nombre attendu
        """
        from sklearn.preprocessing import StandardScaler
        
        # Normaliser
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_matrix)
        
        n_samples = len(features_matrix)
        max_k = min(max_speakers, n_samples - 1)
        
        if max_k < 2:
            return 1
        
        # Tester différents nombres de clusters
        scores = []
        
        for n_clusters in range(2, max_k + 1):
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = clusterer.fit_predict(features_normalized)
            
            # Métrique 1: Silhouette
            sil_score = silhouette_score(features_normalized, labels)
            
            # Métrique 2: Calinski-Harabasz
            ch_score = calinski_harabasz_score(features_normalized, labels)
            ch_normalized = ch_score / 1000  # Normaliser
            
            # Métrique 3: Distribution des clusters
            cluster_sizes = Counter(labels)
            min_cluster_size = min(cluster_sizes.values())
            balance_score = min_cluster_size / n_samples
            
            # Score combiné avec bonus pour expected_speakers
            combined_score = (
                0.4 * sil_score + 
                0.3 * min(ch_normalized, 1.0) + 
                0.3 * balance_score
            )
            
            # BONUS: Favoriser le nombre attendu de speakers
            if n_clusters == expected_speakers:
                combined_score *= 1.15  # Bonus de 15%
            
            scores.append({
                'n_clusters': n_clusters,
                'combined': combined_score,
                'silhouette': sil_score,
                'ch': ch_normalized,
                'balance': balance_score
            })
        
        if not scores:
            return expected_speakers
        
        # Trouver le meilleur score
        best = max(scores, key=lambda x: x['combined'])
        optimal_k = best['n_clusters']
        
        if self.verbose:
            print(f"\n📊 Scores de clustering:")
            for s in scores:
                marker = "🏆" if s['n_clusters'] == optimal_k else "  "
                print(f"{marker} {s['n_clusters']} speakers: "
                      f"combined={s['combined']:.3f} "
                      f"(sil={s['silhouette']:.3f}, "
                      f"balance={s['balance']:.3f})")
            print(f"🎯 Nombre optimal: {optimal_k}")
        
        return optimal_k
    
    def cluster_and_identify_speakers(self, segments: List[Dict], sr: int,
                                     num_speakers: Optional[int] = None,
                                     use_majority_voting: bool = True) -> List[Dict]:
        """
        Clustering + identification AMÉLIORÉE avec post-processing
        """
        if len(segments) == 0:
            return []
        
        if self.verbose:
            print(f"\n🔍 Extraction des features de {len(segments)} segments...")
        
        # Extraire features
        features_list = []
        valid_segments = []
        
        for i, seg in enumerate(segments):
            features = self.extract_speaker_features(seg['audio'], sr)
            
            if features is not None:
                features_list.append(features)
                valid_segments.append(seg)
        
        if len(features_list) == 0:
            if self.verbose:
                print("❌ Aucun segment valide!")
            return segments
        
        features_matrix = np.array(features_list)
        if self.verbose:
            print(f"✅ Matrice de features: {features_matrix.shape}")
        
        # Normalisation
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_matrix)
        
        # Détection automatique ou utilisation du nombre spécifié
        if num_speakers is None:
            num_speakers = self.auto_detect_num_speakers(
                features_matrix, 
                expected_speakers=2  # Par défaut 2 speakers
            )
        
        if self.verbose:
            print(f"\n🎯 Clustering avec {num_speakers} locuteur(s)...")
        
        # Clustering
        if num_speakers == 1:
            cluster_labels = np.zeros(len(valid_segments), dtype=int)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                linkage='ward'
            )
            cluster_labels = clustering.fit_predict(features_normalized)
        
        # Identification avec le modèle
        if self.speaker_model is not None:
            if self.verbose:
                print("🎤 Identification des locuteurs...")
            
            # Prédire pour tous les segments
            speaker_predictions = []
            confidences = []
            
            for i, features in enumerate(features_list):
                try:
                    features_reshaped = features.reshape(1, -1)
                    speaker_id = self.speaker_model.predict(features_reshaped)[0]
                    
                    try:
                        proba = self.speaker_model.predict_proba(features_reshaped)[0]
                        confidence = float(np.max(proba))
                    except:
                        confidence = 1.0
                    
                    speaker_predictions.append(speaker_id)
                    confidences.append(confidence)
                    
                except Exception as e:
                    if self.verbose and i < 3:
                        print(f"⚠️ Erreur segment {i}: {e}")
                    speaker_predictions.append("UNKNOWN")
                    confidences.append(0.0)
            
            # Post-processing: Majority voting par cluster
            if use_majority_voting and num_speakers > 1:
                speaker_predictions = self._apply_majority_voting(
                    speaker_predictions, 
                    cluster_labels,
                    confidences
                )
            
            # Assigner aux segments
            for i, segment in enumerate(valid_segments):
                segment['speaker_id'] = str(speaker_predictions[i])
                segment['cluster_id'] = f"CLUSTER_{cluster_labels[i]:02d}"
                segment['confidence'] = confidences[i]
                segment['speaker'] = str(speaker_predictions[i])
            
            # Afficher résumé
            if self.verbose:
                unique_speakers = set(speaker_predictions)
                print(f"✅ {len(unique_speakers)} locuteur(s) identifié(s): {sorted(unique_speakers)}")
                
                # Afficher quelques exemples
                for i in range(min(5, len(valid_segments))):
                    seg = valid_segments[i]
                    print(f"  [{seg['start']:.1f}s-{seg['end']:.1f}s] "
                          f"{seg['speaker_id']} (conf: {seg['confidence']:.1%})")
        else:
            # Pas de modèle - utiliser clusters
            for segment, label in zip(valid_segments, cluster_labels):
                segment['speaker_id'] = f"SPEAKER_{label:02d}"
                segment['cluster_id'] = f"CLUSTER_{label:02d}"
                segment['confidence'] = 1.0
                segment['speaker'] = f"SPEAKER_{label:02d}"
        
        return segments
    
    def _apply_majority_voting(self, predictions: List, 
                               cluster_labels: np.ndarray,
                               confidences: List[float]) -> List:
        """
        Post-processing: Assigner le speaker majoritaire à chaque cluster
        
        Corrige les erreurs de prédiction en utilisant le consensus du cluster
        """
        cluster_speakers = {}
        
        # Pour chaque cluster, trouver le speaker majoritaire
        for cluster_id in set(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_preds = [predictions[i] for i in range(len(predictions)) if cluster_mask[i]]
            cluster_confs = [confidences[i] for i in range(len(confidences)) if cluster_mask[i]]
            
            if not cluster_preds:
                continue
            
            # Compter les prédictions pondérées par la confiance
            speaker_scores = {}
            for pred, conf in zip(cluster_preds, cluster_confs):
                if pred not in speaker_scores:
                    speaker_scores[pred] = 0
                speaker_scores[pred] += conf
            
            # Speaker avec le score le plus élevé
            majority_speaker = max(speaker_scores, key=speaker_scores.get)
            cluster_speakers[cluster_id] = majority_speaker
        
        # Appliquer le majority voting
        corrected_predictions = []
        for i, pred in enumerate(predictions):
            cluster_id = cluster_labels[i]
            majority = cluster_speakers.get(cluster_id, pred)
            corrected_predictions.append(majority)
        
        return corrected_predictions
    
    def diarize(self, audio_path: str, 
                num_speakers: Optional[int] = None,
                min_segment_duration: float = 0.4,
                merge_gaps: bool = True,
                gap_threshold: float = 0.5) -> List[Dict]:
        """
        Pipeline complet de diarisation avec options avancées
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎤 DIARISATION AUTOMATIQUE")
            print(f"{'='*60}")
            print(f"📂 Fichier: {audio_path}")
        
        # 1. Charger audio
        if self.verbose:
            print("\n📂 Chargement audio...")
        y, sr = librosa.load(audio_path, sr=16000)
        if self.verbose:
            print(f"✅ Durée: {len(y)/sr:.2f}s, SR: {sr}Hz")
        
        # 2. Débruitage
        if self.verbose:
            print("🔇 Réduction du bruit...")
        y_denoised = self._simple_denoise(y, sr)
        
        # 3. Détection VAD
        if self.verbose:
            print("🎤 Détection des segments vocaux...")
        segments = self.extract_voice_segments(
            y_denoised, sr, 
            min_segment_duration=min_segment_duration
        )
        
        if len(segments) == 0:
            if self.verbose:
                print("⚠️ Aucun segment vocal détecté!")
            return []
        
        if self.verbose:
            print(f"✅ {len(segments)} segments détectés")
        
        # 4. Clustering + identification
        segments = self.cluster_and_identify_speakers(
            segments, sr, 
            num_speakers=num_speakers
        )
        
        # 5. Fusion des segments proches (optionnel)
        if merge_gaps:
            if self.verbose:
                print(f"\n🔗 Fusion des segments proches (gap < {gap_threshold}s)...")
            original_count = len(segments)
            segments = self.merge_close_segments(segments, gap_threshold)
            if self.verbose:
                print(f"✅ {original_count} → {len(segments)} segments")
        
        # 6. Afficher résumé
        if self.verbose:
            self._print_summary(segments)
        
        # 7. Nettoyer
        for seg in segments:
            if 'audio' in seg:
                del seg['audio']
        
        return segments
    
    def _simple_denoise(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Débruitage par soustraction spectrale"""
        D = librosa.stft(y)
        magnitude, phase = librosa.magphase(D)
        
        noise_frames = min(10, magnitude.shape[1])
        noise_profile = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        magnitude_clean = np.maximum(magnitude - 0.8 * noise_profile, 0.05 * magnitude)
        
        D_clean = magnitude_clean * phase
        y_clean = librosa.istft(D_clean)
        
        return y_clean
    
    def _print_summary(self, segments: List[Dict]):
        """Affiche un résumé détaillé"""
        print(f"\n{'='*60}")
        print("📊 RÉSUMÉ DE LA DIARISATION")
        print(f"{'='*60}")
        
        if not segments:
            return
        
        # Statistiques par speaker
        stats = self.get_speaker_statistics(segments)
        
        print(f"\n👥 Locuteurs identifiés:")
        for speaker in sorted(stats.keys()):
            s = stats[speaker]
            print(f"\n  {speaker}:")
            print(f"    • Temps de parole: {s['total_time']:.2f}s ({s['percentage']:.1f}%)")
            print(f"    • Nombre de segments: {s['num_segments']}")
            print(f"    • Confiance moyenne: {s['confidence']:.1%}")
        
        print(f"\n{'='*60}\n")
    
    def merge_close_segments(self, segments: List[Dict], 
                            gap_threshold: float = 0.5) -> List[Dict]:
        """Fusionne les segments proches du même locuteur"""
        if not segments:
            return []
        
        segments = sorted(segments, key=lambda x: x['start'])
        merged = [segments[0].copy()]
        
        for current in segments[1:]:
            last = merged[-1]
            
            same_speaker = (
                current.get('speaker_id') == last.get('speaker_id') and
                current['speaker_id'] != "UNKNOWN"
            )
            
            gap = current['start'] - last['end']
            
            if same_speaker and gap < gap_threshold:
                # Fusionner
                last['end'] = current['end']
                last['duration'] = last['end'] - last['start']
                
                # Moyenne des confidences
                if 'confidence' in current and 'confidence' in last:
                    last['confidence'] = (last['confidence'] + current['confidence']) / 2
            else:
                merged.append(current.copy())
        
        return merged
    
    def get_speaker_statistics(self, segments: List[Dict]) -> Dict:
        """Calcule les statistiques détaillées par locuteur"""
        if not segments:
            return {}
        
        stats = {}
        total_duration = max(seg['end'] for seg in segments) if segments else 0
        
        for segment in segments:
            speaker = segment.get('speaker_id', 'UNKNOWN')
            
            if speaker not in stats:
                stats[speaker] = {
                    'total_time': 0.0,
                    'num_segments': 0,
                    'segments': [],
                    'avg_confidence': []
                }
            
            stats[speaker]['total_time'] += segment['duration']
            stats[speaker]['num_segments'] += 1
            stats[speaker]['segments'].append(segment)
            
            if 'confidence' in segment:
                stats[speaker]['avg_confidence'].append(segment['confidence'])
        
        # Calculer les pourcentages et moyennes
        for speaker in stats:
            if total_duration > 0:
                stats[speaker]['percentage'] = (
                    stats[speaker]['total_time'] / total_duration * 100
                )
            else:
                stats[speaker]['percentage'] = 0
                
            if stats[speaker]['avg_confidence']:
                stats[speaker]['confidence'] = np.mean(stats[speaker]['avg_confidence'])
            else:
                stats[speaker]['confidence'] = 0
        
        return stats