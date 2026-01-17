"""
Entraînement de modèles ensemblistes
CORRECTION: Méthode load_model corrigée pour retourner l'instance
"""

import os
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class EnsembleSpeakerRecognition:
    """
    Système de reconnaissance basé sur ensemble de modèles
    
    Amélioration par rapport au mini-projet:
    - Multiple modèles combinés (voting/stacking)
    - Hyperparamètres optimisés
    - Validation croisée
    - Gestion robuste des classes
    """
    
    def __init__(self, ensemble_type: str = 'voting'):
        """
        Args:
            ensemble_type: 'voting', 'stacking', ou 'single'
        """
        self.ensemble_type = ensemble_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_dim = None
        
    def create_base_models(self) -> Dict:
        """
        Crée les modèles de base optimisés
        """
        models = {
            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=500,
                random_state=42
            )
        }
        
        return models
    
    def create_ensemble_model(self):
        """
        Crée le modèle ensembliste
        """
        base_models = self.create_base_models()
        
        if self.ensemble_type == 'voting':
            # Voting Classifier: combine les prédictions
            estimators = [
                ('rf', base_models['rf']),
                ('xgb', base_models['xgb']),
                ('lgb', base_models['lgb']),
                ('svm', base_models['svm'])
            ]
            
            self.model = VotingClassifier(
                estimators=estimators,
                voting='soft',  # Utilise les probabilités
                n_jobs=-1
            )
            
        elif self.ensemble_type == 'stacking':
            # Stacking: meta-modèle apprend à combiner
            estimators = [
                ('rf', base_models['rf']),
                ('xgb', base_models['xgb']),
                ('lgb', base_models['lgb'])
            ]
            
            self.model = StackingClassifier(
                estimators=estimators,
                final_estimator=base_models['mlp'],
                cv=5,
                n_jobs=-1
            )
            
        elif self.ensemble_type == 'single':
            # Un seul modèle (XGBoost performant)
            self.model = base_models['xgb']
            
        else:
            raise ValueError(f"Type inconnu: {self.ensemble_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2) -> Dict:
        """
        Entraîne le modèle avec validation
        
        Returns:
            Dict avec métriques de performance
        """
        print(f"🎯 Entraînement du modèle ({self.ensemble_type})...")
        
        # Encoder les labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=42,
            stratify=y_encoded
        )
        
        print(f"📊 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_dim = X_train.shape[1]
        
        # Créer et entraîner le modèle
        self.create_ensemble_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test_scaled)
        
        # Validation croisée sur train
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5
        )
        
        # Métriques
        results = {
            'test_accuracy': np.mean(y_pred == y_test),
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"✅ Test Accuracy: {results['test_accuracy']:.2%}")
        print(f"✅ CV Accuracy: {results['cv_accuracy_mean']:.2%} ± {results['cv_accuracy_std']:.2%}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédiction simple (retourne juste les labels)
        
        Returns:
            labels (array de strings comme '1272', '2035')
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Décoder les labels vers les IDs originaux
        labels = self.label_encoder.inverse_transform(predictions)
        
        return labels
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédiction avec probabilités
        
        Returns:
            probabilities array
        """
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def save_model(self, path: str = 'models/speaker_model.pkl'):
        """Sauvegarde le modèle complet"""
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'ensemble_type': self.ensemble_type,
            'feature_dim': self.feature_dim
        }
        joblib.dump(model_data, path, compress=3)
        print(f"💾 Modèle sauvegardé: {path}")
    
    @staticmethod
    def load_model(path: str = 'models/speaker_model.pkl'):
        """
        ✅ CORRECTION: Charge un modèle sauvegardé (méthode statique)
        
        Args:
            path: Chemin vers le fichier .pkl
        
        Returns:
            Instance de EnsembleSpeakerRecognition avec le modèle chargé
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle non trouvé: {path}")
        
        # Charger les données
        model_data = joblib.load(path)
        
        # Créer une nouvelle instance
        instance = EnsembleSpeakerRecognition()
        
        # Charger les attributs
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.label_encoder = model_data['label_encoder']
        instance.ensemble_type = model_data['ensemble_type']
        instance.feature_dim = model_data['feature_dim']
        
        print(f"📂 Modèle chargé: {path}")
        
        return instance