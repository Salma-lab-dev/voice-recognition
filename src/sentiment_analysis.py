"""
Improved Sentiment Analysis with multi-language support
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import Dict, List, Optional
import numpy as np
import re


class SentimentAnalyzer:
    """
    Enhanced sentiment analysis with multi-language and context support
    """
    
    def __init__(self, language: str = 'en'):
        """
        Args:
            language: 'en' for English, 'fr' for French
        """
        self.language = language
        self.vader = SentimentIntensityAnalyzer()
        
        # Extended emotion keywords by language
        self.emotion_keywords = self._load_emotion_keywords()
        
        # Intensifiers and negations
        self.intensifiers = self._load_intensifiers()
        self.negations = self._load_negations()
    
    def _load_emotion_keywords(self) -> Dict:
        """Load extended emotion keywords"""
        if self.language == 'fr':
            return {
                'joy': ['heureux', 'joie', 'content', 'ravi', 'enchanté', 'super', 
                        'génial', 'formidable', 'excellent', 'merveilleux', 'parfait'],
                'anger': ['colère', 'énervé', 'furieux', 'fâché', 'irrité', 
                         'agacé', 'déteste', 'horrible', 'inacceptable'],
                'sadness': ['triste', 'malheureux', 'déprimé', 'déçu', 'peine',
                           'terrible', 'catastrophique', 'désespéré'],
                'fear': ['peur', 'inquiet', 'anxieux', 'stressé', 'nerveux',
                        'préoccupé', 'effrayé', 'craintif'],
                'surprise': ['surpris', 'étonné', 'choqué', 'stupéfait', 
                            'inattendu', 'incroyable'],
                'disgust': ['dégoûté', 'répugnant', 'écœurant', 'horrible'],
                'trust': ['confiance', 'fiable', 'sûr', 'croire', 'compter']
            }
        else:  # English
            return {
                'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'love',
                       'amazing', 'fantastic', 'delighted', 'pleased', 'glad',
                       'cheerful', 'joyful', 'thrilled', 'excellent', 'awesome'],
                'anger': ['angry', 'mad', 'furious', 'hate', 'annoyed', 'irritated',
                         'frustrated', 'outraged', 'hostile', 'rage', 'upset'],
                'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'terrible',
                           'disappointed', 'hurt', 'sorry', 'regret', 'gloomy'],
                'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous',
                        'frightened', 'terrified', 'panic', 'dread', 'concerned'],
                'surprise': ['surprised', 'shocked', 'amazed', 'unexpected',
                            'astonished', 'startled', 'incredible', 'wow'],
                'disgust': ['disgusted', 'revolting', 'gross', 'nasty', 'awful'],
                'trust': ['trust', 'reliable', 'confident', 'believe', 'depend']
            }
    
    def _load_intensifiers(self) -> List[str]:
        """Load intensifier words"""
        if self.language == 'fr':
            return ['très', 'vraiment', 'extrêmement', 'tellement', 'super',
                   'absolument', 'complètement', 'totalement']
        else:
            return ['very', 'really', 'extremely', 'so', 'absolutely',
                   'completely', 'totally', 'incredibly', 'highly']
    
    def _load_negations(self) -> List[str]:
        """Load negation words"""
        if self.language == 'fr':
            return ['ne', 'pas', 'non', 'jamais', 'rien', 'personne', 'aucun']
        else:
            return ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere',
                   'neither', 'none', "n't", 'cannot', "won't", "don't"]
    
    def detect_negation(self, text: str) -> bool:
        """Detect if text contains negation"""
        text_lower = text.lower()
        return any(neg in text_lower.split() for neg in self.negations)
    
    def detect_intensifiers(self, text: str) -> int:
        """Count intensifiers in text"""
        text_lower = text.lower()
        return sum(1 for intensifier in self.intensifiers 
                  if intensifier in text_lower)
    
    def analyze_with_vader(self, text: str) -> Dict:
        """Enhanced VADER analysis with negation handling"""
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        # Adjust for negation
        has_negation = self.detect_negation(text)
        if has_negation:
            compound *= -0.5  # Dampen but don't fully reverse
        
        # Adjust for intensifiers
        num_intensifiers = self.detect_intensifiers(text)
        if num_intensifiers > 0:
            compound *= (1 + 0.1 * num_intensifiers)
            compound = np.clip(compound, -1, 1)
        
        # Determine label with adaptive threshold
        if compound >= 0.1:
            label = 'positive'
        elif compound <= -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'method': 'vader',
            'scores': scores,
            'compound': compound,
            'label': label,
            'confidence': abs(compound),
            'has_negation': has_negation,
            'intensifiers': num_intensifiers
        }
    
    def analyze_with_textblob(self, text: str) -> Dict:
        """TextBlob analysis (works better for English)"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Adaptive thresholds
            if polarity > 0.15:
                label = 'positive'
            elif polarity < -0.15:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'method': 'textblob',
                'polarity': polarity,
                'subjectivity': subjectivity,
                'label': label,
                'confidence': abs(polarity)
            }
        except:
            return None
    
    def analyze_emotion_keywords(self, text: str) -> Dict:
        """Enhanced keyword-based emotion detection"""
        text_lower = text.lower()
        words = text_lower.split()
        
        emotion_scores = {emotion: 0 for emotion in self.emotion_keywords.keys()}
        
        # Score with context window
        for i, word in enumerate(words):
            for emotion, keywords in self.emotion_keywords.items():
                if word in keywords:
                    # Base score
                    score = 1
                    
                    # Check for intensifiers nearby (within 2 words)
                    context_start = max(0, i - 2)
                    context_end = min(len(words), i + 3)
                    context = words[context_start:context_end]
                    
                    if any(intensifier in context for intensifier in self.intensifiers):
                        score *= 1.5
                    
                    # Check for negation
                    if any(neg in context for neg in self.negations):
                        score *= -0.5
                    
                    emotion_scores[emotion] += score
        
        # Normalize scores
        total_score = sum(abs(s) for s in emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
        
        # Get dominant emotion
        if total_score > 0:
            dominant = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant]
        else:
            dominant = 'neutral'
            confidence = 1.0
        
        return {
            'method': 'keywords',
            'emotions': emotion_scores,
            'dominant_emotion': dominant,
            'confidence': confidence
        }
    
    def analyze_complete(self, text: str, context: Optional[List[str]] = None) -> Dict:
        """
        Complete analysis with optional conversation context
        
        Args:
            text: Text to analyze
            context: Previous utterances for context (optional)
        """
        if not text.strip():
            return {
                'text': text,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'emotion': 'neutral'
            }
        
        # Run all analyses
        vader_result = self.analyze_with_vader(text)
        textblob_result = self.analyze_with_textblob(text)
        emotion_result = self.analyze_emotion_keywords(text)
        
        # Weighted voting (VADER gets higher weight for social text)
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        # VADER vote (weight: 2)
        sentiment_scores[vader_result['label']] += 2 * vader_result['confidence']
        
        # TextBlob vote (weight: 1) - skip if French
        if textblob_result and self.language == 'en':
            sentiment_scores[textblob_result['label']] += textblob_result['confidence']
        
        # Emotion contributes to sentiment
        emotion = emotion_result['dominant_emotion']
        if emotion in ['joy', 'trust', 'surprise']:
            sentiment_scores['positive'] += emotion_result['confidence']
        elif emotion in ['anger', 'sadness', 'fear', 'disgust']:
            sentiment_scores['negative'] += emotion_result['confidence']
        
        # Final sentiment
        final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        total_score = sum(sentiment_scores.values())
        confidence = sentiment_scores[final_sentiment] / total_score if total_score > 0 else 0
        
        return {
            'text': text,
            'sentiment': final_sentiment,
            'confidence': float(confidence),
            'emotion': emotion,
            'emotion_confidence': emotion_result['confidence'],
            'details': {
                'vader': vader_result,
                'textblob': textblob_result,
                'emotions': emotion_result,
                'sentiment_scores': sentiment_scores
            }
        }
    
    def analyze_segments(self, segments: List[Dict], 
                        use_context: bool = True) -> List[Dict]:
        """
        Analyze sentiment with optional conversation context
        
        Args:
            segments: List of segments with 'text' and 'speaker_id'
            use_context: Whether to use previous utterances as context
        """
        analyzed_segments = []
        context_window = []
        
        for seg in segments:
            text = seg.get('text', '')
            
            if text.strip():
                # Analyze with context
                context = context_window[-3:] if use_context else None
                sentiment_result = self.analyze_complete(text, context)
                
                seg_analyzed = seg.copy()
                seg_analyzed['sentiment'] = sentiment_result['sentiment']
                seg_analyzed['emotion'] = sentiment_result['emotion']
                seg_analyzed['sentiment_confidence'] = sentiment_result['confidence']
                seg_analyzed['emotion_confidence'] = sentiment_result.get('emotion_confidence', 0)
                seg_analyzed['sentiment_details'] = sentiment_result['details']
                
                analyzed_segments.append(seg_analyzed)
                
                # Update context
                if use_context:
                    context_window.append(text)
            else:
                analyzed_segments.append(seg)
        
        return analyzed_segments
    
    def get_conversation_sentiment_flow(self, segments: List[Dict]) -> Dict:
        """
        Analyze sentiment flow over conversation
        """
        timeline = []
        
        for seg in segments:
            if 'sentiment' in seg:
                timeline.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'sentiment': seg['sentiment'],
                    'confidence': seg.get('sentiment_confidence', 0),
                    'speaker': seg.get('speaker_id', 'UNKNOWN')
                })
        
        # Calculate sentiment trajectory
        sentiment_values = []
        for item in timeline:
            if item['sentiment'] == 'positive':
                sentiment_values.append(1 * item['confidence'])
            elif item['sentiment'] == 'negative':
                sentiment_values.append(-1 * item['confidence'])
            else:
                sentiment_values.append(0)
        
        return {
            'timeline': timeline,
            'sentiment_trajectory': sentiment_values,
            'average_sentiment': np.mean(sentiment_values) if sentiment_values else 0,
            'sentiment_variance': np.var(sentiment_values) if sentiment_values else 0,
            'overall_trend': 'improving' if len(sentiment_values) > 1 and 
                           sentiment_values[-1] > sentiment_values[0] else 'declining'
        }
    
    def get_speaker_sentiment_stats(self, segments: List[Dict]) -> Dict:
        """Enhanced speaker statistics with trends"""
        speaker_stats = {}
        
        for seg in segments:
            speaker = seg.get('speaker_id', 'UNKNOWN')
            sentiment = seg.get('sentiment')
            emotion = seg.get('emotion')
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'sentiments': [],
                    'emotions': [],
                    'confidences': [],
                    'total_duration': 0,
                    'utterance_count': 0
                }
            
            if sentiment:
                speaker_stats[speaker]['sentiments'].append(sentiment)
                speaker_stats[speaker]['confidences'].append(
                    seg.get('sentiment_confidence', 0)
                )
            if emotion:
                speaker_stats[speaker]['emotions'].append(emotion)
            
            speaker_stats[speaker]['total_duration'] += seg.get('duration', 0)
            speaker_stats[speaker]['utterance_count'] += 1
        
        # Calculate aggregate statistics
        for speaker, stats in speaker_stats.items():
            if stats['sentiments']:
                # Sentiment distribution
                sentiment_counts = {
                    'positive': stats['sentiments'].count('positive'),
                    'negative': stats['sentiments'].count('negative'),
                    'neutral': stats['sentiments'].count('neutral')
                }
                stats['sentiment_distribution'] = sentiment_counts
                stats['dominant_sentiment'] = max(sentiment_counts, key=sentiment_counts.get)
                
                # Average confidence
                stats['average_confidence'] = np.mean(stats['confidences'])
                
                # Sentiment consistency (low variance = consistent)
                sentiment_numeric = [1 if s == 'positive' else -1 if s == 'negative' else 0 
                                   for s in stats['sentiments']]
                stats['sentiment_consistency'] = 1 - np.var(sentiment_numeric)
            
            if stats['emotions']:
                emotion_counts = {}
                for emotion in stats['emotions']:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                stats['emotion_distribution'] = emotion_counts
                stats['dominant_emotion'] = max(emotion_counts, key=emotion_counts.get)
        
        return speaker_stats