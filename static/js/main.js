// État global
let currentFile = null;
let currentResults = null;

// Éléments DOM
const uploadArea = document.getElementById('uploadArea');
const audioFile = document.getElementById('audioFile');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const analyzeBtn = document.getElementById('analyzeBtn');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// Upload handlers
uploadArea.addEventListener('click', () => audioFile.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

audioFile.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    // Check type
    if (!file.type.startsWith('audio/')) {
        showError('Please select a valid audio file.');
        return;
    }
    
    // Check size (50MB)
    if (file.size > 50 * 1024 * 1024) {
        showError('File is too large (max 50MB).');
        return;
    }
    
    currentFile = file;
    
    // Afficher les infos
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.remove('hidden');
    analyzeBtn.disabled = false;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Analyse
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    
    try {
        // Hide previous sections
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        
        // Show progress
        progressSection.classList.remove('hidden');
        updateProgress(0, 'Uploading file...');
        
        // 1. Upload
        const formData = new FormData();
        formData.append('audio_file', currentFile);
        
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!uploadResponse.ok) {
            throw new Error('Error during upload');
        }
        
        const uploadData = await uploadResponse.json();
        updateProgress(20, 'File uploaded, analyzing...');
        
        // 2. Analysis
        const options = {
            filepath: uploadData.filepath,
            transcription: document.getElementById('optTranscription').checked,
            sentiment: document.getElementById('optSentiment').checked,
            num_speakers: document.getElementById('numSpeakers').value || null
        };
        
        updateProgress(30, 'Diarization in progress...');
        
        const analyzeResponse = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(options)
        });
        
        if (!analyzeResponse.ok) {
            const errorData = await analyzeResponse.json();
            throw new Error(errorData.error || 'Error during analysis');
        }
        
        updateProgress(70, 'Processing results...');
        
        const results = await analyzeResponse.json();
        currentResults = results.results;
        
        updateProgress(100, 'Analysis complete!');
        
        // Wait a bit before showing results
        setTimeout(() => {
            progressSection.classList.add('hidden');
            displayResults(currentResults);
        }, 500);
        
    } catch (error) {
        console.error('Error:', error);
        progressSection.classList.add('hidden');
        showError(error.message || 'An error occurred during analysis.');
    }
});

function updateProgress(percent, text) {
    progressFill.style.width = percent + '%';
    progressText.textContent = text;
}

function displayResults(results) {
    resultsSection.classList.remove('hidden');
    
    // 1. Timeline de diarisation
    if (results.timeline_plot) {
        document.getElementById('timelineImg').src = `/static/results/${results.timeline_plot}`;
    }
    
    // Diarization stats
    const diarizationStats = document.getElementById('diarizationStats');
    diarizationStats.innerHTML = `
        <div class="stat-item">
            <span class="stat-value">${results.num_speakers_detected}</span>
            <span class="stat-label">Detected Speakers</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">${results.diarization.length}</span>
            <span class="stat-label">Total Segments</span>
        </div>
    `;
    
    // 2. Transcription
    if (results.transcription) {
        document.getElementById('transcriptionCard').classList.remove('hidden');
        displayTranscription(results.transcription);
    }
    
    // 3. Sentiment
    if (results.sentiment_analysis) {
        document.getElementById('sentimentCard').classList.remove('hidden');
        
        if (results.sentiment_chart) {
            document.getElementById('sentimentImg').src = `/static/results/${results.sentiment_chart}`;
        }
        
        displaySentimentStats(results.sentiment_statistics);
    }
    
    // 4. Statistiques
    if (results.statistics_chart) {
        document.getElementById('statisticsImg').src = `/static/results/${results.statistics_chart}`;
    }
    
    displaySpeakerStats(results.speaker_statistics);
}

function displayTranscription(segments) {
    const container = document.getElementById('transcriptContainer');
    container.innerHTML = '';
    
    segments.forEach(seg => {
        const line = document.createElement('div');
        line.className = 'transcript-line';
        
        const speaker = seg.speaker_id || 'UNKNOWN';
        const startTime = formatTime(seg.start);
        const endTime = formatTime(seg.end);
        const text = seg.text || '';
        
        line.innerHTML = `
            <span class="transcript-speaker">${speaker}</span>
            <span class="transcript-time">[${startTime} - ${endTime}]</span>
            <span>${text}</span>
        `;
        
        container.appendChild(line);
    });
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function displaySentimentStats(stats) {
    const container = document.getElementById('sentimentStats');
    container.innerHTML = '';
    
    for (const [speaker, data] of Object.entries(stats)) {
        const item = document.createElement('div');
        item.className = 'sentiment-item';
        
        const sentiment = data.dominant_sentiment || 'Inconnu';
        const emotion = data.dominant_emotion || 'Inconnu';
        
        item.innerHTML = `
            <h4>${speaker}</h4>
            <p><strong>Sentiment:</strong> ${translateSentiment(sentiment)}</p>
            <p><strong>Émotion:</strong> ${translateEmotion(emotion)}</p>
        `;
        
        container.appendChild(item);
    }
}

function translateSentiment(sentiment) {
    const map = {
        'positive': '😊 Positive',
        'negative': '😞 Negative',
        'neutral': '😐 Neutral'
    };
    return map[sentiment] || sentiment;
}

function translateEmotion(emotion) {
    const map = {
        'joy': '😊 Joy',
        'anger': '😠 Anger',
        'sadness': '😢 Sadness',
        'fear': '😨 Fear',
        'surprise': '😮 Surprise',
        'neutral': '😐 Neutral'
    };
    return map[emotion] || emotion;
}

function displaySpeakerStats(stats) {
    const container = document.getElementById('speakerStats');
    container.innerHTML = '';
    
    for (const [speaker, data] of Object.entries(stats)) {
        const card = document.createElement('div');
        card.className = 'speaker-card';
        
        const totalTime = data.total_time.toFixed(2);
        const numSegments = data.num_segments;
        const percentage = data.percentage?.toFixed(1) || 0;
        const confidence = data.confidence ? (data.confidence * 100).toFixed(1) : 'N/A';
        
        card.innerHTML = `
            <h4>🎤 ${speaker}</h4>
            <div class="speaker-detail">
                <span>Speaking time:</span>
                <strong>${totalTime}s (${percentage}%)</strong>
            </div>
            <div class="speaker-detail">
                <span>Interventions:</span>
                <strong>${numSegments}</strong>
            </div>
            <div class="speaker-detail">
                <span>Confidence:</span>
                <strong>${confidence}%</strong>
            </div>
        `;
        
        container.appendChild(card);
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
    progressSection.classList.add('hidden');
}

// Download transcript
document.getElementById('downloadTranscript')?.addEventListener('click', () => {
    if (!currentResults || !currentResults.transcription) return;
    
    let text = 'TRANSCRIPTION\n';
    text += '='.repeat(60) + '\n\n';
    
    currentResults.transcription.forEach(seg => {
        const speaker = seg.speaker_id || 'UNKNOWN';
        const time = `[${formatTime(seg.start)} - ${formatTime(seg.end)}]`;
        text += `${speaker} ${time}: ${seg.text}\n\n`;
    });
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'transcription.txt';
    a.click();
    URL.revokeObjectURL(url);
});