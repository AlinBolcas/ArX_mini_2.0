let isProcessing = false;
let pollInterval = null;

async function processSong() {
    const input = document.getElementById('song-input');
    const btn = document.getElementById('process-btn');
    const status = document.getElementById('status-message');
    
    if (!input.value) return;
    
    // Get the application prefix 
    const prefix = window.APP_CONFIG ? APP_CONFIG.prefix : '';
    
    btn.disabled = true;
    isProcessing = true;
    status.innerHTML = '<div class="info">Please wait about 1 minute, your song is processing...</div>';
    
    try {
        const response = await fetch(`${prefix}/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                song: input.value
            })
        });

        // Start polling for updates
        pollStatus();
        
    } catch (error) {
        console.log('Error:', error);
        // Even if there's an error, try polling anyway
        pollStatus();
    }
}

function pollStatus() {
    // Clear any existing interval
    if (pollInterval) {
        clearInterval(pollInterval);
    }
    
    // Set up a new polling interval
    pollInterval = setInterval(() => {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                // Update artwork count
                document.getElementById('artwork-count').textContent = data.artwork_count;
                document.getElementById('sidebar-artwork-count').textContent = data.artwork_count;
                
                // Update gallery if we have artworks
                if (data.artworks && data.artworks.length > 0) {
                    updateGallery(data.artworks);
                    // Update recent song
                    document.getElementById('recent-song').textContent = data.artworks[0].song_title;
                }
                
                // Stop polling if processing is complete
                if (!data.processing) {
                    isProcessing = false;
                    document.getElementById('process-btn').disabled = false;
                    document.getElementById('status-message').innerHTML = '';
                    clearInterval(pollInterval);
                    pollInterval = null;
                }
            })
            .catch(error => {
                console.log('Polling error:', error);
                // Keep polling even if there's an error
            });
    }, 2000); // Poll every 2 seconds
}

function updateGallery(artworks) {
    const gallery = document.getElementById('artwork-gallery');
    gallery.innerHTML = ''; // Clear existing artworks
    
    artworks.forEach(artwork => {
        const artworkElement = createArtworkElement(artwork);
        gallery.appendChild(artworkElement);
    });
}

function createArtworkElement(artwork) {
    const article = document.createElement('article');
    article.className = 'artwork';
    
    // Create image container
    const imgContainer = document.createElement('div');
    imgContainer.className = 'artwork-image';
    
    // Create and set image
    const img = document.createElement('img');
    img.src = `/output/${artwork.image_path}`;
    img.alt = artwork.song_title;
    imgContainer.appendChild(img);
    
    // Create details container
    const details = document.createElement('div');
    details.className = 'artwork-details';
    
    // Add expand button
    const expandBtn = document.createElement('button');
    expandBtn.className = 'expand-btn';
    expandBtn.textContent = 'Show Details';
    expandBtn.onclick = () => toggleDetails(details);
    
    // Create details content
    const detailsContent = document.createElement('div');
    detailsContent.className = 'details-content';
    detailsContent.style.display = 'none';
    
    // Add song title and artist
    const title = document.createElement('h3');
    title.textContent = `${artwork.song_title} by ${artwork.artist}`;
    detailsContent.appendChild(title);
    
    // Add audio player if available
    if (artwork.audio_path) {
        const audio = document.createElement('audio');
        audio.controls = true;
        audio.src = `/output/${artwork.audio_path}`;
        detailsContent.appendChild(audio);
    }
    
    // Add interpretation
    if (artwork.interpretation) {
        const interpretation = document.createElement('div');
        interpretation.className = 'interpretation';
        interpretation.innerHTML = `<h4>Analysis</h4><p>${artwork.interpretation}</p>`;
        detailsContent.appendChild(interpretation);
    }
    
    // Add prompt
    if (artwork.prompt) {
        const prompt = document.createElement('div');
        prompt.className = 'prompt';
        prompt.innerHTML = `<h4>Image Prompt</h4><pre>${artwork.prompt}</pre>`;
        detailsContent.appendChild(prompt);
    }
    
    // Add creation time
    const time = document.createElement('p');
    time.className = 'creation-time';
    time.textContent = `Created: ${artwork.human_time}`;
    detailsContent.appendChild(time);
    
    // Assemble the details section
    details.appendChild(expandBtn);
    details.appendChild(detailsContent);
    
    // Assemble the article
    article.appendChild(imgContainer);
    article.appendChild(details);
    
    return article;
}

function toggleDetails(detailsElement) {
    const content = detailsElement.querySelector('.details-content');
    const button = detailsElement.querySelector('.expand-btn');
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        button.textContent = 'Hide Details';
    } else {
        content.style.display = 'none';
        button.textContent = 'Show Details';
    }
}

// Load initial artworks when page loads
window.addEventListener('load', async () => {
    try {
        // Get artworks as before
        const response = await fetch('/status');
        const data = await response.json();
        if (data.artworks && data.artworks.length > 0) {
            updateGallery(data.artworks);
        }
    } catch (error) {
        console.error('Failed to load initial data:', error);
    }
});

// Add event listener for Enter key on the input field
document.getElementById('song-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        document.getElementById('process-btn').click();
    }
});

// Load user data on page load
window.addEventListener('load', function() {
    fetch('/get-user-data')
        .then(response => response.json())
        .then(data => {
            // Fill in API keys if available
            if (data.api_keys) {
                document.getElementById('openai-key').value = data.api_keys.openai_key || '';
                document.getElementById('replicate-key').value = data.api_keys.replicate_key || '';
            }
            
            // Fill in custom prompts if available
            if (data.custom_prompts) {
                document.getElementById('music-system-prompt').value = data.custom_prompts.music_analysis_system_prompt || '';
                document.getElementById('music-user-prompt').value = data.custom_prompts.music_analysis_user_prompt || '';
                document.getElementById('image-system-prompt').value = data.custom_prompts.image_prompt_system_prompt || '';
                document.getElementById('image-user-prompt').value = data.custom_prompts.image_prompt_user_prompt || '';
            }
        });
});