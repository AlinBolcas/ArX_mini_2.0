import os
import requests
import time
import urllib.parse
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

def search_and_download_images(query, num_images, download_folder="downloads"):
    """
    Searches for images using Google Custom Search API 
    and downloads them to the specified folder.
    """
    print(f"Searching for {num_images} images of '{query}'...")

    # Check if API credentials exist
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("Error: Missing Google API credentials in .env file.")
        print("Make sure GOOGLE_API_KEY and GOOGLE_CSE_ID are set.")
        return

    # Create downloads folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
        print(f"Created downloads folder: {download_folder}")

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': GOOGLE_CSE_ID,
        'key': GOOGLE_API_KEY,
        'searchType': 'image',
        'num': num_images  # Request the exact number of images needed
    }

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        search_results = response.json()

        # Check for API errors
        if 'error' in search_results:
            error = search_results['error']
            print(f"API Error: {error.get('message', 'Unknown error')}")
            if 'errors' in error:
                for err in error['errors']:
                    print(f"  - {err.get('message', '')}")
            return

        if 'items' not in search_results or not search_results['items']:
            print(f"No image results found for query: {query}")
            return

        print(f"Found {len(search_results['items'])} images. Downloading...")
        downloaded_count = 0
        
        for i, item in enumerate(search_results['items']):
            if downloaded_count >= num_images:
                break
                
            try:
                # Get image URL
                image_url = item['link']
                
                # Generate a safe filename
                filename = f"{query}_{i+1}_{int(time.time())}.jpg"
                # Remove any unsafe characters
                filename = ''.join(c for c in filename if c.isalnum() or c in '._- ')
                
                # Full path to save the image
                save_path = os.path.join(download_folder, filename)
                
                print(f"Downloading: {image_url}")
                
                # Download the image
                img_response = requests.get(image_url, stream=True, timeout=10)
                img_response.raise_for_status()
                
                # Save the image to disk
                with open(save_path, 'wb') as img_file:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        if chunk:
                            img_file.write(chunk)
                
                print(f"Saved to: {save_path}")
                downloaded_count += 1
                
                # Small delay between downloads
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"Error downloading image {image_url}: {e}")
            except Exception as e:
                print(f"Error processing image {i+1}: {e}")

        print(f"Successfully downloaded {downloaded_count} images to {download_folder}")

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Status Code: {e.response.status_code}")
             print(f"Response Body: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    search_and_download_images("blue dragon concept design", 15)