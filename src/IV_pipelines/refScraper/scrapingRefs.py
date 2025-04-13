import os
import requests
import openai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import urllib.request
import time


load_dotenv()

openai.api_key = os.getenv('openAIKey')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')


def scrape_google_images(query, num_images):
    """
    Uses Google Custom Search API to find and download images.
    More reliable than screen scraping methods.
    """
    downloaded_images = []
    search_url = "https://www.googleapis.com/customsearch/v1"
    
    # Check if API credentials exist
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("Error: Missing Google API credentials. Check your .env file.")
        print("Make sure GOOGLE_API_KEY and GOOGLE_CSE_ID are set.")
        return downloaded_images
    
    # Google CSE only returns 10 results per page, so we need to paginate
    for i in range(0, min(num_images, 100), 10):  # Google CSE has a limit of 100 results
        # Set up the search parameters
        params = {
            'q': query,
            'cx': GOOGLE_CSE_ID,
            'key': GOOGLE_API_KEY,
            'searchType': 'image',
            'start': i + 1 if i > 0 else 1,  # API uses 1-based indexing
            'num': min(10, num_images - i)  # Get up to 10 results at a time
        }
        
        try:
            # --- DEBUGGING: Print the key being used ---
            print(f"DEBUG: Using API Key: {GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-5:]}") 
            # --- End Debugging ---
            print(f"Searching for: {query} (page {i//10 + 1})")
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
                print("\nTips to fix:")
                print("1. Verify your API key is valid")
                print("2. Make sure your Custom Search Engine is set up for image search")
                print("3. Check if you've enabled the Custom Search API in Google Cloud Console")
                break
                
            if 'items' not in search_results:
                print(f"No image results found for query: {query}")
                if 'searchInformation' in search_results:
                    print(f"Total results: {search_results['searchInformation'].get('totalResults', '0')}")
                break
                
            print(f"Found {len(search_results['items'])} images")
            for item in search_results['items']:
                try:
                    image_url = item['link']
                    filename = sanitize_filename(f"{query}_{len(downloaded_images)}.jpg")
                    
                    # Download the image
                    print(f"Downloading: {image_url}")
                    download_path = os.path.join(os.getcwd(), filename)
                    urllib.request.urlretrieve(image_url, download_path)
                    
                    downloaded_images.append(download_path)
                    print(f"Downloaded image: {filename}")
                    
                    if len(downloaded_images) >= num_images:
                        break
                        
                    # Be nice to Google's servers
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error downloading image {image_url}: {e}")
                    continue
                    
            if len(downloaded_images) >= num_images:
                break
                
        except requests.exceptions.HTTPError as e:
            print(f"Error querying Google CSE: {e}")
            if e.response.status_code == 400:
                print("\nYour Custom Search Engine is not properly configured. Please make sure:")
                print("1. You've enabled Image Search in your CSE settings")
                print("2. Your CSE is set to 'Search the entire web'")
                print("3. Go to https://programmablesearchengine.google.com/cse/all to fix these settings")
            elif e.response.status_code == 403:
                print("\nAccess denied. Possible reasons:")
                print("1. API key doesn't have access to Custom Search API")
                print("2. You've exceeded your daily quota (100 queries per day on free tier)")
            break
        except Exception as e:
            print(f"Error querying Google CSE: {e}")
            break
            
    return downloaded_images

def expand_theme_gpt(theme, instruction, num_branches):
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a creative character artist assistant. I want you to provide a list of related keywords or topics for image collection. "
                "Ensure they are contextually related to the main theme or subject for Google searching purposes. "
                f"For example, if the theme is 'dragons', some related topics could be: "
                "'dragon anatomy', 'dragon concept art', 'dragon sketches', 'dragon scales', 'dragon drawings', 'dragon designs', and so on, but use what's most relevant to the theme instead of these examples exactly. "
                f"Specific Requirements: {instruction}. "
                f"Given these requirements, please provide ideal google search syntaxes for the theme provided. If instructions is '-', you're not constrained by them."
            )
        },
        {"role": "user", "content": f"Given the theme '{theme}', what are the essential related google search topics needed to inform the creation of a CGI project on the given theme?"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0.3,
    )
    
    expanded_themes = [item.strip() for item in response.choices[0].message['content'].split('\n') if item][:num_branches]
    print(f"Expanded themes for '{theme}': {expanded_themes}")
    return expanded_themes

def sanitize_filename(filename):
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename

def explore_images():
    theme = theme_entry.get()
    instruction = instruction_entry.get()
    num_branches = int(num_branches_entry.get())
    num_images = int(num_images_entry.get())
    file_location = filedialog.askdirectory(title="Select a Folder")

    theme_folder = os.path.join(file_location, theme)
    if not os.path.exists(theme_folder):
        os.makedirs(theme_folder)

    expanded_themes = [theme] + expand_theme_gpt(theme, instruction, num_branches - 1)
    for branch in expanded_themes:
        search_query = f"{theme} {branch}"
        img_paths = scrape_google_images(search_query, num_images)
        for img_path in img_paths:
            print(f"Saved image to: {img_path}")
            # Move the image to the desired directory
            new_path = os.path.join(theme_folder, os.path.basename(img_path))
            os.rename(img_path, new_path)

    messagebox.showinfo("Info", "Downloading complete!")

app = tk.Tk()
app.title("Image Scraper")

theme_label = tk.Label(app, text="Theme:")
theme_label.pack(pady=10)
theme_entry = tk.Entry(app)
theme_entry.pack(pady=10)

instruction_label = tk.Label(app, text="Instruction:")
instruction_label.pack(pady=10)
instruction_entry = tk.Entry(app)
instruction_entry.pack(pady=10)

num_branches_label = tk.Label(app, text="Number of Search Branches:")
num_branches_label.pack(pady=10)
num_branches_entry = tk.Entry(app)
num_branches_entry.pack(pady=10)

num_images_label = tk.Label(app, text="Number of Images per Search Branch:")
num_images_label.pack(pady=10)
num_images_entry = tk.Entry(app)
num_images_entry.pack(pady=10)

explore_button = tk.Button(app, text="Explore", command=explore_images)
explore_button.pack(pady=20)

app.mainloop()
