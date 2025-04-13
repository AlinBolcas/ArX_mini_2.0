import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Meta (Facebook) API credentials from environment variables
meta_creds = {
    'access_token': os.getenv("META_KEY"),
    'user_id': os.getenv("INSTAGRAM_USER_ID")
}

class InstagramManager:
    def __init__(self, meta_creds):
        self.access_token = meta_creds['access_token']
        self.user_id = meta_creds['user_id']
        print("Initialized InstagramManager with provided credentials.")

    def trim_message(self, message, limit):
        return message if len(message) <= limit else message[:limit-3] + '...'

    def post_photo(self, image_url, caption, hashtags=[]):
        caption = self.trim_message(caption + ' ' + ' '.join(['#' + tag for tag in hashtags]), 2200)
        url = f"https://graph.facebook.com/v12.0/{self.user_id}/media"
        payload = {
            'image_url': image_url,
            'caption': caption,
            'access_token': self.access_token
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("Successfully uploaded the photo.")
            creation_id = response.json()['id']
            publish_url = f"https://graph.facebook.com/v12.0/{self.user_id}/media_publish"
            publish_payload = {
                'creation_id': creation_id,
                'access_token': self.access_token
            }
            publish_response = requests.post(publish_url, data=publish_payload)
            if publish_response.status_code == 200:
                print("Successfully posted the photo.")
            else:
                print(f"Error publishing photo: {publish_response.text}")
        else:
            print(f"Error uploading photo: {response.text}")

    def post_reel(self, video_url, caption, hashtags=[]):
        caption = self.trim_message(caption + ' ' + ' '.join(['#' + tag for tag in hashtags]), 2200)
        url = f"https://graph.facebook.com/v12.0/{self.user_id}/media"
        payload = {
            'media_type': 'VIDEO',
            'video_url': video_url,
            'caption': caption,
            'access_token': self.access_token
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("Successfully uploaded the reel.")
            creation_id = response.json()['id']
            publish_url = f"https://graph.facebook.com/v12.0/{self.user_id}/media_publish"
            publish_payload = {
                'creation_id': creation_id,
                'access_token': self.access_token
            }
            publish_response = requests.post(publish_url, data=publish_payload)
            if publish_response.status_code == 200:
                print("Successfully posted the reel.")
            else:
                print(f"Error publishing reel: {publish_response.text}")
        else:
            print(f"Error uploading reel: {response.text}")

if __name__ == "__main__":
    # Example usage
    manager = InstagramManager(meta_creds)
    manager.post_photo("https://example.com/photo.jpg", "Hello Instagram!", ["example", "test"])
    manager.post_reel("https://example.com/video.mp4", "Hello Instagram Reel!", ["example", "test"])
