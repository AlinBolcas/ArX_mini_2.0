# DOCS:
# X - TWITTER: 
# https://developer.x.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api
# https://github.com/tweepy/tweepy/blob/master/examples/API_v2/create_tweet.py

# INSTAGRAM:
# https://developers.facebook.com/docs/instagram-api/guides/content-publishing#reels-posts


import requests
# from googleapiclient.discovery import build
# from google.oauth2 import service_account
# from googleapiclient.http import MediaFileUpload
import threading
import json
import os
from datetime import datetime
from dotenv import load_dotenv

import sys
from pathlib import Path
# Calculate the project root (which is three directories up from this file)
project_root = Path(__file__).resolve().parents[2]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))

import tweepy
from flask import Flask, redirect, request, session, url_for

output_path = project_root / 'output'

WORKING_DIRECTORY = output_path

# ARV-O MODULES IMPORT
from modules.utils import utils

# Load environment variables
load_dotenv()

# Load Twitter credentials from environment variables
twitter_creds = {
    'api_key': os.getenv("X_API_KEY"),
    'api_secret_key': os.getenv("X_API_KEY_SECRET"),
    'bearer_token': os.getenv("X_BEARER"),
    'access_token': os.getenv("X_ACCESS_TOKEN"),
    'access_token_secret': os.getenv("X_ACCESS_TOKEN_SECRET"),
    'client_id': os.getenv("X_CLIENT_ID"),
    'client_secret': os.getenv("X_CLIENT_SECRET")
}


app = Flask(__name__)
app.secret_key = 'supersecretkey'
callback_url = 'http://127.0.0.1:5000/callback'

class SocialManager:
    def __init__(self, twitter_creds):
        self.twitter_creds = twitter_creds
        self.twitter_client = None
        print("Initialized SocialManager with provided credentials.")

    def auth_twitter(self):
        auth = tweepy.OAuth2UserHandler(
            client_id=self.twitter_creds['client_id'],
            client_secret=self.twitter_creds['client_secret'],
            redirect_uri=callback_url,
            scope=['tweet.read', 'tweet.write', 'users.read', 'offline.access']
        )
        auth_url = auth.get_authorization_url()
        session['request_token'] = auth.oauth.client.session.token
        print("Authenticated Twitter API.")
        return redirect(auth_url)

    def handle_callback(self):
        auth = tweepy.OAuth2UserHandler(
            client_id=self.twitter_creds['client_id'],
            client_secret=self.twitter_creds['client_secret'],
            redirect_uri=callback_url,
            scope=['tweet.read', 'tweet.write', 'users.read', 'offline.access']
        )
        token = auth.fetch_token(request.url)
        self.twitter_client = tweepy.Client(bearer_token=token['access_token'])
        session['access_token'] = token['access_token']
        print("Callback handled and token fetched.")
        return redirect(url_for('index'))

    def x_post(self, message, hashtags=[]):
        try:
            tweet = self.trim_message(message + ' ' + ' '.join(['#' + tag for tag in hashtags]), 280)
            response = self.twitter_client.create_tweet(text=tweet)
            print("Successfully posted the tweet.", response)
        except Exception as e:
            print(f"Error posting tweet: {e}")

    def trim_message(self, message, limit):
        return message if len(message) <= limit else message[:limit-3] + '...'

manager = SocialManager(twitter_creds)

@app.route('/')
def index():
    if 'access_token' in session:
        return 'Authenticated and ready to post!'
    return manager.auth_twitter()

@app.route('/callback')
def callback():
    return manager.handle_callback()

if __name__ == "__main__":
    app.run(debug=True)

# class SocialManager:
#     def __init__(self, twitter_creds, instagram_creds, linkedin_creds, youtube_creds, pinterest_creds, facebook_creds):
#         self.twitter_api = self.auth_twitter(twitter_creds)
#         self.instagram_access_token = instagram_creds['access_token']
#         self.instagram_user_id = instagram_creds['user_id']
#         self.linkedin_access_token = linkedin_creds['access_token']
#         self.youtube_service = self.auth_youtube(youtube_creds)
#         self.pinterest_access_token = pinterest_creds['access_token']
#         self.pinterest_board_id = pinterest_creds['board_id']
#         self.facebook_access_token = facebook_creds['access_token']
#         self.facebook_page_id = facebook_creds['page_id']
#         print("Initialized SocialManager with provided credentials.")

#     def auth_twitter(self, creds):
#         auth = tweepy.OAuthHandler(creds['api_key'], creds['api_secret_key'])
#         auth.set_access_token(creds['access_token'], creds['access_token_secret'])
#         print("Authenticated Twitter API.")
#         return tweepy.API(auth)

#     def auth_youtube(self, creds):
#         service_account_file = creds['service_account_file']
#         scopes = ['https://www.googleapis.com/auth/youtube.force-ssl']
#         credentials = service_account.Credentials.from_service_account_file(service_account_file, scopes=scopes)
#         print("Authenticated YouTube API.")
#         return build('youtube', 'v3', credentials=credentials)

#     def trim_message(self, message, limit):
#         return message if len(message) <= limit else message[:limit-3] + '...'

#     def x_post(self, message, hashtags=[]):
#         try:
#             tweet = self.trim_message(message + ' ' + ' '.join(['#' + tag for tag in hashtags]), 280)
#             self.twitter_api.update_status(status=tweet)
#             print("Successfully posted the tweet.")
#         except Exception as e:
#             print(f"Error posting tweet: {e}")

#     def instagram_post(self, image_urls, caption, hashtags=[], music=None):
#         caption = self.trim_message(caption + ' ' + ' '.join(['#' + tag for tag in hashtags]), 2200)
#         for image_url in image_urls:
#             url = f"https://graph.facebook.com/v12.0/{self.instagram_user_id}/media"
#             payload = {
#                 'image_url': image_url,
#                 'caption': caption,
#                 'access_token': self.instagram_access_token
#             }
#             if music:
#                 payload['music'] = music
#             response = requests.post(url, data=payload)
#             if response.status_code == 200:
#                 print("Successfully posted the photo.")
#             else:
#                 print(f"Error posting photo: {response.text}")

#     def instagram_reel(self, video_url, caption, hashtags=[]):
#         caption = self.trim_message(caption + ' ' + ' '.join(['#' + tag for tag in hashtags]), 2200)
#         url = f"https://graph.facebook.com/v12.0/{self.instagram_user_id}/media"
#         payload = {
#             'media_type': 'REEL',
#             'video_url': video_url,
#             'caption': caption,
#             'access_token': self.instagram_access_token
#         }
#         response = requests.post(url, data=payload)
#         if response.status_code == 200:
#             print("Successfully posted the reel.")
#         else:
#             print(f"Error posting reel: {response.text}")

#     def instagram_story(self, image_url, caption):
#         caption = self.trim_message(caption, 2200)
#         url = f"https://graph.facebook.com/v12.0/{self.instagram_user_id}/media"
#         payload = {
#             'media_type': 'STORY',
#             'image_url': image_url,
#             'caption': caption,
#             'access_token': self.instagram_access_token
#         }
#         response = requests.post(url, data=payload)
#         if response.status_code == 200:
#             print("Successfully posted the story.")
#         else:
#             print(f"Error posting story: {response.text}")

#     def instagram_message(self, recipient_id, message):
#         url = f"https://graph.facebook.com/v12.0/{self.instagram_user_id}/messages"
#         payload = {
#             'recipient': {'id': recipient_id},
#             'message': {'text': message},
#             'access_token': self.instagram_access_token
#         }
#         response = requests.post(url, data=payload)
#         if response.status_code == 200:
#             print("Successfully sent the message.")
#         else:
#             print(f"Error sending message: {response.text}")

#     def linkedin_post(self, message, media_url=None, media_type=None, hashtags=[]):
#         url = 'https://api.linkedin.com/v2/ugcPosts'
#         headers = {
#             'Authorization': f'Bearer {self.linkedin_access_token}',
#             'Content-Type': 'application/json'
#         }
#         message = self.trim_message(message + ' ' + ' '.join(['#' + tag for tag in hashtags]), 3000)
#         payload = {
#             "author": "urn:li:person:your_person_id",
#             "lifecycleState": "PUBLISHED",
#             "specificContent": {
#                 "com.linkedin.ugc.ShareContent": {
#                     "shareCommentary": {
#                         "text": message
#                     },
#                     "shareMediaCategory": "ARTICLE" if media_url else "NONE",
#                     "media": [
#                         {
#                             "status": "READY",
#                             "description": {
#                                 "text": message
#                             },
#                             "originalUrl": media_url,
#                             "title": {
#                                 "text": "LinkedIn Post Media"
#                             },
#                             "mediaType": media_type
#                         }
#                     ] if media_url else []
#                 }
#             },
#             "visibility": {
#                 "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
#             }
#         }
#         response = requests.post(url, headers=headers, json=payload)
#         if response.status_code == 201:
#             print("Successfully posted the update.")
#         else:
#             print(f"Error posting update: {response.text}")

#     def youtube_video(self, file, title, description, tags=[]):
#         body = {
#             'snippet': {
#                 'title': title,
#                 'description': description,
#                 'tags': tags,
#                 'categoryId': '22'
#             },
#             'status': {
#                 'privacyStatus': 'public'
#             }
#         }
#         media = MediaFileUpload(file, chunksize=-1, resumable=True)
#         request = self.youtube_service.videos().insert(part="snippet,status", body=body, media_body=media)
#         response = request.execute()
#         print(f"Successfully uploaded the video: {response}")

#     def pinterest_pin(self, image_url, note, link, hashtags=[]):
#         url = 'https://api.pinterest.com/v1/pins/'
#         note = self.trim_message(note + ' ' + ' '.join(['#' + tag for tag in hashtags]), 500)
#         payload = {
#             'board': self.pinterest_board_id,
#             'note': note,
#             'link': link,
#             'image_url': image_url
#         }
#         headers = {
#             'Authorization': f'Bearer {self.pinterest_access_token}',
#         }
#         response = requests.post(url, headers=headers, data=payload)
#         if response.status_code == 201:
#             print("Successfully created the pin.")
#         else:
#             print(f"Error creating pin: {response.text}")

#     def facebook_post(self, message, hashtags=[]):
#         url = f"https://graph.facebook.com/v12.0/{self.facebook_page_id}/feed"
#         message = self.trim_message(message + ' ' + ' '.join(['#' + tag for tag in hashtags]), 63206)
#         payload = {
#             'message': message,
#             'access_token': self.facebook_access_token
#         }
#         response = requests.post(url, data=payload)
#         if response.status_code == 200:
#             print("Successfully posted the message.")
#         else:
#             print(f"Error posting message: {response.text}")

#     def facebook_message(self, recipient_id, message):
#         url = f"https://graph.facebook.com/v12.0/{self.facebook_page_id}/messages"
#         payload = {
#             'recipient': {'id': recipient_id},
#             'message': {'text': message},
#             'access_token': self.facebook_access_token
#         }
#         response = requests.post(url, data=payload)
#         if response.status_code == 200:
#             print("Successfully sent the message.")
#         else:
#             print(f"Error sending message: {response.text}")

#     def threads_post(self, message, hashtags=[]):
#         print("Threads API is not available yet.")

# class SocialScheduler:
#     def __init__(self, manager):
#         self.manager = manager
#         self.jobs = []
#         self.schedule_file = 'schedules.json'
#         if not os.path.exists(self.schedule_file):
#             with open(self.schedule_file, 'w') as f:
#                 json.dump([], f)
#         print("Initialized SocialScheduler and loaded schedules.")

#     def schedule_post(self, platform, method, delay, *args):
#         job_time = datetime.now().timestamp() + delay
#         job = {
#             'platform': platform,
#             'method': method,
#             'time': job_time,
#             'args': args
#         }
#         self.jobs.append(job)
#         self._save_schedule()
#         threading.Timer(delay, getattr(self.manager, method), args).start()
#         print(f"Scheduled {platform} post in {delay} seconds.")

#     def _save_schedule(self):
#         with open(self.schedule_file, 'w') as f:
#             json.dump(self.jobs, f)
#         print("Saved schedule to file.")

#     def load_schedules(self):
#         with open(self.schedule_file, 'r') as f:
#             self.jobs = json.load(f)
#         for job in self.jobs:
#             delay = job['time'] - datetime.now().timestamp()
#             if delay > 0:
#                 threading.Timer(delay, getattr(self.manager, job['method']), job['args']).start()
#             else:
#                 getattr(self.manager, job['method'])(*job['args'])
#         print("Loaded schedules from file.")

# if __name__ == "__main__":
#     # Example credentials for testing
#     twitter_creds = {
#         'api_key': 'your_api_key',
#         'api_secret_key': 'your_api_secret_key',
#         'access_token': 'your_access_token',
#         'access_token_secret': 'your_access_token_secret'
#     }
    
#     instagram_creds = {
#         'access_token': 'your_access_token',
#         'user_id': 'your_user_id'
#     }
    
#     linkedin_creds = {
#         'access_token': 'your_access_token'
#     }
    
#     youtube_creds = {
#         'service_account_file': 'path_to_service_account.json'
#     }
    
#     pinterest_creds = {
#         'access_token': 'your_access_token',
#         'board_id': 'your_board_id'
#     }
    
#     facebook_creds = {
#         'access_token': 'your_page_access_token',
#         'page_id': 'your_page_id'
#     }
    
#     manager = SocialManager(twitter_creds, instagram_creds, linkedin_creds, youtube_creds, pinterest_creds, facebook_creds)
#     scheduler = SocialScheduler(manager)
    
#     # Post examples
#     manager.x_post("Hello Twitter!", ["example", "test"])
#     manager.instagram_post(["https://example.com/photo1.jpg", "https://example.com/photo2.jpg"], "Hello Instagram!", ["example", "test"], "Your Favorite Music")
#     manager.linkedin_post("Hello LinkedIn!", "https://example.com/image.jpg", "IMAGE", ["example", "test"])
#     manager.youtube_video("video.mp4", "Hello YouTube", "This is a test video.", ["example", "test"])
#     manager.pinterest_pin("https://example.com/photo.jpg", "Hello Pinterest!", "https://example.com", ["example", "test"])
#     manager.facebook_post("Hello Facebook!", ["example", "test"])
    
#     # Schedule posts with a delay (e.g., 5 seconds for testing)
#     scheduler.schedule_post('Twitter', 'x_post', 5, "Scheduled Twitter post", ["scheduled", "test"])
#     scheduler.schedule_post('Instagram', 'instagram_post', 5, ["https://example.com/photo1.jpg"], "Scheduled Instagram post", ["scheduled", "test"])
#     scheduler.schedule_post('LinkedIn', 'linkedin_post', 5, "Scheduled LinkedIn post", "https://example.com/image.jpg", "IMAGE", ["scheduled", "test"])
#     scheduler.schedule_post('YouTube', 'youtube_video', 5, "video.mp4", "Scheduled YouTube video", "This is a scheduled test video.", ["scheduled", "test"])
#     scheduler.schedule_post('Pinterest', 'pinterest_pin', 5, "https://example.com/photo.jpg", "Scheduled Pinterest pin", "https://example.com", ["scheduled", "test"])
#     scheduler.schedule_post('Facebook', 'facebook_post', 5, "Scheduled Facebook post", ["scheduled", "test"])
