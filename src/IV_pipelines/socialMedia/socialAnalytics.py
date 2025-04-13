import tweepy
import requests
from googleapiclient.discovery import build
from google.oauth2 import service_account

import sys
from pathlib import Path

# Calculate the project root (which is three directories up from this file)
project_root = Path(__file__).resolve().parents[2]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))

# ARV-O MODULES IMPORT
from modules.utils.dataPlot import DataVisualizer


class SocialMediaAnalytics:
    def __init__(self, twitter_api, instagram_access_token, linkedin_access_token, youtube_service, pinterest_access_token, facebook_access_token):
        self.twitter_api = twitter_api
        self.instagram_access_token = instagram_access_token
        self.linkedin_access_token = linkedin_access_token
        self.youtube_service = youtube_service
        self.pinterest_access_token = pinterest_access_token
        self.facebook_access_token = facebook_access_token

    # Twitter
    def gather_twitter_data(self):
        data = self.twitter_api.me()._json
        return data

    # Instagram
    def gather_instagram_data(self):
        url = f"https://graph.instagram.com/me?fields=id,username,media_count,account_type&access_token={self.instagram_access_token}"
        response = requests.get(url)
        return response.json()

    # LinkedIn
    def gather_linkedin_data(self):
        url = "https://api.linkedin.com/v2/me"
        headers = {
            'Authorization': f"Bearer {self.linkedin_access_token}"
        }
        response = requests.get(url, headers=headers)
        return response.json()

    # YouTube
    def gather_youtube_data(self):
        request = self.youtube_service.channels().list(
            part="snippet,contentDetails,statistics",
            mine=True
        )
        response = request.execute()
        return response

    # Pinterest
    def gather_pinterest_data(self):
        url = f"https://api.pinterest.com/v3/pidgets/users/me/?access_token={self.pinterest_access_token}"
        response = requests.get(url)
        return response.json()

    # Facebook
    def gather_facebook_data(self):
        url = f"https://graph.facebook.com/me?fields=id,name,posts&access_token={self.facebook_access_token}"
        response = requests.get(url)
        return response.json()

    # Method to visualize data
    def visualize_data(self, data, chart_type="bar"):
        if chart_type == "bar":
            DataVisualizer.bar_chart(data)
        elif chart_type == "pie":
            DataVisualizer.pie_chart(data)
        elif chart_type == "line":
            DataVisualizer.line_chart(data)
        elif chart_type == "scatter":
            DataVisualizer.scatter_plot(data)

if __name__ == "__main__":
    # Example credentials for testing
    twitter_creds = {
        'api_key': 'your_api_key',
        'api_secret_key': 'your_api_secret_key',
        'access_token': 'your_access_token',
        'access_token_secret': 'your_access_token_secret'
    }

    instagram_creds = {
        'access_token': 'your_access_token',
        'user_id': 'your_user_id'
    }

    linkedin_creds = {
        'access_token': 'your_access_token'
    }

    youtube_creds = {
        'service_account_file': 'path_to_service_account.json'
    }

    pinterest_creds = {
        'access_token': 'your_access_token',
        'board_id': 'your_board_id'
    }

    facebook_creds = {
        'access_token': 'your_page_access_token',
        'page_id': 'your_page_id'
    }

    # Initialize APIs
    twitter_api = tweepy.API(tweepy.OAuthHandler(twitter_creds['api_key'], twitter_creds['api_secret_key']))
    twitter_api.set_access_token(twitter_creds['access_token'], twitter_creds['access_token_secret'])

    youtube_service = build('youtube', 'v3', credentials=service_account.Credentials.from_service_account_file(
        youtube_creds['service_account_file'], scopes=['https://www.googleapis.com/auth/youtube.force-ssl']))

    # Initialize SocialMediaAnalytics with APIs
    analytics = SocialMediaAnalytics(twitter_api, instagram_creds['access_token'], linkedin_creds['access_token'], 
                                     youtube_service, pinterest_creds['access_token'], facebook_creds['access_token'])

    # Gather and visualize data
    twitter_data = analytics.gather_twitter_data()
    instagram_data = analytics.gather_instagram_data()
    linkedin_data = analytics.gather_linkedin_data()
    youtube_data = analytics.gather_youtube_data()
    pinterest_data = analytics.gather_pinterest_data()
    facebook_data = analytics.gather_facebook_data()

    # Example visualization (customize as needed)
    analytics.visualize_data(twitter_data, "bar")
    analytics.visualize_data(instagram_data, "pie")
    analytics.visualize_data(linkedin_data, "line")
    analytics.visualize_data(youtube_data, "scatter")
    analytics.visualize_data(pinterest_data, "bar")
    analytics.visualize_data(facebook_data, "pie")
