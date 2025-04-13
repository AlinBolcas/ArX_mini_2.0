import tweepy
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Twitter credentials from environment variables
twitter_creds = {
    'api_key': os.getenv("X_API_KEY"),
    'api_secret_key': os.getenv("X_API_KEY_SECRET"),
    'access_token': os.getenv("X_ACCESS_TOKEN"),
    'access_token_secret': os.getenv("X_ACCESS_TOKEN_SECRET"),
}

class SocialManager:
    def __init__(self, twitter_creds):
        self.twitter_creds = twitter_creds
        self.auth_twitter()

    def auth_twitter(self):
        self.auth = tweepy.OAuth1UserHandler(
            self.twitter_creds['api_key'],
            self.twitter_creds['api_secret_key']
        )
        try:
            redirect_url = self.auth.get_authorization_url()
            print(f"Visit this URL to authorize the app: {redirect_url}")
            verifier = input("Enter the verifier code from Twitter: ").strip()
            self.auth.get_access_token(verifier)
            self.api = tweepy.API(self.auth)
            print("Authenticated Twitter API.")
        except tweepy.TweepyException as e:
            print(f"Error during authentication: {e}")

    def x_post(self, message, hashtags=[]):
        try:
            tweet = self.trim_message(message + ' ' + ' '.join(['#' + tag for tag in hashtags]), 280)
            self.api.update_status(status=tweet)
            print("Successfully posted the tweet.")
        except Exception as e:
            print(f"Error posting tweet: {e}")

    def trim_message(self, message, limit):
        return message if len(message) <= limit else message[:limit-3] + '...'

if __name__ == "__main__":
    manager = SocialManager(twitter_creds)
    if hasattr(manager, 'api'):
        manager.x_post("Hello Twitter!", ["example", "test"])
    else:
        print("Twitter API not authenticated.")
