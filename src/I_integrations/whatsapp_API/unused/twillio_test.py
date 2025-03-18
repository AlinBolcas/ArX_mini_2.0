import os
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables from .env file
load_dotenv()

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')

client = Client(account_sid, auth_token)

# Example: filter by "to" or "from" for SMS messages
my_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
print(f"Filtering messages sent to: {my_phone_number}")

messages = client.messages.list(
)

for message in messages:
    print(f"SID: {message.sid}")
    print(f"From: {message.from_}")
    print(f"To:   {message.to}")
    print(f"Body: {message.body}")
    print("-" * 40)