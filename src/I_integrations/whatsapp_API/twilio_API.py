import os
import json
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from twilio.twiml.messaging_response import MessagingResponse
from typing import Dict, List, Optional, Union
import logging
from dotenv import load_dotenv
from datetime import datetime

logger = logging.getLogger(__name__)

class TwilioAPIError(Exception):
    """Custom exception for Twilio API errors"""
    pass

class TwilioAPI:
    """
    Twilio API wrapper for WhatsApp messaging
    Docs: https://www.twilio.com/docs/whatsapp/api
    """
    
    def __init__(self):
        load_dotenv()
        
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.whatsapp_number = "+447427805145"
        
        if not all([self.account_sid, self.auth_token, self.whatsapp_number]):
            raise ValueError("Missing Twilio credentials")
            
        try:
            self.client = Client(self.account_sid, self.auth_token)
        except Exception as e:
            raise TwilioAPIError(f"Failed to initialize Twilio client: {str(e)}")

    def _format_number(self, number: str) -> str:
        """Format number for WhatsApp"""
        if not number.startswith('whatsapp:'):
            return f'whatsapp:{number}'
        return number

    def _convert_message_to_dict(self, message) -> Dict:
        """Convert Twilio message instance to dictionary"""
        return {
            'sid': message.sid,
            'status': message.status,
            'direction': message.direction,
            'body': message.body,
            'error_message': getattr(message, 'error_message', None),
            'date_created': str(message.date_created),
            'date_sent': str(message.date_sent) if message.date_sent else None,
            'media_url': message.media_url[0] if hasattr(message, 'media_url') and message.media_url else None
        }

    def send_message(self, to: str, message: str) -> Dict:
        """Send a WhatsApp text message"""
        try:
            message_instance = self.client.messages.create(
                from_=self._format_number(self.whatsapp_number),
                to=self._format_number(to),
                body=message
            )
            return self._convert_message_to_dict(message_instance)
        except TwilioRestException as e:
            raise TwilioAPIError(f"Failed to send message: {str(e)}")

    def send_image(self, to: str, image_url: str, caption: Optional[str] = None) -> Dict:
        """Send an image with optional caption via WhatsApp"""
        try:
            message_instance = self.client.messages.create(
                from_=self._format_number(self.whatsapp_number),
                to=self._format_number(to),
                media_url=[image_url],
                body=caption
            )
            return self._convert_message_to_dict(message_instance)
        except TwilioRestException as e:
            raise TwilioAPIError(f"Failed to send image: {str(e)}")

    def send_template(self, to: str, template_id: str = None, variables: Optional[Dict[str, str]] = None) -> Dict:
        """Send a WhatsApp template message"""
        try:
            message_instance = self.client.messages.create(
                from_=self._format_number(self.whatsapp_number),
                to=self._format_number(to),
                content_sid=template_id,
                content_variables=json.dumps(variables) if variables else None
            )
            return self._convert_message_to_dict(message_instance)
        except TwilioRestException as e:
            raise TwilioAPIError(f"Failed to send template: {str(e)}")

    def get_message_history(self, limit: int = 20, to: Optional[str] = None) -> List[Dict]:
        """Get WhatsApp message history"""
        try:
            filters = {'limit': limit}
            if to:
                filters['to'] = self._format_number(to)

            messages = self.client.messages.list(**filters)
            return [self._convert_message_to_dict(msg) for msg in messages]
        except TwilioRestException as e:
            raise TwilioAPIError(f"Failed to get message history: {str(e)}")

    def get_account_info(self) -> Dict:
        """Get account information including balance and usage"""
        try:
            # Get account details
            account = self.client.api.accounts(self.account_sid).fetch()
            
            # Get today's usage records for WhatsApp
            today = datetime.now().strftime('%Y-%m-%d')
            usage_records = self.client.usage.records.daily.list(
                category='wireless',  # Changed from 'messages' to 'wireless'
                start_date=today,
                end_date=today
            )

            # Calculate totals
            total_messages = 0
            total_cost = 0.0
            
            for record in usage_records:
                # Convert string count to int before adding
                total_messages += int(record.count) if record.count else 0
                total_cost += float(record.price or 0)

            return {
                'account_name': account.friendly_name,
                'status': account.status,
                'type': account.type,
                'balance': None,  # Twilio doesn't provide direct balance access
                'today_messages': total_messages,
                'today_cost': total_cost
            }
        except TwilioRestException as e:
            raise TwilioAPIError(f"Failed to get account info: {str(e)}")

    def calculate_message_delay(self, message: str) -> float:
        """
        Calculate appropriate delay for message based on length and content.
        Returns delay in seconds.
        
        Rules:
        - Base typing speed: ~40 words per minute
        - Additional time for emojis and special characters
        - Minimum delay: 1.5 seconds
        - Maximum delay: 8 seconds
        """
        # Count words and special characters
        words = len(message.split())
        special_chars = len([c for c in message if not c.isalnum() and not c.isspace()])
        
        # Base calculation: 60 seconds / 40 words = 1.5 seconds per word
        base_delay = words * 0.6
        
        # Add time for special characters (0.3s each)
        emoji_delay = special_chars * 0.2
        
        # Calculate total delay
        total_delay = base_delay + emoji_delay
        
        # Clamp between minimum and maximum
        return max(1.0, min(total_delay, 7.0)) 