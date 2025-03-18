import os
import requests
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class WhatsAppAPI:
    """
    WhatsApp Cloud API integration class for ArX system.
    Handles API calls to WhatsApp's Cloud API v18.0
    """
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('META_ACCESS_TOKEN')
        self.app_id = os.getenv('META_APP_ID')
        self.app_secret = os.getenv('META_APP_SECRET')
        self.phone_number_id = os.getenv('META_PHONE_ID')
        self.base_url = "https://graph.facebook.com/v18.0"
        self.version = "v18.0"
        
        if not all([self.api_key, self.app_id, self.app_secret, self.phone_number_id]):
            raise ValueError(
                "Missing required environment variables. "
                "Please ensure META_ACCESS_TOKEN, META_APP_ID, META_APP_SECRET, "
                "and META_PHONE_ID are set in .env"
            )
        
        # Initialize token info
        self.token_expires_at = None
        self.refresh_token = None

    def _refresh_access_token(self) -> None:
        """
        Refresh the access token using app credentials
        """
        try:
            url = f"{self.base_url}/oauth/access_token"
            params = {
                'client_id': self.app_id,
                'client_secret': self.app_secret,
                'grant_type': 'client_credentials',
                'scope': 'whatsapp_business_messaging'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            token_data = response.json()
            self.api_key = token_data['access_token']
            # Convert expires_in (seconds) to timestamp
            self.token_expires_at = datetime.now().timestamp() + token_data.get('expires_in', 3600)
            
            logger.info("Access token refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {str(e)}")
            raise WhatsAppAPIError("Token refresh failed") from e

    def _is_token_expired(self) -> bool:
        """Check if the current token is expired or about to expire"""
        if not self.token_expires_at:
            return True
        
        # Consider token expired if less than 5 minutes remaining
        return datetime.now().timestamp() >= (self.token_expires_at - 300)

    def send_text_message(self, to: str, message: str, preview_url: bool = True) -> Dict:
        """
        Send a text message with optional link preview
        
        Args:
            to: Recipient phone number
            message: Message text (max 4096 chars). Can include URLs
            preview_url: Whether to show link preview for URLs
            
        Reference: https://developers.facebook.com/docs/whatsapp/cloud-api/messages/text-messages
        """
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {
                "preview_url": preview_url,
                "body": message
            }
        }
        return self._make_request('POST', f'/{self.phone_number_id}/messages', json=payload)

    def send_template_message(self, to: str, template_name: str, language: str = "en", components: Optional[List[Dict]] = None) -> Dict:
        """
        Send a template message
        
        Args:
            to: Recipient phone number
            template_name: Name of approved template
            language: Template language code
            components: Optional template components (variables)
        """
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {
                    "code": language
                }
            }
        }
        
        if components:
            payload["template"]["components"] = components
            
        return self._make_request('POST', f'/{self.phone_number_id}/messages', json=payload)

    def send_image(self, 
                   to: str, 
                   image_url: Optional[str] = None,
                   image_id: Optional[str] = None,
                   caption: Optional[str] = None) -> Dict:
        """
        Send an image message using either uploaded media ID or URL
        
        Args:
            to: Recipient phone number
            image_url: URL of publicly accessible image
            image_id: ID of previously uploaded image (recommended)
            caption: Optional image caption (max 1024 chars)
        
        Note: Must provide either image_id or image_url
        Supported formats: JPEG, PNG (max 5MB)
        Reference: https://developers.facebook.com/docs/whatsapp/cloud-api/messages/image-messages
        """
        if not image_id and not image_url:
            raise ValueError("Must provide either image_id or image_url")
        
        image_data = {}
        if image_id:
            image_data["id"] = image_id
        else:
            image_data["link"] = image_url
        
        if caption:
            if len(caption) > 1024:
                raise ValueError("Caption must not exceed 1024 characters")
            image_data["caption"] = caption
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "image",
            "image": image_data
        }
        return self._make_request('POST', f'/{self.phone_number_id}/messages', json=payload)

    def send_document(self, to: str, document_url: str, caption: Optional[str] = None, filename: Optional[str] = None) -> Dict:
        """Send a document message"""
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "document",
            "document": {
                "link": document_url,
                **({"caption": caption} if caption else {}),
                **({"filename": filename} if filename else {})
            }
        }
        return self._make_request('POST', f'/{self.phone_number_id}/messages', json=payload)

    def send_interactive_buttons(self, to: str, message: str, buttons: List[Dict]) -> Dict:
        """
        Send interactive buttons message
        
        Args:
            to: Recipient phone number
            message: Message body
            buttons: List of button objects [{"type": "reply", "reply": {"id": "id1", "title": "Button 1"}}]
        """
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": message},
                "action": {
                    "buttons": buttons
                }
            }
        }
        return self._make_request('POST', f'/{self.phone_number_id}/messages', json=payload)

    def send_list_message(self, to: str, message: str, sections: List[Dict], button_text: str = "Select an option") -> Dict:
        """
        Send a list message with sections and options
        
        Args:
            to: Recipient phone number
            message: Message body
            sections: List of section objects with title and rows
            button_text: Text for the list button
        """
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "body": {"text": message},
                "action": {
                    "button": button_text,
                    "sections": sections
                }
            }
        }
        return self._make_request('POST', f'/{self.phone_number_id}/messages', json=payload)

    def send_location(self, 
                     to: str, 
                     latitude: float, 
                     longitude: float, 
                     name: Optional[str] = None, 
                     address: Optional[str] = None) -> Dict:
        """
        Send a location with optional name and address
        
        Args:
            to: Recipient phone number
            latitude: Location latitude in decimal degrees
            longitude: Location longitude in decimal degrees
            name: Optional location name
            address: Optional location address
            
        Reference: https://developers.facebook.com/docs/whatsapp/cloud-api/messages/location-messages
        """
        location_data = {
            "latitude": str(latitude),
            "longitude": str(longitude)
        }
        
        if name:
            location_data["name"] = name
        if address:
            location_data["address"] = address
            
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "location",
            "location": location_data
        }
        return self._make_request('POST', f'/{self.phone_number_id}/messages', json=payload)

    def mark_message_as_read(self, message_id: str) -> Dict:
        """
        Mark a message as read
        
        Args:
            message_id: ID of an incoming message to mark as read
            
        Note: Can only mark incoming messages as read
        """
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        return self._make_request('POST', f'/{self.phone_number_id}/messages/mark_as_read', json=payload)

    def get_business_profile(self) -> Dict:
        """Get the WhatsApp Business Profile information"""
        return self._make_request('GET', f'/{self.phone_number_id}/whatsapp_business_profile')

    def update_business_profile(self, profile_data: Dict) -> Dict:
        """
        Update the WhatsApp Business Profile
        
        Args:
            profile_data: Dict containing profile fields:
                - about: Business description
                - address: Business address
                - description: Business description
                - email: Business email
                - websites: List of business websites
                - profile_picture_url: URL to profile picture
        """
        payload = {
            "messaging_product": "whatsapp",
            **profile_data
        }
        return self._make_request('POST', f'/{self.phone_number_id}/whatsapp_business_profile', json=payload)

    def get_media_url(self, media_id: str) -> Dict:
        """Get the URL for a media file"""
        return self._make_request('GET', f'/{media_id}')

    def upload_media(self, file_path: str) -> Dict:
        """Upload media to WhatsApp servers"""
        with open(file_path, 'rb') as file:
            files = {
                'file': (os.path.basename(file_path), file, 'application/octet-stream')
            }
            return self._make_request('POST', f'/{self.phone_number_id}/media', files=files)

    def get_message_templates(self) -> List[Dict]:
        """Get list of available message templates"""
        return self._make_request('GET', f'/{self.version}/business_account/message_templates')

    def validate_template(self, template_name: str, language: str = "en") -> bool:
        """Validate if template exists and is approved"""
        try:
            templates = self.get_message_templates()
            for template in templates.get('data', []):
                if (template.get('name') == template_name and 
                    template.get('language') == language and 
                    template.get('status') == 'APPROVED'):
                    return True
            return False
        except WhatsAppAPIError:
            # If templates can't be fetched, assume template is valid
            # and let the send_template_message handle any errors
            return True

    def send_reaction(self, to: str, message_id: str, emoji: str) -> Dict:
        """
        Send a reaction emoji to a message
        
        Args:
            to: Recipient phone number
            message_id: ID of message to react to
            emoji: Unicode emoji or escape sequence
            
        Note: Message must be less than 30 days old
        Reference: https://developers.facebook.com/docs/whatsapp/cloud-api/messages/reaction-messages
        """
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "reaction",
            "reaction": {
                "message_id": message_id,
                "emoji": emoji
            }
        }
        return self._make_request('POST', f'/{self.phone_number_id}/messages', json=payload)

    def validate_media_size(self, file_path: str, max_size_mb: int = 5) -> bool:
        """
        Validate if media file size is within WhatsApp limits
        
        Args:
            file_path: Path to media file
            max_size_mb: Maximum allowed size in MB
        """
        max_size_bytes = max_size_mb * 1024 * 1024
        file_size = os.path.getsize(file_path)
        return file_size <= max_size_bytes

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Make authenticated request to WhatsApp Cloud API with automatic token refresh
        """
        # Check if token needs refresh
        if self._is_token_expired():
            self._refresh_access_token()
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Handle file uploads differently
        if 'files' in kwargs:
            headers.pop('Content-Type', None)
        
        url = f"{self.base_url}{endpoint}"
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.request(method, url, headers=headers, **kwargs)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_details = e.response.json()
                        if 'error' in error_details:
                            error = error_details['error']
                            error_code = error.get('code', 0)
                            
                            # If token error and not last attempt, refresh and retry
                            if error_code in [190, 102] and attempt < max_retries - 1:
                                logger.info("Token expired, refreshing...")
                                self._refresh_access_token()
                                headers['Authorization'] = f'Bearer {self.api_key}'
                                continue
                                
                            error_msg = f"WhatsApp API Error: ({error_code}) {error.get('message', 'Unknown error')}"
                            if 'error_data' in error:
                                error_msg += f"\nDetails: {error['error_data'].get('details', '')}"
                    except ValueError:
                        pass
                
                # If we're here on last attempt, raise the error
                if attempt == max_retries - 1:
                    logger.error(error_msg)
                    raise WhatsAppAPIError(error_msg) from e

class WhatsAppAPIError(Exception):
    """Custom exception for WhatsApp API errors"""
    pass 