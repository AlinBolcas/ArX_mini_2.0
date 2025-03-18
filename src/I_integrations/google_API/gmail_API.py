import os
import sys
import base64
from pathlib import Path
from typing import List, Dict, Optional, Union
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
import mimetypes

# Add project root to PYTHONPATH
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import base class using FileFinder
GoogleBaseAPI = finder.get_class('google_base_API.py', 'GoogleBaseAPI')

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class GmailWrapper(GoogleBaseAPI):
    """Wrapper for Gmail API operations"""
    
    # Define scopes based on needed functionality
    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",        # Basic read operations
        "https://www.googleapis.com/auth/gmail.send",           # Send emails
        "https://www.googleapis.com/auth/gmail.compose",        # Create/send/drafts
        "https://www.googleapis.com/auth/gmail.modify",         # All read/write operations
        "https://www.googleapis.com/auth/gmail.labels",         # Manage labels
        "https://www.googleapis.com/auth/gmail.settings.basic", # Manage settings
    ]
    
    def __init__(self):
        """Initialize Gmail API service"""
        super().__init__(scopes=['https://www.googleapis.com/auth/gmail.modify'])
        self.finder = FileFinder()
        self.credentials_path = self.finder.find_file('credentials.json')
        if not self.credentials_path:
            raise FileNotFoundError("credentials.json not found in project directory")
        
        self.token_path = self.credentials_path.parent / 'token.json'
        self.service = self._init_service()
    
    def _init_service(self):
        """Get authenticated Gmail API service"""
        creds = self._get_credentials()
        
        return build("gmail", "v1", credentials=creds)

    # Message Operations
    def list_messages(self, query: str = "", max_results: int = 10) -> List[Dict]:
        """List messages matching the specified query"""
        try:
            results = self.service.users().messages().list(
                userId="me", q=query, maxResults=max_results
            ).execute()
            
            messages = results.get("messages", [])
            return [self.get_message(msg["id"]) for msg in messages]
        except HttpError as error:
            print(f"Error listing messages: {error}")
            return []

    def get_message(self, message_id: str) -> Dict:
        """Get a specific message by ID"""
        try:
            return self.service.users().messages().get(
                userId="me", id=message_id, format="full"
            ).execute()
        except HttpError as error:
            print(f"Error getting message: {error}")
            return {}

    def send_email(self, to: str, subject: str, body: str, 
                  attachments: List[str] = None) -> Optional[Dict]:
        """Send an email with optional attachments"""
        try:
            message = MIMEMultipart()
            message["To"] = to
            message["Subject"] = subject
            
            # Add body
            message.attach(MIMEText(body, "plain"))
            
            # Add attachments if any
            if attachments:
                for file_path in attachments:
                    content_type, _ = mimetypes.guess_type(file_path)
                    if content_type is None:
                        content_type = "application/octet-stream"
                    
                    main_type, sub_type = content_type.split("/", 1)
                    
                    with open(file_path, "rb") as fp:
                        attachment = MIMEBase(main_type, sub_type)
                        attachment.set_payload(fp.read())
                        attachment.add_header(
                            "Content-Disposition", "attachment", 
                            filename=os.path.basename(file_path)
                        )
                        message.attach(attachment)
            
            # Encode and send
            encoded_message = base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode()
            
            sent_message = self.service.users().messages().send(
                userId="me",
                body={"raw": encoded_message}
            ).execute()
            
            print(f"Message Id: {sent_message['id']}")
            return sent_message
            
        except HttpError as error:
            print(f"Error sending email: {error}")
            return None

    def create_draft(self, to: str, subject: str, body: str) -> Optional[Dict]:
        """Create an email draft"""
        try:
            message = EmailMessage()
            message.set_content(body)
            message["To"] = to
            message["Subject"] = subject
            
            encoded_message = base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode()
            
            draft = self.service.users().drafts().create(
                userId="me",
                body={"message": {"raw": encoded_message}}
            ).execute()
            
            print(f"Draft Id: {draft['id']}")
            return draft
            
        except HttpError as error:
            print(f"Error creating draft: {error}")
            return None

    # Label Operations
    def list_labels(self) -> List[Dict]:
        """List all labels"""
        try:
            results = self.service.users().labels().list(userId="me").execute()
            return results.get("labels", [])
        except HttpError as error:
            print(f"Error listing labels: {error}")
            return []

    def create_label(self, name: str, label_list_visibility: str = "labelShow",
                    message_list_visibility: str = "show") -> Optional[Dict]:
        """Create a new label"""
        try:
            label = {
                "name": name,
                "labelListVisibility": label_list_visibility,
                "messageListVisibility": message_list_visibility
            }
            
            created_label = self.service.users().labels().create(
                userId="me", body=label
            ).execute()
            
            print(f"Created label: {created_label['name']}")
            return created_label
            
        except HttpError as error:
            print(f"Error creating label: {error}")
            return None

    def modify_message_labels(self, message_id: str, 
                            add_labels: List[str] = None,
                            remove_labels: List[str] = None) -> Optional[Dict]:
        """Modify labels for a specific message"""
        try:
            return self.service.users().messages().modify(
                userId="me",
                id=message_id,
                body={
                    "addLabelIds": add_labels or [],
                    "removeLabelIds": remove_labels or []
                }
            ).execute()
        except HttpError as error:
            print(f"Error modifying labels: {error}")
            return None

    # Filter Operations
    def create_filter(self, criteria: Dict, actions: Dict) -> Optional[Dict]:
        """Create an email filter"""
        try:
            filter_content = {
                "criteria": criteria,
                "action": actions
            }
            
            result = self.service.users().settings().filters().create(
                userId="me", body=filter_content
            ).execute()
            
            print(f"Created filter with id: {result['id']}")
            return result
            
        except HttpError as error:
            print(f"Error creating filter: {error}")
            return None

    # Convenience Methods
    def star_message(self, message_id: str) -> Optional[Dict]:
        """Star a message"""
        return self.modify_message_labels(message_id, add_labels=["STARRED"])

    def unstar_message(self, message_id: str) -> Optional[Dict]:
        """Unstar a message"""
        return self.modify_message_labels(message_id, remove_labels=["STARRED"])

    def mark_as_read(self, message_id: str) -> Optional[Dict]:
        """Mark a message as read"""
        return self.modify_message_labels(message_id, remove_labels=["UNREAD"])

    def mark_as_unread(self, message_id: str) -> Optional[Dict]:
        """Mark a message as unread"""
        return self.modify_message_labels(message_id, add_labels=["UNREAD"])

    def trash_message(self, message_id: str) -> Optional[Dict]:
        """Move a message to trash"""
        try:
            return self.service.users().messages().trash(
                userId="me", id=message_id
            ).execute()
        except HttpError as error:
            print(f"Error trashing message: {error}")
            return None

    def untrash_message(self, message_id: str) -> Optional[Dict]:
        """Remove a message from trash"""
        try:
            return self.service.users().messages().untrash(
                userId="me", id=message_id
            ).execute()
        except HttpError as error:
            print(f"Error untrashing message: {error}")
            return None

    def get_unread_messages(self, max_results: int = 10) -> List[Dict]:
        """Get unread messages"""
        return self.list_messages(query="is:unread", max_results=max_results)

    def get_starred_messages(self, max_results: int = 10) -> List[Dict]:
        """Get starred messages"""
        return self.list_messages(query="is:starred", max_results=max_results)

    def search_messages(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search messages using Gmail's search syntax"""
        return self.list_messages(query=query, max_results=max_results)

    def get_message_attachments(self, message_id: str, download_dir: Path = None) -> List[Dict]:
        """Download attachments from a specific email"""
        try:
            if download_dir is None:
                download_dir = Path.cwd() / "attachments"
            download_dir.mkdir(exist_ok=True)
            
            message = self.get_message(message_id)
            attachments = []
            
            if 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part.get('filename'):
                        attachment = {
                            'filename': part['filename'],
                            'mimeType': part['mimeType'],
                            'size': part.get('body', {}).get('size', 0)
                        }
                        
                        if 'data' in part['body']:
                            data = part['body']['data']
                        else:
                            att_id = part['body']['attachmentId']
                            att = self.service.users().messages().attachments().get(
                                userId='me', messageId=message_id, id=att_id
                            ).execute()
                            data = att['data']
                        
                        file_data = base64.urlsafe_b64decode(data)
                        file_path = download_dir / part['filename']
                        
                        with open(file_path, 'wb') as f:
                            f.write(file_data)
                        
                        attachment['path'] = str(file_path)
                        attachments.append(attachment)
            
            return attachments
            
        except HttpError as error:
            print(f"Error getting attachments: {error}")
            return []

    def set_vacation_responder(self, 
                             enabled: bool = True,
                             response_subject: str = "Out of Office",
                             response_body: str = "I am currently out of office.",
                             start_time: str = None,
                             end_time: str = None) -> Dict:
        """Set vacation auto-reply settings"""
        try:
            vacation_settings = {
                "enableAutoReply": enabled,
                "responseSubject": response_subject,
                "responseBodyHtml": response_body,
            }
            
            if start_time:
                vacation_settings["startTime"] = start_time
            if end_time:
                vacation_settings["endTime"] = end_time
            
            return self.service.users().settings().updateVacation(
                userId="me",
                body=vacation_settings
            ).execute()
            
        except HttpError as error:
            print(f"Error setting vacation responder: {error}")
            return None

    def set_email_forwarding(self, 
                           forward_to: str,
                           enabled: bool = True,
                           action: str = "keep") -> Dict:
        """Setup email forwarding
        action can be: 'keep', 'archive', 'trash', 'markRead'
        """
        try:
            forwarding = {
                "emailAddress": forward_to,
                "enabled": enabled,
                "disposition": action
            }
            
            return self.service.users().settings().forwardingAddresses().create(
                userId="me",
                body=forwarding
            ).execute()
            
        except HttpError as error:
            print(f"Error setting forwarding: {error}")
            return None

    def create_filter_with_forward(self, 
                                 from_email: str,
                                 forward_to: str,
                                 mark_as_read: bool = True) -> Dict:
        """Create a filter that forwards specific emails"""
        try:
            filter_content = {
                "criteria": {
                    "from": from_email
                },
                "action": {
                    "forward": forward_to,
                    "removeLabelIds": ["UNREAD"] if mark_as_read else []
                }
            }
            
            return self.service.users().settings().filters().create(
                userId="me",
                body=filter_content
            ).execute()
            
        except HttpError as error:
            print(f"Error creating forward filter: {error}")
            return None

    def get_message_history(self, message_id: str) -> List[Dict]:
        """Get the history of changes to a message"""
        try:
            message = self.get_message(message_id)
            history_id = message.get('historyId')
            
            if history_id:
                history = self.service.users().history().list(
                    userId="me",
                    startHistoryId=history_id
                ).execute()
                
                return history.get('history', [])
            return []
            
        except HttpError as error:
            print(f"Error getting message history: {error}")
            return []

    def batch_modify_messages(self, 
                            message_ids: List[str], 
                            add_labels: List[str] = None,
                            remove_labels: List[str] = None) -> Dict:
        """Modify multiple messages at once"""
        try:
            return self.service.users().messages().batchModify(
                userId="me",
                body={
                    "ids": message_ids,
                    "addLabelIds": add_labels or [],
                    "removeLabelIds": remove_labels or []
                }
            ).execute()
            
        except HttpError as error:
            print(f"Error in batch modification: {error}")
            return None

    def import_message(self, 
                      raw_email_file: str, 
                      labels: List[str] = None) -> Dict:
        """Import an external email into Gmail"""
        try:
            with open(raw_email_file, 'rb') as f:
                msg_data = base64.urlsafe_b64encode(f.read()).decode()
            
            body = {
                'raw': msg_data,
                'labelIds': labels or ['INBOX']
            }
            
            return self.service.users().messages().import_(
                userId="me",
                body=body
            ).execute()
            
        except HttpError as error:
            print(f"Error importing message: {error}")
            return None

    def get_storage_usage(self) -> Dict:
        """Get Gmail storage usage information"""
        try:
            profile = self.service.users().getProfile(
                userId="me"
            ).execute()
            
            quota = self.service.users().settings().getImap(
                userId="me"
            ).execute()
            
            return {
                "email": profile.get("emailAddress"),
                "storage_used": profile.get("quotaInGb", 0),
                "history_id": profile.get("historyId"),
                "imap_enabled": quota.get("enabled", False)
            }
            
        except HttpError as error:
            print(f"Error getting storage info: {error}")
            return {}