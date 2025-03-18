"""Base class for Google API wrappers with automatic token management."""

import os
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

class GoogleBaseAPI:
    def __init__(self, scopes: list):
        """Initialize base Google API with token management."""
        self.scopes = scopes
        self.creds = None
        self.token_path = Path('token.pickle')  # Using pickle instead of json
        self.credentials_path = Path('credentials.json')
        
    def _get_credentials(self) -> Credentials:
        """Get or refresh Google credentials with automatic token management."""
        try:
            if self.token_path.exists():
                with open(self.token_path, 'rb') as token:
                    self.creds = pickle.load(token)
            
            # If credentials don't exist or are invalid
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    try:
                        self.creds.refresh(Request())
                    except Exception:
                        # If refresh fails, remove token and start fresh
                        self.token_path.unlink(missing_ok=True)
                        return self._get_fresh_credentials()
                else:
                    return self._get_fresh_credentials()
                
                # Save the refreshed credentials
                with open(self.token_path, 'wb') as token:
                    pickle.dump(self.creds, token)
                    
            return self.creds
            
        except Exception as e:
            print(f"Error in credential management: {str(e)}")
            return self._get_fresh_credentials()
    
    def _get_fresh_credentials(self) -> Credentials:
        """Get fresh credentials through OAuth flow."""
        if not self.credentials_path.exists():
            raise FileNotFoundError("credentials.json not found")
            
        flow = InstalledAppFlow.from_client_secrets_file(
            str(self.credentials_path),
            self.scopes
        )
        self.creds = flow.run_local_server(port=0)
        
        # Save new credentials
        with open(self.token_path, 'wb') as token:
            pickle.dump(self.creds, token)
            
        return self.creds 