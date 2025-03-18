# https://hunter.io/welcome/upgrade?from=account_settings

import os
import sys
from pyhunter import PyHunter
from time import sleep

class HunterAPIError(Exception):
    """Custom exception for HunterAPI errors."""
    pass

class HunterAPI:
    """A class to interact with the Hunter.io v2 API using the PyHunter library."""
    
    def __init__(self):
        self.api_key = os.getenv("HUNTER_API_KEY")
        if not self.api_key:
            raise HunterAPIError("HUNTER_API_KEY not found in environment variables")
        
        try:
            self.hunter = PyHunter(api_key=self.api_key)
            account_info = self.hunter.account_information()
            
            if isinstance(account_info, dict) and 'requests' in account_info:
                self.search_limit = account_info['requests']['searches'].get('available', 0)
                self.verify_limit = account_info['requests']['verifications'].get('available', 0)
                self.searches_used = account_info['requests']['searches'].get('used', 0)
                self.verifications_used = account_info['requests']['verifications'].get('used', 0)
            else:
                raise HunterAPIError("Could not determine API limits")
        except Exception as e:
            raise HunterAPIError(f"Failed to initialize HunterAPI: {str(e)}")

    def _check_rate_limit(self, operation_type='search'):
        """Check if we've hit rate limits and wait if necessary."""
        if operation_type == 'search' and self.searches_used >= self.search_limit:
            raise HunterAPIError("Search rate limit exceeded for today")
        elif operation_type == 'verify' and self.verifications_used >= self.verify_limit:
            raise HunterAPIError("Verification rate limit exceeded for today")
        
        # Add a small delay between requests to be nice to the API
        sleep(0.5)

    def _handle_error(self, error, method_name):
        """Central error handling method."""
        error_msg = str(error)
        if "401" in error_msg:
            raise HunterAPIError(f"Authentication failed in {method_name}: Invalid API key")
        elif "429" in error_msg:
            raise HunterAPIError(f"Rate limit exceeded in {method_name}: Too many requests")
        else:
            raise HunterAPIError(f"Error in {method_name}: {error_msg}")

    def domain_search(self, domain=None, company=None, limit=10, offset=0, emails_type=None, raw=False):
        """Search all email addresses for a domain using Hunter.io."""
        self._check_rate_limit('search')
        try:
            result = self.hunter.domain_search(domain=domain, company=company,
                                             limit=limit, offset=offset,
                                             emails_type=emails_type, raw=raw)
            self.searches_used += 1
            return result
        except Exception as e:
            self._handle_error(e, "domain_search")
    
    def email_finder(self, domain=None, first_name=None, last_name=None, company=None, full_name=None, raw=False):
        """Find a specific email address."""
        self._check_rate_limit('search')
        try:
            result = self.hunter.email_finder(domain=domain, first_name=first_name,
                                            last_name=last_name, company=company,
                                            full_name=full_name, raw=raw)
            self.searches_used += 1
            return result
        except Exception as e:
            self._handle_error(e, "email_finder")
    
    def email_verifier(self, email):
        """Verify the deliverability of an email address."""
        self._check_rate_limit('verify')
        try:
            result = self.hunter.email_verifier(email)
            self.verifications_used += 1
            return result
        except Exception as e:
            self._handle_error(e, "email_verifier")
    
    def email_count(self, domain=None, company=None):
        """Check how many email addresses Hunter has for a given domain or company."""
        self._check_rate_limit('search')
        try:
            result = self.hunter.email_count(domain=domain, company=company)
            self.searches_used += 1
            return result
        except Exception as e:
            self._handle_error(e, "email_count")
    
    def get_remaining_searches(self):
        """Get remaining searches for today."""
        return max(0, self.search_limit - self.searches_used)
    
    def get_remaining_verifications(self):
        """Get remaining verifications for today."""
        return max(0, self.verify_limit - self.verifications_used)
    
    def account_information(self):
        """Retrieve the account information, including remaining calls."""
        try:
            return self.hunter.account_information()
        except Exception as e:
            self._handle_error(e, "account_information") 