import sys
import os
from pathlib import Path
import logging
import json
from typing import Dict, List
from pprint import pformat
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities using FileFinder
from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import our HunterAPI class and utils
HunterAPI = finder.get_class('hunter_API.py', 'HunterAPI')
utils = finder.import_module('utils.py')
printColoured = utils.printColoured

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def load_companies(file_path: str) -> Dict:
    """Load companies data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_company_domain(company_name: str) -> str:
    """Convert company name to domain name."""
    domains = {
        'Meta': 'meta.com',
        'Google': 'google.com',
        'Microsoft': 'microsoft.com',
        'Netflix': 'netflix.com',
        'MPC': 'moving-picture.com',
        'Paramount Pictures': 'paramount.com'
    }
    return domains.get(company_name, f"{company_name.lower().replace(' ', '')}.com")

def is_relevant_position(position: str) -> bool:
    """Check if the position is relevant for our outreach."""
    if not position:
        return False
        
    position = position.lower()
    
    relevant_keywords = [
        'ai', 'ml', 'artificial intelligence', 'machine learning',
        'innovation', 'r&d', 'research', 'technology',
        'engineering', 'technical', 'architect',
        'vfx', 'visual effects', 'virtual production',
        'creative', 'director', 'head', 'lead',
        'cto', 'chief technical', 'chief technology'
    ]
    
    return any(keyword in position for keyword in relevant_keywords)

def process_company(api: HunterAPI, company: str, domain: str) -> List[Dict]:
    """Process a single company to find relevant contacts."""
    prospects = []
    logger.info(f"\n📊 Analyzing {company} ({domain})")
    
    try:
        # Check email count first
        count_info = api.email_count(domain=domain)
        if count_info.get('total', 0) == 0:
            logger.warning(f"⚠️  No emails found for {domain}")
            return prospects
        
        logger.info(f"✓ Found {count_info.get('total')} email addresses")
        
        # For large companies, use a smaller limit and department filter
        limit = 10 if count_info.get('total', 0) > 1000 else 20
        
        try:
            # First try with department filter for tech/engineering
            domain_info = api.domain_search(
                domain=domain, 
                limit=limit,
                emails_type='personal',  # Only personal emails
                department='technology'  # Focus on tech department
            )
        except Exception:
            # If department search fails, try without department filter
            domain_info = api.domain_search(
                domain=domain,
                limit=limit,
                emails_type='personal'
            )
        
        pattern = domain_info.get('pattern')
        if not pattern:
            logger.warning(f"⚠️  No email pattern found for {domain}")
            return prospects
        
        logger.info(f"✓ Email pattern: {pattern}")
        
        # Process all emails and find relevant positions
        relevant_contacts = []
        for email in domain_info.get('emails', []):
            position = email.get('position', '')
            if position and is_relevant_position(position):
                # Only verify very high confidence emails to save credits
                if email.get('confidence', 0) > 90 and api.get_remaining_verifications() > 0:
                    verification = api.email_verifier(email['value'])
                    status = verification.get('status')
                else:
                    status = 'unverified'
                
                relevant_contacts.append({
                    'email': email['value'],
                    'position': position,
                    'confidence': email.get('confidence'),
                    'verification': status,
                    'department': email.get('department'),
                    'seniority': email.get('seniority'),
                    'linkedin': email.get('linkedin')
                })
        
        if relevant_contacts:
            prospects.append({
                'company': company,
                'domain': domain,
                'pattern': pattern,
                'contacts': relevant_contacts
            })
            logger.info("\n🎯 Found relevant contacts:")
            for contact in relevant_contacts:
                verification = "✓" if contact['verification'] == 'valid' else "?"
                logger.info(f"{verification} {contact['position']}: {contact['email']} ({contact['confidence']}% confidence)")
        else:
            logger.info("ℹ️  No relevant contacts found")
        
        # Show remaining API calls
        logger.info(f"\nRemaining searches: {api.get_remaining_searches()}")
        logger.info(f"Remaining verifications: {api.get_remaining_verifications()}")
        
    except Exception as e:
        logger.error(f"Error processing {company}: {str(e)}")
    
    return prospects

def main():
    """Run Hunter.io API test with practical outbound email search."""
    logger.info("\n🤖 Starting Hunter.io Outbound Email Search")
    logger.info("=====================================")
    
    try:
        # Initialize API
        api = HunterAPI()
        logger.info(f"✓ API initialized")
        logger.info(f"Searches available: {api.get_remaining_searches()}")
        logger.info(f"Verifications available: {api.get_remaining_verifications()}")
        
        # Load companies data
        companies_file = project_root / 'data/fineTune/outbound_companies.json'
        companies_data = load_companies(companies_file)
        
        # Process each company
        all_prospects = []
        for industry, companies in companies_data.items():
            logger.info(f"\n🎯 Processing {industry} companies...")
            
            for company in companies.keys():
                # Check remaining API calls
                if api.get_remaining_searches() <= 0:
                    logger.warning("⚠️  Out of API calls for today. Stopping search.")
                    break
                
                domain = get_company_domain(company)
                prospects = process_company(api, company, domain)
                all_prospects.extend(prospects)
                
                # Be nice to the API
                time.sleep(1)
        
        # Save results
        if all_prospects:
            output_file = project_root / 'data/outbound/prospects.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(all_prospects, f, indent=2)
            
            logger.info(f"\n✨ Search complete! Found contacts in {len(all_prospects)} companies")
            logger.info(f"Results saved to: {output_file}")
        else:
            logger.warning("\n⚠️  No prospects found")
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 