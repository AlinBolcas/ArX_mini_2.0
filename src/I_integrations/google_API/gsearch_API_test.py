import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List
from pprint import pformat
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities using FileFinder
from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import required modules and functions
GSearchAPI = finder.get_class('gsearch_API.py', 'GSearchAPI')
utils = finder.import_module('utils.py')
printColoured = utils.printColoured

# Custom formatter for prettier logging
class PrettyFormatter(logging.Formatter):
    def format(self, record):
        if 'Testing' in record.msg:
            return f"\n{printColoured('>> ' + record.msg, 'magenta')}"
        elif 'Response:' in record.msg:
            return f"\n{printColoured('Response:', 'yellow')}\n{record.msg.replace('Response:', '')}"
        elif 'Error:' in record.msg:
            return f"\n{printColoured('❌ ' + record.msg, 'red')}"
        elif 'Success:' in record.msg:
            return f"\n{printColoured('✓ ' + record.msg, 'green')}"
        else:
            return f"{printColoured('>', 'blue')} {record.msg}"

# Setup logging
logger = logging.getLogger(__name__)

# 1) Clear any existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# 2) Prevent logs from going to the root logger
logger.propagate = False

# 3) Attach custom PrettyFormatter
handler = logging.StreamHandler()
handler.setFormatter(PrettyFormatter())
logger.addHandler(handler)

logger.setLevel(logging.INFO)

def format_search_result(result: Dict, search_type: str = "web") -> None:
    """Format and print search result details."""
    # Remove direct prints, use only logger
    title = result.get('title', 'No Title')
    logger.info(f"\n🔍 {printColoured(title, 'white')}")
    
    if search_type == "web":
        logger.info(f"🌐 URL: {result.get('link', 'No URL')}")
        if result.get('snippet'):
            logger.info(f"📝 Snippet: {result['snippet']}")
        if result.get('pagemap', {}).get('metatags'):
            logger.info(f"🏷️ Type: {result['pagemap']['metatags'][0].get('og:type', 'N/A')}")
            
    elif search_type == "image":
        image_url = result.get('link', '')
        logger.info(f"🖼️ URL: {image_url}")
        logger.info(f"📏 Size: {result.get('image', {}).get('width', 'N/A')}x{result.get('image', {}).get('height', 'N/A')}")
        
        # Download and display image
        if image_url:
            try:
                import requests
                from tempfile import NamedTemporaryFile
                
                response = requests.get(image_url)
                if response.status_code == 200:
                    with NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                        f.write(response.content)
                        utils.quick_look(f.name)
            except Exception as e:
                logger.error(f"Failed to display image: {str(e)}")
            
    elif search_type == "news":
        logger.info(f"📰 URL: {result.get('link', 'No URL')}")
        if result.get('snippet'):
            logger.info(f"📝 Summary: {result['snippet']}")
        if result.get('source'):
            logger.info(f"📱 Source: {result['source']}")
        if result.get('publishedAt'):
            logger.info(f"📅 Published: {result['publishedAt']}")

def main():
    """Run Google Search API test suite focused on VFX industry lead generation."""
    logger.info("\n🎯 VFX Industry Lead Research")
    logger.info("===========================")
    
    try:
        gsearch = GSearchAPI()
        
        # Test 1: Research Top VFX Studios
        logger.info("\nResearching Top VFX Studios")
        try:
            # Load target companies
            with open(project_root / "data/fineTune/outbound_companies_full.json") as f:
                companies = json.load(f)
            
            vfx_companies = companies.get("Entertainment & Media", {})
            test_companies = ["DNEG", "Framestore", "Industrial Light & Magic"]
            
            for company in test_companies:
                logger.info(f"\n🔍 Researching {company}")
                
                # Company website search
                results = gsearch.search(
                    query=f"{company} VFX studio careers leadership team",
                    num=3,
                    language="en"
                )
                
                if results.get('items'):
                    logger.info(f"Found {len(results['items'])} company results")
                    for item in results['items']:
                        format_search_result(item, "web")
                
                # Look for key positions
                positions = vfx_companies.get(company, [])
                for position in positions[:2]:  # Test with 2 positions
                    logger.info(f"\n👥 Looking for: {position}")
                    results = gsearch.search(
                        query=f"{position} {company} linkedin",
                        num=2
                    )
                    if results.get('items'):
                        for item in results['items']:
                            format_search_result(item, "web")
                            
        except Exception as e:
            logger.error(f"Company research failed: {str(e)}")
            
        # Test 2: Recent VFX Industry News
        logger.info("\nGathering Recent Industry News")
        try:
            results = gsearch.search_news(
                query="VFX studio AI pipeline innovation",
                recent=True,
                num=3
            )
            
            if results.get('items'):
                logger.info(f"Found {len(results['items'])} recent news articles")
                for item in results['items']:
                    format_search_result(item, "news")
                    
        except Exception as e:
            logger.error(f"News search failed: {str(e)}")
            
        # Test 3: Portfolio Research
        logger.info("\nResearching Studio Work")
        try:
            results = gsearch.search_images(
                query="DNEG Framestore ILM digital human character showreel",
                size="LARGE",
                type="photo",
                num=2
            )
            
            if results.get('items'):
                logger.info(f"Found {len(results['items'])} portfolio images")
                for item in results['items']:
                    format_search_result(item, "image")
                    
        except Exception as e:
            logger.error(f"Portfolio research failed: {str(e)}")
            
        # Test 4: Technical Documentation
        logger.info("\nGathering Technical Insights")
        try:
            results = gsearch.search(
                query="VFX pipeline AI integration whitepaper documentation",
                file_type="pdf",
                num=2
            )
            
            if results.get('items'):
                logger.info(f"Found {len(results['items'])} technical documents")
                for item in results['items']:
                    format_search_result(item, "web")
                    
        except Exception as e:
            logger.error(f"Documentation search failed: {str(e)}")
            
        # Test 5: Conference and Event Research
        logger.info("\nFinding Industry Events")
        try:
            results = gsearch.search(
                query="VFX AI conference SIGGRAPH FMX 2024",
                num=2
            )
            
            if results.get('items'):
                logger.info(f"Found {len(results['items'])} industry events")
                for item in results['items']:
                    format_search_result(item, "web")
                    
        except Exception as e:
            logger.error(f"Event search failed: {str(e)}")
            
        # Test 6: Site-Specific Research
        logger.info("\nResearching Company Careers")
        try:
            results = gsearch.search(
                query="AI research character pipeline",
                site="dneg.com",
                num=2
            )
            
            if results.get('items'):
                logger.info(f"Found {len(results['items'])} career opportunities")
                for item in results['items']:
                    format_search_result(item, "web")
                    
        except Exception as e:
            logger.error(f"Career research failed: {str(e)}")
            
        # Test 7: Video Content
        logger.info("\nFinding Studio Presentations")
        try:
            results = gsearch.search_videos(
                query="VFX studio AI pipeline presentation",
                duration="m",  # medium length
                num=2
            )
            
            if results.get('items'):
                logger.info(f"Found {len(results['items'])} presentations")
                for item in results['items']:
                    format_search_result(item, "web")
                    
        except Exception as e:
            logger.error(f"Video search failed: {str(e)}")

        logger.info(f"\n{printColoured('✨ Research Complete', 'white')}")
        
    except Exception as e:
        logger.error(f"Research failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 