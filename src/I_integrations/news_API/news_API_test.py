import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, List
from pprint import pformat

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities using FileFinder
from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import required modules and functions
NewsAPI = finder.get_class('newsapi_API.py', 'NewsAPI')
GNewsAPI = finder.get_class('gnews_API.py', 'GNewsAPI')
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
handler = logging.StreamHandler()
handler.setFormatter(PrettyFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def format_article(article: Dict, source: str = "unknown") -> None:
    """Format and print article details."""
    logger.info(f"\n📰 {printColoured(article['title'], 'white')}")
    logger.info(f"📅 Published: {article['publishedAt']}")
    logger.info(f"🗞️ Source: {article['source']['name']}")
    if article.get('description'):
        logger.info(f"📝 Description: {article['description'][:200]}...")
    logger.info(f"🔗 URL: {article['url']}")

def compare_world_news(newsapi_data: List[Dict], gnews_data: List[Dict]) -> None:
    """Compare world news results from both services."""
    logger.info(f"\n{printColoured('🌍 World News Comparison:', 'white')}")
    logger.info(printColoured("-" * 50, 'white'))
    
    # Compare coverage
    newsapi_sources = set(article['source']['name'] for article in newsapi_data)
    gnews_sources = set(article['source']['name'] for article in gnews_data)
    
    # Get timestamps
    newsapi_times = [datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')) 
                    for article in newsapi_data]
    gnews_times = [datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')) 
                   for article in gnews_data]
    
    # Calculate statistics
    logger.info(f"\n📊 Coverage Statistics:")
    logger.info(f"NewsAPI:")
    logger.info(f"  - Articles: {len(newsapi_data)}")
    logger.info(f"  - Unique Sources: {len(newsapi_sources)}")
    logger.info(f"  - Time Range: {min(newsapi_times).strftime('%Y-%m-%d %H:%M')} to {max(newsapi_times).strftime('%Y-%m-%d %H:%M')}")
    
    logger.info(f"\nGNews:")
    logger.info(f"  - Articles: {len(gnews_data)}")
    logger.info(f"  - Unique Sources: {len(gnews_sources)}")
    logger.info(f"  - Time Range: {min(gnews_times).strftime('%Y-%m-%d %H:%M')} to {max(gnews_times).strftime('%Y-%m-%d %H:%M')}")
    
    # Compare common sources
    common_sources = newsapi_sources.intersection(gnews_sources)
    logger.info(f"\n🔄 Common Sources: {len(common_sources)}")
    if common_sources:
        logger.info(f"Sources: {', '.join(sorted(common_sources))}")

def main():
    """Run focused AI news comparison test."""
    logger.info("\n🤖 Starting AI News Comparison Test")
    logger.info("=================================")
    
    try:
        # Initialize API wrappers
        newsapi = NewsAPI()
        gnews = GNewsAPI()
        
        # Test parameters
        query = "artificial intelligence breakthrough"
        num_results = 5  # Limit results to conserve API calls
        
        # NewsAPI search
        logger.info(f"\n{printColoured('📰 NewsAPI Results:', 'cyan')}")
        try:
            newsapi_results = newsapi.get_everything(
                query=query,
                language="en",
                sort_by="relevancy",
                page_size=num_results
            )['articles']
            
            for article in newsapi_results:
                format_article(article, "NewsAPI")
                
            logger.info(f"\nRemaining NewsAPI calls today: {newsapi.daily_limit - newsapi.calls_today}")
        except Exception as e:
            logger.error(f"NewsAPI Error: {str(e)}")
        
        # GNews search
        logger.info(f"\n{printColoured('📰 GNews Results:', 'yellow')}")
        try:
            gnews_results = gnews.search_news(
                query=query,
                lang="en",
                country="us",
                max_results=num_results,
                sort="relevance"
            )['articles']
            
            for article in gnews_results:
                format_article(article, "GNews")
                
            logger.info(f"\nRemaining GNews calls today: {gnews.daily_limit - gnews.calls_today}")
        except Exception as e:
            logger.error(f"GNews Error: {str(e)}")
            logger.info("\nTroubleshooting GNews API:")
            logger.info("1. Check your API key in .env")
            logger.info("2. Visit https://gnews.io/dashboard to activate key")
            logger.info("3. Verify subscription status")
        
        # Compare results if we have data from both APIs
        if 'newsapi_results' in locals() and 'gnews_results' in locals():
            if newsapi_results and gnews_results:
                compare_world_news(newsapi_results, gnews_results)
            else:
                logger.error("No results available for comparison")
        
        logger.info(f"\n{printColoured('✨ Test Complete', 'white')}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 