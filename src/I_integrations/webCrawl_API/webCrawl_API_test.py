import sys
from pathlib import Path
import logging
from typing import List, Dict
import json
import requests
import tempfile
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities using FileFinder
from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import required modules and functions
ExaWrapper = finder.get_class('Exa_API.py', 'ExaWrapper')
DDGWrapper = finder.get_class('DDG_API.py', 'DDGWrapper')
WikiWrapper = finder.get_class('Wiki_API.py', 'WikiWrapper')
utils = finder.import_module('utils.py')
printColoured = utils.printColoured
quick_look = utils.quick_look

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

def download_and_preview_image(url: str) -> None:
    """Download an image and show it using quick_look."""
    try:
        # Add User-Agent header to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, headers=headers)
        if response.status_code == 200:
            # Create temp file with correct extension
            content_type = response.headers.get('content-type', '').lower()
            ext = '.jpg' if 'jpeg' in content_type else '.png' if 'png' in content_type else '.jpg'
            
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                tmp_path = tmp.name
            
            logger.info(f"🖼️ Previewing image from: {url}")
            quick_look(tmp_path)
            os.unlink(tmp_path)  # Clean up temp file
        else:
            logger.error(f"Failed to download image: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error previewing image: {str(e)}")

def main():
    """Run comparative web crawling API tests."""
    logger.info("\n🌐 Starting Web Crawling API Tests")
    logger.info("===============================")
    
    try:
        # Initialize wrappers
        exa = ExaWrapper()
        ddg = DDGWrapper()
        wiki = WikiWrapper()
        
        # Test query
        query = "best pottery classes in London 2025"
        
        logger.info(f"\n{printColoured('🔎 Testing Query:', 'white')} {query}")
        logger.info("=" * 50)
        
        # Exa search with all features
        logger.info(f"\n{printColoured('🔍 Exa Search Results:', 'cyan')}")
        logger.info(printColoured("-" * 30, 'cyan'))
        
        # Basic neural search
        logger.info(printColoured("Neural Search:", "cyan"))
        results = exa.search(
            query,
            num_results=1,
            search_type="neural",
            use_autoprompt=True,
            category="local business"
        )
        if results:
            for r in results.results:
                logger.info(f"📄 Title: {r.title}")
                logger.info(f"🔗 URL: {r.url}")
                logger.info(f"📅 Published: {r.published_date}")
                logger.info("")
        
        # Content search with highlights
        logger.info(printColoured("Content Search with Highlights:", "cyan"))
        results = exa.search_with_contents(
            query,
            num_results=1,
            max_chars=500,
            highlights=True
        )
        if results:
            for r in results:
                logger.info(f"📄 Title: {r.title}")
                logger.info(f"🔗 URL: {r.url}")
                if hasattr(r, 'highlights') and r.highlights:
                    logger.info(f"💡 Highlight: {r.highlights[0]}")
                logger.info("")
        
        # DuckDuckGo search with all features
        logger.info(f"\n{printColoured('🦆 DuckDuckGo Results:', 'yellow')}")
        logger.info(printColoured("-" * 30, 'yellow'))
        
        # Text search
        logger.info(printColoured("Text Search:", "yellow"))
        results = ddg.search(
            query,
            num_results=2,
            region="uk-en",
            safesearch="moderate"
        )
        if results:
            for r in results:
                if r.get('title') and r.get('link'):
                    logger.info(f"📄 Title: {r['title']}")
                    logger.info(f"🔗 Link: {r['link']}")
                    if r.get('snippet'):
                        logger.info(f"📝 Snippet: {r['snippet'][:200]}...")
                    logger.info("")
        else:
            logger.info("No text results found")
        
        # News search
        logger.info(printColoured("News Search:", "yellow"))
        news_results = ddg.search_news(
            query,
            num_results=2,
            region="uk-en"
        )
        if news_results:
            for r in news_results:
                if r.get('title') and r.get('link'):
                    logger.info(f"📰 Title: {r['title']}")
                    logger.info(f"📅 Published: {r.get('published', 'No date')}")
                    logger.info(f"🔗 Link: {r['link']}")
                    logger.info("")
        else:
            logger.info("No news results found")
        
        # Image search
        logger.info(printColoured("Image Search:", "yellow"))
        image_results = ddg.search_images(
            "pottery class london workshop",
            num_results=1,
            size="large",
            type_image="photo",
            layout="wide"
        )
        if image_results:
            for r in image_results:
                if r.get('image'):
                    logger.info(f"📷 Title: {r['title']}")
                    logger.info(f"📏 Size: {r['width']}x{r['height']}")
                    download_and_preview_image(r['image'])
                    logger.info("")
        
        # Wikipedia search with all features
        logger.info(f"\n{printColoured('📚 Wikipedia Results:', 'green')}")
        logger.info(printColoured("-" * 30, 'green'))
        
        # Article search
        logger.info(printColoured("Article Search:", "green"))
        articles = wiki.search(query, num_results=2)
        if articles:
            logger.info(f"🔍 Found articles: {', '.join(articles)}")
            logger.info("")
        
        # Get summary of first article
        if articles:
            logger.info(printColoured("Article Summary:", "green"))
            summary = wiki.get_summary(articles[0], sentences=2)
            if summary:
                logger.info(f"📄 Article: {articles[0]}")
                logger.info(f"📝 Summary: {summary}")
                logger.info("")
            
            # Get full page content
            logger.info(printColoured("Full Page Content:", "green"))
            page = wiki.get_page(articles[0])
            if page:
                logger.info(f"📄 Title: {page['title']}")
                logger.info(f"🔗 URL: {page['url']}")
                logger.info(f"📚 References: {len(page['references'])}")
                logger.info(f"🏷️ Categories: {len(page['categories'])}")
                if page.get('images'):
                    logger.info(f"🖼️ Images: {len(page['images'])}")
                    # Preview first image if available
                    for img in page['images'][:1]:
                        if any(ext in img.lower() for ext in ['.jpg', '.png', '.gif']):
                            if not img.startswith('http'):
                                img = f"https:{img}" if img.startswith('//') else f"https://{img}"
                            logger.info(f"📷 Preview first image:")
                            download_and_preview_image(img)
                logger.info("")
        
        # Random articles feature
        logger.info(printColoured("Random Articles:", "green"))
        random_articles = wiki.get_random(2)
        if random_articles:
            logger.info(f"🎲 Random articles: {', '.join(random_articles)}")
        
        logger.info(f"\n{printColoured('✨ All Tests Complete', 'white')}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 