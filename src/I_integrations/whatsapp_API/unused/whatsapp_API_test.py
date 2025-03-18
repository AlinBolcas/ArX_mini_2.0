import sys
from pathlib import Path
import logging
from datetime import datetime
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
WhatsAppAPI = finder.get_class('whatsapp_API.py', 'WhatsAppAPI')
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

def format_message_response(response: Dict) -> None:
    """Format and print message response details."""
    logger.info("\n📤 Message Details:")
    logger.info(f"Status: {response.get('status', 'N/A')}")
    logger.info(f"Direction: {response.get('direction', 'N/A')}")
    if response.get('error_message'):
        logger.error(f"Error: {response['error_message']}")

def test_text_messaging(whatsapp: WhatsAppAPI, test_number: str) -> None:
    """Test text messaging"""
    logger.info("\n🔹 Testing Text Messaging")
    try:
        test_msg = "Hello from ArX WhatsApp API test! 🤖"
        logger.info(f"\n📝 Sending text message: '{test_msg}'")
        response = whatsapp.send_text_message(
            to=test_number,
            message=test_msg,
            preview_url=True
        )
        format_message_response(response)
        
    except Exception as e:
        logger.error(f"Text messaging test failed: {str(e)}")

def test_media_messaging(whatsapp: WhatsAppAPI, test_number: str) -> None:
    """Test media messaging"""
    logger.info("\n🔹 Testing Media Messaging")
    try:
        # Test image message
        image_url = "https://example.com/arx-logo.jpg"
        logger.info(f"\n🖼️ Sending image with caption")
        response = whatsapp.send_image(
            to=test_number,
            image_url=image_url,
            caption="Check out our logo!"
        )
        format_message_response(response)
        
        # Test document message
        logger.info("\n📄 Sending document")
        doc_url = "https://example.com/arx-whitepaper.pdf"
        response = whatsapp.send_document(
            to=test_number,
            document_url=doc_url,
            caption="ArX Whitepaper",
            filename="arx-whitepaper.pdf"
        )
        format_message_response(response)
        
    except Exception as e:
        logger.error(f"Media messaging test failed: {str(e)}")

def test_template_messaging(whatsapp: WhatsAppAPI, test_number: str) -> None:
    """Test template messaging"""
    logger.info("\n🔹 Testing Template Messaging")
    try:
        template_name = "appointment_reminder"
        components = [
            {
                "type": "body",
                "parameters": [
                    {"type": "text", "text": "December 1"},
                    {"type": "text", "text": "3:00 PM"}
                ]
            }
        ]
        
        logger.info(f"\n📋 Sending template message")
        response = whatsapp.send_template_message(
            to=test_number,
            template_name=template_name,
            language="en",
            components=components
        )
        format_message_response(response)
        
    except Exception as e:
        logger.error(f"Template messaging test failed: {str(e)}")

def test_location_messaging(whatsapp: WhatsAppAPI, test_number: str) -> None:
    """Test location messaging"""
    logger.info("\n🔹 Testing Location Messaging")
    try:
        logger.info("\n📍 Sending location: 'Arvolve HQ'")
        response = whatsapp.send_location(
            to=test_number,
            latitude=51.5074,
            longitude=-0.1278,
            name="Arvolve HQ"
        )
        format_message_response(response)
        
    except Exception as e:
        logger.error(f"Location messaging test failed: {str(e)}")

def test_interactive_messaging(whatsapp: WhatsAppAPI, test_number: str) -> None:
    """Test interactive messaging"""
    logger.info("\n🔹 Testing Interactive Messaging")
    try:
        # Test buttons
        buttons = [
            {"type": "reply", "reply": {"id": "demo", "title": "Request Demo"}},
            {"type": "reply", "reply": {"id": "sales", "title": "Contact Sales"}},
            {"type": "reply", "reply": {"id": "website", "title": "Visit Website"}}
        ]
        
        logger.info("\n🔘 Sending button message")
        response = whatsapp.send_interactive_buttons(
            to=test_number,
            message="Would you like to learn more about ArX?",
            buttons=buttons
        )
        format_message_response(response)
        
        # Test list message
        sections = [
            {
                "title": "ArX Features",
                "rows": [
                    {"id": "ai", "title": "AI Integration", "description": "Learn about AI capabilities"},
                    {"id": "analytics", "title": "Analytics", "description": "Discover our analytics tools"}
                ]
            }
        ]
        
        logger.info("\n📝 Sending list message")
        response = whatsapp.send_list_message(
            to=test_number,
            message="Explore ArX Features",
            sections=sections,
            button_text="View Options"
        )
        format_message_response(response)
        
    except Exception as e:
        logger.error(f"Interactive messaging test failed: {str(e)}")

def test_reaction_messaging(whatsapp: WhatsAppAPI, test_number: str, message_id: str) -> None:
    """Test reaction messaging"""
    logger.info("\n🔹 Testing Reaction Messaging")
    try:
        logger.info("\n👍 Sending reaction")
        response = whatsapp.send_reaction(
            to=test_number,
            message_id=message_id,
            emoji="👍"
        )
        format_message_response(response)
        
    except Exception as e:
        logger.error(f"Reaction messaging test failed: {str(e)}")

def main():
    """Run WhatsApp API test suite"""
    logger.info("\n🚀 Starting WhatsApp API Test Suite")
    logger.info("================================")
    
    try:
        whatsapp = WhatsAppAPI()
        test_number = "+447934054388"
        logger.info(f"Using number: {test_number}")
        
        # Run test modules
        test_text_messaging(whatsapp, test_number)
        test_media_messaging(whatsapp, test_number)
        test_template_messaging(whatsapp, test_number)
        test_location_messaging(whatsapp, test_number)
        test_interactive_messaging(whatsapp, test_number)
        
        # Test reaction (needs a valid message_id)
        # test_reaction_messaging(whatsapp, test_number, "valid_message_id")
        
        logger.info(f"\n✨ Test Suite Complete")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 