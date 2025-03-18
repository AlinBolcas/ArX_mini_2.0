import os
import sys
from pathlib import Path
import logging
from time import sleep
from datetime import datetime
import threading
import queue

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.I_integrations.whatsapp_API.twilio_API import TwilioAPI

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Disable Twilio's debug logging
logging.getLogger('twilio.http_client').setLevel(logging.WARNING)

# Global variables for threading
input_queue = queue.Queue()
should_exit = threading.Event()

class MessageTracker:
    def __init__(self, sid=None):
        self.last_sid = sid
        self.lock = threading.Lock()

def get_user_input():
    """Thread function to get user input"""
    while not should_exit.is_set():
        try:
            user_input = input().strip()
            if user_input:
                input_queue.put(user_input)
        except EOFError:
            break

def send_initial_message(twilio: TwilioAPI, number: str) -> bool:
    """Send initial template message"""
    try:
        logger.info("🤖 Sending initial template message...")
        response = twilio.send_template(
            to=number,
            template_id="HXa7eda43a77a0cf4460c0796834bc3991",
            variables={}  # Add variables if needed
        )
        
        if response['status'] in ['queued', 'sent', 'delivered']:
            logger.info("✅ Template message sent successfully!")
            return True
    except Exception as e:
        logger.error(f"❌ Template failed: {str(e)}")
        return False

def send_direct_message(twilio: TwilioAPI, number: str) -> bool:
    """Send direct message (within 24h window)"""
    try:
        logger.info("🤖 Sending direct message...")
        response = twilio.send_message(
            to=number,
            message="Thanks for agreeing to chat! How can I help you today? 🚀"
        )
        
        if response['status'] in ['queued', 'sent', 'delivered']:
            logger.info("✅ Direct message sent successfully!")
            return True
        else:
            if response.get('error_message'):
                logger.error(f"❌ Error: {response['error_message']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error sending direct message: {str(e)}")
        return False

def print_message_history(messages: list) -> None:
    """Print recent message history"""
    logger.info("\n📋 Recent messages:")
    for msg in messages[:5]:  # Show last 5 messages
        direction = "→" if msg['direction'] == 'outbound-api' else "←"
        timestamp = datetime.fromisoformat(msg['date_created'].replace('Z', '+00:00'))
        time_str = timestamp.strftime('%H:%M:%S')
        logger.info(f"{time_str} {direction} {msg['body']}")

def message_checker(twilio: TwilioAPI, tracker: MessageTracker):
    """Thread function to check for new messages"""
    while not should_exit.is_set():
        try:
            messages = twilio.get_message_history(limit=5)
            new_messages = []
            
            with tracker.lock:
                current_sid = tracker.last_sid
                
                for msg in messages:
                    # If we've reached the last seen message, stop looking
                    if msg.get('sid') == current_sid:
                        break
                    
                    # We only want inbound messages
                    if msg.get('direction') == 'inbound':
                        new_messages.append({
                            'sid': msg.get('sid'),
                            'from': msg.get('from_'),
                            'body': msg.get('body', '')
                        })
                
                # Update last seen message
                if messages:
                    tracker.last_sid = messages[0].get('sid')
            
            # Display messages in chronological order
            for nm in reversed(new_messages):
                print('\r', end='')  # Clear current input line
                from_str = nm['from'] if nm['from'] else 'User'
                logger.info(f"\n📱 {from_str}: {nm['body']}")
                print("\n💬 You: ", end='', flush=True)
            
            sleep(1.0)
            
        except Exception as e:
            logger.error(f"❌ Error checking messages: {str(e)}")
            sleep(1)

def main():
    try:
        twilio = TwilioAPI()
        # Load phone numbers from environment variables
        your_number = os.getenv('your_number')
        her_number = os.getenv('her_number')
        
        if not your_number or not her_number:
            logger.error("❌ Missing required environment variables: YOUR_NUMBER and/or HER_NUMBER")
            return
        
        # Send initial template
        if not send_initial_message(twilio, your_number):
            logger.error("❌ Failed to send initial message")
            return
            
        logger.info("\n💭 Chat ready! Type your messages (or 'quit' to exit)")
        logger.info("─" * 50)
        print("\n💬 You: ", end='', flush=True)
        
        # Get initial message ID and create tracker
        messages = twilio.get_message_history(limit=1)
        tracker = MessageTracker(messages[0]['sid'] if messages else None)
        
        # Start message checker thread
        checker_thread = threading.Thread(
            target=message_checker, 
            args=(twilio, tracker)
        )
        checker_thread.daemon = True
        checker_thread.start()
        
        # Start input thread
        input_thread = threading.Thread(target=get_user_input)
        input_thread.daemon = True
        input_thread.start()
        
        while not should_exit.is_set():
            # Check for user input
            try:
                user_input = input_queue.get_nowait()
                if user_input.lower() == 'quit':
                    should_exit.set()
                    break
                
                # Send message
                response = twilio.send_message(
                    to=your_number,
                    message=user_input
                )
                if response['status'] not in ['queued', 'sent', 'delivered']:
                    logger.error("❌ Failed to send message")
                else:
                    # Update tracker with our sent message
                    with tracker.lock:
                        tracker.last_sid = response['sid']
                
                print("\n💬 You: ", end='', flush=True)
                
            except queue.Empty:
                pass
            
            sleep(0.5)
            
    except KeyboardInterrupt:
        should_exit.set()
        logger.info("\n👋 Chat ended")
        messages = twilio.get_message_history(limit=5)
        print_message_history(messages)
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 