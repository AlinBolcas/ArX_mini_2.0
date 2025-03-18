import sys
import os
from pathlib import Path
import logging
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import threading
import queue
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.I_integrations.whatsapp_API.twilio_API import TwilioAPI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Disable Flask and Twilio's debug logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('twilio.http_client').setLevel(logging.WARNING)

app = Flask(__name__)
twilio = TwilioAPI()
message_queue = queue.Queue()
input_queue = queue.Queue()
should_exit = threading.Event()

def get_user_input():
    """Thread function to get user input"""
    while not should_exit.is_set():
        try:
            user_input = input().strip()
            if user_input:
                input_queue.put(user_input)
        except EOFError:
            break

@app.route("/webhook", methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages"""
    incoming_msg = request.form.get('Body', '')
    from_number = request.form.get('From', '')
    
    message_queue.put({
        'body': incoming_msg,
        'from': from_number,
        'timestamp': request.form.get('DateCreated', '')
    })
    
    resp = MessagingResponse()
    return str(resp)

def chat_interface():
    """Run the chat interface"""
    
    # Load phone numbers from environment variables
    your_number = os.getenv('your_number')
    her_number = os.getenv('her_number')
    
    if not your_number or not her_number:
        logger.error("❌ Missing required environment variables: YOUR_NUMBER and/or HER_NUMBER")
        return
    
    # Send initial template
    logger.info("🤖 Starting WhatsApp chat...")
    twilio.send_template(
        to=your_number,
        template_id="HXa7eda43a77a0cf4460c0796834bc3991",
        variables={}
    )
    
    logger.info("\n💭 Chat ready! Type your messages (or 'quit' to exit)")
    logger.info("─" * 50)
    print("\n💬 You: ", end='', flush=True)  # Initial prompt
    
    # Start input thread
    input_thread = threading.Thread(target=get_user_input)
    input_thread.daemon = True
    input_thread.start()
    
    while not should_exit.is_set():
        # Check for new messages
        try:
            while not message_queue.empty():
                msg = message_queue.get_nowait()
                print('\r', end='')  # Clear current input line
                logger.info(f"\n📱 {msg['from'].replace('whatsapp:', '')}: {msg['body']}")
                print("\n💬 You: ", end='', flush=True)  # Restore input prompt
        except queue.Empty:
            pass
            
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
                
            print("\n💬 You: ", end='', flush=True)  # New input prompt
            
        except queue.Empty:
            pass
            
        time.sleep(0.1)  # Small sleep to prevent CPU hogging

def run_server():
    """Run the Flask server"""
    app.run(port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    try:
        # Start Flask server in a separate thread
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Start chat interface
        chat_interface()
        
    except KeyboardInterrupt:
        should_exit.set()
        logger.info("\n👋 Chat ended")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}") 