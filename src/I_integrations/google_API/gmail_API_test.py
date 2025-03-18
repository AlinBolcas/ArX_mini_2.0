import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
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
GmailWrapper = finder.get_class('gmail_API.py', 'GmailWrapper')
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

def format_message(message: Dict) -> None:
    """Format and print message details."""
    logger.info(f"\n📧 {printColoured('Message Details:', 'white')}")
    if 'subject' in message:
        logger.info(f"📝 Subject: {message['subject']}")
    if 'from' in message:
        logger.info(f"👤 From: {message['from']}")
    if 'date' in message:
        logger.info(f"📅 Date: {message['date']}")
    if 'snippet' in message:
        logger.info(f"💬 Preview: {message['snippet'][:200]}...")

def main():
    """Run Gmail API test suite."""
    logger.info("\n📧 Starting Gmail API Test Suite")
    logger.info("==============================")
    
    try:
        gmail = GmailWrapper()
        test_email = "abolcas@gmail.com"
        
        # Test 1: Send Email
        logger.info("\nTesting Send Email")
        try:
            test_subject = f"Test Email {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            sent_message = gmail.send_email(
                to=test_email,
                subject=test_subject,
                body="This is a test email from the Gmail API wrapper test suite."
            )
            if sent_message:
                logger.info("Success: Email sent successfully")
                test_message_id = sent_message['id']
            else:
                logger.error("Failed to send email")
                
        except Exception as e:
            logger.error(f"Send email failed: {str(e)}")
            
        # Test 2: Create Draft
        logger.info("\nTesting Create Draft")
        try:
            draft = gmail.create_draft(
                to=test_email,
                subject="Draft Test Email",
                body="This is a test draft email."
            )
            if draft:
                logger.info("Success: Draft created successfully")
                test_draft_id = draft['id']
            else:
                logger.error("Failed to create draft")
                
        except Exception as e:
            logger.error(f"Create draft failed: {str(e)}")
            
        # Test 3: List Messages
        logger.info("\nTesting List Messages")
        try:
            messages = gmail.list_messages(max_results=5)
            if messages:
                logger.info(f"Found {len(messages)} recent messages:")
                for msg in messages:
                    format_message(msg)
                logger.info("Success: Messages listed successfully")
            else:
                logger.error("No messages found")
                
        except Exception as e:
            logger.error(f"List messages failed: {str(e)}")
            
        # Test 4: Label Operations
        logger.info("\nTesting Label Operations")
        try:
            # Create new label
            test_label = f"TestLabel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            created_label = gmail.create_label(test_label)
            
            if created_label:
                logger.info(f"Created label: {created_label['name']}")
                
                # List all labels
                labels = gmail.list_labels()
                if labels:
                    logger.info("Available labels:")
                    for label in labels:
                        logger.info(f"  - {label['name']}")
                
                # Modify message labels if we have a test message
                if 'test_message_id' in locals():
                    modified = gmail.modify_message_labels(
                        test_message_id,
                        add_labels=[created_label['id']]
                    )
                    if modified:
                        logger.info("Success: Label applied to test message")
                        
            else:
                logger.error("Failed to create label")
                
        except Exception as e:
            logger.error(f"Label operations failed: {str(e)}")
            
        # Test 5: Message Operations
        logger.info("\nTesting Message Operations")
        try:
            if 'test_message_id' in locals():
                # Star message
                starred = gmail.star_message(test_message_id)
                if starred:
                    logger.info("Success: Message starred")
                
                # Mark as read
                read = gmail.mark_as_read(test_message_id)
                if read:
                    logger.info("Success: Message marked as read")
                
                # Get message history
                history = gmail.get_message_history(test_message_id)
                if history:
                    logger.info(f"Message has {len(history)} history entries")
                    
        except Exception as e:
            logger.error(f"Message operations failed: {str(e)}")
            
        # Test 6: Search Operations
        logger.info("\nTesting Search Operations")
        try:
            # Search for unread messages
            unread = gmail.get_unread_messages(max_results=3)
            if unread:
                logger.info(f"Found {len(unread)} unread messages")
                
            # Search for starred messages
            starred = gmail.get_starred_messages(max_results=3)
            if starred:
                logger.info(f"Found {len(starred)} starred messages")
                
            # Custom search
            search_results = gmail.search_messages(
                query="subject:Test",
                max_results=3
            )
            if search_results:
                logger.info(f"Found {len(search_results)} messages matching search")
                
        except Exception as e:
            logger.error(f"Search operations failed: {str(e)}")
            
        # Test 7: Storage and Settings
        logger.info("\nTesting Storage and Settings")
        try:
            # Get storage usage
            storage = gmail.get_storage_usage()
            if storage:
                logger.info("Storage Information:")
                logger.info(f"  Email: {storage.get('email')}")
                logger.info(f"  Storage Used: {storage.get('storage_used')} GB")
                logger.info(f"  IMAP Enabled: {storage.get('imap_enabled')}")
                
        except Exception as e:
            logger.error(f"Storage info failed: {str(e)}")
            
        # Test 8: Vacation Responder
        logger.info("\nTesting Vacation Responder")
        try:
            # Set vacation responder with correct timestamp format
            start_time = int(datetime.now().timestamp())
            end_time = int((datetime.now() + timedelta(days=1)).timestamp())
            
            responder = gmail.set_vacation_responder(
                enabled=True,
                response_subject="Out of Office Test",
                response_body="This is a test auto-reply.",
                start_time=start_time,
                end_time=end_time
            )
            
            if responder:
                logger.info("Success: Vacation responder set")
                # Disable it immediately after test
                gmail.set_vacation_responder(enabled=False)
                
        except Exception as e:
            logger.error(f"Vacation responder failed: {str(e)}")
            
        # Test 9: Batch Operations
        logger.info("\nTesting Batch Operations")
        try:
            if 'test_message_id' in locals():
                batch_result = gmail.batch_modify_messages(
                    message_ids=[test_message_id],
                    add_labels=['IMPORTANT'],
                    remove_labels=['UNREAD']
                )
                if batch_result is not None:
                    logger.info("Success: Batch modification completed")
                    
        except Exception as e:
            logger.error(f"Batch operations failed: {str(e)}")
            
        # Cleanup
        logger.info("\nPerforming Cleanup")
        try:
            if 'test_message_id' in locals():
                # Move test message to trash
                trashed = gmail.trash_message(test_message_id)
                if trashed:
                    logger.info("Success: Test message moved to trash")
                    
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

        logger.info(f"\n{printColoured('✨ Test Suite Complete', 'white')}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 