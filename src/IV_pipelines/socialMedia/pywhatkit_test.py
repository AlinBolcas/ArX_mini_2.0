import pywhatkit as pwk
import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

class WhatsAppManager:
    """Wrapper class for pywhatkit functionalities"""
    
    def __init__(self):
        """Initialize WhatsAppManager"""
        print("Please ensure you're logged into WhatsApp Web in your default browser")

    def send_message_instantly(self, phone_number: str, message: str):
        """Send instant WhatsApp message"""
        try:
            pwk.sendwhatmsg_instantly(
                phone_no=phone_number,
                message=message,
                tab_close=True
            )
            return True
        except Exception as e:
            print(f"Error sending instant message: {e}")
            return False

    def send_image(self, receiver: str, image_path: str, caption: str = ""):
        """Send image via WhatsApp"""
        try:
            pwk.sendwhats_image(
                receiver=receiver,
                img_path=str(image_path),
                caption=caption,
                tab_close=True
            )
            return True
        except Exception as e:
            print(f"Error sending image: {e}")
            return False

    def send_email(self, sender: str, password: str, subject: str, message: str, receiver: str):
        """Send plain text email"""
        try:
            pwk.send_mail(
                email_sender=sender,
                password=password,
                subject=subject,
                message=message,
                email_receiver=receiver
            )
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    def play_youtube(self, topic_or_url: str):
        """Play YouTube video"""
        try:
            pwk.playonyt(topic_or_url)
            return True
        except Exception as e:
            print(f"Error playing YouTube video: {e}")
            return False

    def google_search(self, query: str):
        """Perform Google search"""
        try:
            pwk.search(query)
            return True
        except Exception as e:
            print(f"Error performing search: {e}")
            return False

    def get_info(self, topic: str, num_lines: int = 3):
        """Get Wikipedia information"""
        try:
            info = pwk.info(topic, lines=num_lines)
            print(f"\nInformation about {topic}:")
            print(info)
            return info
        except Exception as e:
            print(f"Error getting info: {e}")
            return None

def main():
    """Test all functionalities"""
    manager = WhatsAppManager()
    
    print("\n=== Testing PyWhatKit Features ===\n")
    
    phone_number = "+447934054388"
    
    # Test instant message only
    print("Testing WhatsApp messages...")
    
    # Instant message
    print("\n1. Testing instant message...")
    instant_message = "Sending instant message from ARX."
    manager.send_message_instantly(
        phone_number=phone_number, 
        message=instant_message
    )
    
    # Send latest frame photo
    output_dir = project_root / "output"
    latest_photo = None
    if output_dir.exists():
        photos = list(output_dir.glob("frame_photo_*.jpg"))
        if photos:
            latest_photo = max(photos, key=lambda x: x.stat().st_mtime)
            print(f"\nSending latest photo: {latest_photo.name}")
            manager.send_image(
                receiver=phone_number,
                image_path=str(latest_photo),
                caption="Latest frame capture"
            )
    
    # Test email with corrected parameter names
    print("\nTesting email...")
    manager.send_email(
        sender="abolcas@gmail.com",
        password="auzv ziys qavj mdyp",
        subject="Test from ArX",
        message="This is a test email from ArX system",
        receiver="alin@arvolve.ai"
    )
    
    # Test info lookup
    print("\nTesting Wikipedia info...")
    manager.get_info("Artificial General Intelligence", num_lines=2)
    
    # Test YouTube
    print("\nOpening YouTube...")
    manager.play_youtube("Arvolve CGI Character Design")
    
    # Test Google search
    print("\nPerforming Google search...")
    manager.google_search("Arvolve AI company")

if __name__ == "__main__":
    main()
