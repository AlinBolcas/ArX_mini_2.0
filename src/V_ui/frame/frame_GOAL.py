import cv2
import mediapipe as mp
import speech_recognition as sr
import requests
import time
from PIL import Image
import subprocess
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))

# Add this line to explicitly add the threeD_gen directory to sys.path
sys.path.append(str(project_root / "modules/threeD_gen"))

# Import custom modules from ui_blenderPipeline.py
from modules.blender_gen.blenderGen import BlenderGen
from modules.text_gen.textGen_revamp import TextGen
from modules.image_gen.imageGen import ImageGen
from modules.threeD_gen.threedGen_API import ThreeDGen as threedGen_API

class FrameIntegration:
    def __init__(self):
        print("Initializing components...")
        self.recognizer = sr.Recognizer()
        self.mp_hands = mp.solutions.hands.Hands()
        self.text_gen = TextGen(model_provider="OpenAI", model="gpt-4o", max_tokens=300, temperature=0.7)
        self.image_gen = ImageGen()
        self.blender_gen = BlenderGen()
        print("Components initialized.")

    def process_hand_gestures(self, image_data):
        print("Processing hand gestures...")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        results = self.mp_hands.process(image)
        if results.multi_hand_landmarks:
            print("Gesture detected.")
            return "gesture_detected"
        print("No gesture detected.")
        return "no_gesture"

    def recognize_speech(self, audio_data):
        print("Recognizing speech...")
        audio = sr.AudioFile(audio_data)
        with audio as source:
            audio_content = self.recognizer.record(source)
        try:
            text = self.recognizer.recognize_google(audio_content)
            print(f"Recognized text: {text}")
            return text
        except sr.UnknownValueError:
            print("Speech not understood.")
            return "Could not understand audio"

    def detect_head_movement(self, image_data):
        print("Detecting head movement...")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        head_movement = "head_moved_left" if image.mean() > 127 else "head_moved_right"
        print(f"Detected head movement: {head_movement}")
        return head_movement

    def generate_text(self, input_text):
        arvolve_context = """
        You are talking with Alin Bolcas, the founder of Arvolve.
        He is developing you right now as you are in testing.
        Arvolve stands at the forefront of blending artistic creativity with cutting-edge technology, pioneering the evolution of art through advanced 3D CGI and VFX productions. 
        Alin Bolcas is a visionary artist and technologist, the company excels in character concept design, crafting mesmerizing visuals that captivate global audiences. 
        Arvolve's commitment to innovation extends into the realm of artificial intelligence, where it develops ARV-O, a multimodal AI system inspired by human cognition. 
        This system enhances artistic workflows, facilitates creative ideation, and fosters a deeper exploration of AI's potential in arts and consciousness. 
        With a robust portfolio of high-profile projects and a dynamic approach to AI and CGI integration, Arvolve is dedicated to redefining creative expression and advancing humanity through its pioneering work.
        """
        print(f"Generating text for input: {input_text}")
        generated_text = self.text_gen.textGen_Alin(input_text, arvolve_context)
        return generated_text

    def describe_image(self, image_data):
        print("Describing image...")
        description = self.image_gen.describe_image(image_data)  # Assuming ImageGen has a method like this
        print(f"Image description: {description}")
        return description

    def generate_image(self, prompt):
        print(f"Generating image for prompt: {prompt}")
        image_path = self.image_gen.imageGen_fullPipeline(prompt)
        print(f"Image generated and saved at: {image_path}")
        return Image.open(image_path)  # Assuming you want to return the image for display

    def fetch_data_from_api(self, url):
        print(f"Fetching data from API: {url}")
        response = requests.get(url)
        print(f"Data fetched: {response.text[:100]}...")  # Print first 100 characters
        return response.text

    def generate_3d_model(self, prompt):
        print("Starting 3D model generation...")
        obj_path = "output/testttt.obj"
        try:
            threed_gen = threedGen_API()
            threed_gen.threedGen(prompt, save_path=obj_path)
            print("3D model generation completed.")
            return obj_path
        except Exception as e:
            print(f"An error occurred during 3D model generation: {e}")
            return None

    def rotate_model(self, direction):
        print("Rotating model...")
        try:
            # Implement model rotation logic here using BlenderGen
            self.blender_gen.rotate_model(direction)  # Placeholder function
            print(f"Model rotated {direction}.")
        except Exception as e:
            print(f"An error occurred while rotating the model: {e}")

    def send_response_to_frame(self, response):
        print(f"Sending response to Frame: {response}")
        with open("response.txt", "w") as file:
            file.write(response)

    def on_ble_data_received(self, data_type, data):
        print(f"Received data type: {data_type}")
        if data_type == "image":
            gesture = self.process_hand_gestures(data)
            self.send_response_to_frame(gesture)
        elif data_type == "audio":
            text = self.recognize_speech(data)
            self.send_response_to_frame(text)
        elif data_type == "head_movement_image":
            movement = self.detect_head_movement(data)
            self.send_response_to_frame(movement)
        elif data_type == "text_input":
            text = self.generate_text(data)
            self.send_response_to_frame(text)
        elif data_type == "image_for_description":
            description = self.describe_image(data)
            self.send_response_to_frame(description)
        elif data_type == "image_prompt":
            image = self.generate_image(data)
            self.send_response_to_frame("Image generated and displayed")
        elif data_type == "api_request":
            response = self.fetch_data_from_api(data)
            self.send_response_to_frame(response)
        elif data_type == "rotate_model":
            rotation_response = self.rotate_model(data)
            self.send_response_to_frame(rotation_response)

if __name__ == "__main__":
    print("Starting FrameIntegration...")
    frame_integration = FrameIntegration()

    # # Test hand gesture processing
    # print("\n--- Testing Hand Gesture Processing ---")
    # with open("test_image.jpg", "rb") as image_file:
    #     image_data = image_file.read()
    #     gesture_result = frame_integration.process_hand_gestures(image_data)
    #     print("Gesture Result:", gesture_result)

    # # Test voice command processing
    # print("\n--- Testing Voice Command Processing ---")
    # with open("test_audio.wav", "rb") as audio_file:
    #     audio_data = audio_file.read()
    #     voice_result = frame_integration.recognize_speech(audio_data)
    #     print("Voice Command Result:", voice_result)

    # # Test head movement detection
    # print("\n--- Testing Head Movement Detection ---")
    # with open("test_image.jpg", "rb") as image_file:
    #     image_data = image_file.read()
    #     head_movement_result = frame_integration.detect_head_movement(image_data)
    #     print("Head Movement Result:", head_movement_result)

    # Test text generation
    print("\n--- Testing Text Generation ---")
    text_result = frame_integration.generate_text("What is the weather like today?")
    print("Generated Text:", text_result)

    # # Test image description
    # print("\n--- Testing Image Description ---")
    # with open("test_image.jpg", "rb") as image_file:
    #     image_data = image_file.read()
    #     description_result = frame_integration.describe_image(image_data)
    #     print("Image Description Result:", description_result)

    # Test image generation
    print("\n--- Testing Image Generation ---")
    image_generation_result = frame_integration.generate_image("A futuristic cityscape with flying cars")
    print("Image Generation completed.")

    # # Test API data fetching
    # print("\n--- Testing API Data Fetching ---")
    # api_result = frame_integration.fetch_data_from_api("https://api.example.com/data")
    # print("API Fetch Result:", api_result)

    # Test 3D model generation
    print("\n--- Testing 3D Model Generation ---")
    model_generation_result = frame_integration.generate_3d_model("A futuristic cityscape with flying cars")
    print("3D Model Generation Result:", model_generation_result)

    # # Test 3D model rotation
    # print("\n--- Testing 3D Model Rotation ---")
    # rotation_result = frame_integration.rotate_model("left")
    # print("3D Model Rotation Result:", rotation_result)

    # # Simulate a BLE data received event for image generation
    # print("\n--- Simulating BLE Data Received Event ---")
    # frame_integration.on_ble_data_received("image_prompt", "A futuristic cityscape with flying cars")

    # # Read and display the response from the simulated Frame
    # print("\n--- Reading Response from Frame ---")
    # with open("response.txt", "r") as file:
    #     response = file.read()
    #     print("Response from Frame:", response)
