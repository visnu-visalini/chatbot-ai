import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

speech_file_path = "speech.mp3"

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="echo",
    input="Hello Visalini! Welcome to your AI assistant.",
) as response:
    response.stream_to_file(speech_file_path)

print("Audio generated successfully!")