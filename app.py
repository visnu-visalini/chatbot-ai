import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ==========================================
# TEXT TO SPEECH
# ==========================================

def generate_speech(text):

    if not text:
        return None

    try:

        speech_file = "response.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        ) as response:

            response.stream_to_file(speech_file)

        return speech_file

    except Exception as e:

        print("TTS Error:", e)

        return None


# ==========================================
# CHATBOT
# ==========================================

def chatbot(message, history):

    messages = [
        {
            "role": "developer",
            "content": (
                "You are a helpful AI assistant. "
                "Give clear and concise answers. "
                "Explain technical concepts using simple examples."
            )
        }
    ]

    # ------------------------------------------
    # ADD PREVIOUS CONVERSATION
    # ------------------------------------------

    for msg in history:

        text = ""

        for content_block in msg["content"]:

            if content_block["type"] == "text":

                text += content_block["text"]

        messages.append(
            {
                "role": msg["role"],
                "content": text
            }
        )

    # ------------------------------------------
    # ADD CURRENT USER MESSAGE
    # ------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    try:

        # ------------------------------------------
        # STREAM OPENAI RESPONSE
        # ------------------------------------------

        stream = client.responses.create(
            model="gpt-5-mini",
            input=messages,
            stream=True
        )

        full_response = ""

        for event in stream:

            if event.type == "response.output_text.delta":

                full_response += event.delta

                # During streaming, return text
                # and no audio yet
                yield full_response, None

        # ------------------------------------------
        # GENERATE AUDIO AFTER RESPONSE COMPLETES
        # ------------------------------------------

        audio_file = generate_speech(full_response)

        # Return final response + audio
        yield full_response, audio_file

    except Exception as e:

        print("Chatbot Error:", e)

        yield (
            "⚠️ Sorry, something went wrong. Please try again.",
            None
        )


# ==========================================
# GRADIO UI
# ==========================================

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # 🤖 My AI Assistant

        ### Chat with AI and listen to its responses 🔊
        """
    )

    # ------------------------------------------
    # AUDIO OUTPUT
    # ------------------------------------------

    audio_output = gr.Audio(
        label="🔊 AI Voice",
        type="filepath",
        autoplay=True
    )

    # ------------------------------------------
    # CHAT INTERFACE
    # ------------------------------------------

    gr.ChatInterface(
        fn=chatbot,

        additional_outputs=[
            audio_output
        ],

        title="AI Assistant",

        description=(
            "Ask anything and receive an AI-generated "
            "text and voice response."
        )
    )


# ==========================================
# LAUNCH
# ==========================================

demo.launch()