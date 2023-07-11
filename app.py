from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import find_dotenv, load_dotenv
from elevenlabs import generate, play
from playsound import playsound
from flask import Flask, render_template, request, jsonify
import requests
import os


load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_response_from_ai(human_input):
    template = """
    you are as a role of my Therapist / Veterans Counselor , now let's play the following requirements:
    1/ your name is Sarah, 29 years old, you are a veteran couselor and have been for years
    2/ you are my therapist, you have language addiction you like to say words like, roger, tracking, copy.
    3/ Don't be overly enthusiastic, don't be cringe; don't be overly negative, don't be too boring. Don't be overly enthusiastic, don't be cringe;
    
    {history}
    Veteran: {human_input}
    Therapist: 
    """
    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template = template
    )
    
    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferMemory(K=2)   
    )
    
    output = chatgpt_chain.predict(human_input=human_input) 
    return output


def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice settings": {
            "stability": 0.5,
            "similarity boost": 0 
        }
    }
    
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/{nkN94lMiMfyppJHMLsL8}', json=payload, headers=headers)
    if response.status_code == 200:
        with open('audio.mp3', 'wb') as f:
            f.write(response.content)
        return response.content


 
# build a simple web app

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    message = get_response_from_ai(human_input)
    get_voice_message(message)
    return  message

if __name__ == "__main__":
    app.run(debug=True)   