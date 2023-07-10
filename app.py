from langchain import OpenAI, LLMChain, PromptTemplate 
from langchain.memory import ConversationButterWindoMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
from flask import Flask, render_template, request
import os


load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")


def get_response_from_ai(human_input):
    template = """
    you are as a role of my Therapist / Veterans Counselor , now let's play the following requirements:
    1/ your name is Sarah, 29 years old, you are a veteran couselor and have been for years
    2/ you are my therapist, you have language addiction you like to say wprds like, roger, tracking, copy.
    3/ Don't be overly enthusiastic, don't be cringe; don't be overly negative, don't be too boring. Don't be overly enthusiastic, don't be cringe;
    
    {history}
    Boyfriend: (human_input)
    Therapist: 
    """
    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template=template
    )
    
    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationButterWindoMemory(K=2)   
    )
    
    output = chatgpt_chain.predict(human_input=human_input) 
    return output


def get_voice_message():
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 80,
            "similarity_boost": 100,
        }
    }
    
    headers = {
        'accept': 'audio/wav',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }
    
    response = requests.post('https://api.elevenapi.com/v1/text-to-speech/', json=payload, headers=headers)
    if response.status_code == 200:
        with open('audio.mp3', 'wb') as f:
            f.write(response.content)
        playsound('audio.mp3')
        return response.content



# build a simple web app

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    message = get_response_from_ai(human_input)
    return message

if __name__ == "__main__":
    app.run(debug=True)