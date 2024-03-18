from dotenv import load_dotenv
load_dotenv()
import os
import requests
import streamlit as st
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI

HUGGINGFACE_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def image2text(url):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text = pipe(url)[0]['generated_text']
    print(text)
    return text

image2text("photo.png")


def generate_story(scenario):
    template = """
               You are a story teller;
               you can generate a story based on the simple narrative, the story should be not more than 200 words;

               CONTEXT: {scenario}
               STORY:
               """
    prompt = PromptTemplate(input_variables=["scenario"], template=template)

    story_llm = LLMChain(llm=OpenAI(temperature=0.6), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)

    print(story)
    return story




def text2speech(message):
    

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}
    payloads = {"inputs": message}

    response = requests.post(API_URL, json=payloads, headers=headers)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


# scenario = image2text("photo.png")
# story = generate_story(scenario)
# text2speech(story)


def main():
    st.set_page_config(page_title = "img to audio story")

    st.header("Turn image into audio story")
    uploaded_file = st.file_uploader("choose an image...", type="png")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        scenario = image2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story) 

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)   

        st.audio("audio.flac")

if __name__ == '__main__':
    main()       