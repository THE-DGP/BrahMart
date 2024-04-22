import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text
from streamlit_chat import message
import numpy as np
import pyttsx3
from dotenv import load_dotenv
from htmlTemplates2 import new, bot_template, user_template, background_image, page_bg_img
import time
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import JSONLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import json
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.vectorstores import FAISS
import os
import getpass
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,LLMChain
from langchain.docstore.document import Document
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import openai

import os
import getpass

# os.environ['GOOGLE_API_KEY'] = getpass.getpass('Gemini API Key:')

# engine=pyttsx3.init()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyAewIz-dDslyp-XE28hvojOJny6MTBd5rE')
x = FAISS.load_local("final_faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_conversation_chain(x):
    llm = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key='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')
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer",return_messages=True)
    system_template = r"""
    You're a helpful e-commerce AI assistant. Given a user description about a product or pricing or color and other attributes. Answer the questions correctly. If the query is not related to the product just say that you didn't find any relevant product.

    Here are the description and attributes of the product:
    --------
    {context}
    --------
    """
    user_template = "Question:```{question}```"
    messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template),
            HumanMessage(
            content=(
                "Tips: If you don't find any related product find the most similar ones to the description."
                "and don't answer any question which does not fall in context"
                    )
                        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=x.as_retriever(),
        combine_docs_chain_kwargs={'prompt':prompt})
    # st.conversation=rag_chain
    return rag_chain
    

state = st.session_state
def handle_user_input(prompt):
    response = st.conversation({'question': prompt})
    st.chat_history = response['chat_history']
    mes = st.chat_history[-1]
    typewriter(mes, user_template, 10)
    # engine.say(mes)
    # engine.runAndWait()
    # st.write(mes.content)

def typewriter(text, template, speed):
   
    tokens = (text.content).split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)
        
def submit():
        st.text_received=st.session_state.widget
        st.session_state.widget=""
        st.conversation = get_conversation_chain(x)
        handle_user_input(st.text_received)
def submit1():
        # st.text_received=st.session_state.widget
        # st.session_state.widget=""
        st.conversation = get_conversation_chain(x)
        handle_user_input(st.text_received)
        
def main():
    load_dotenv()
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.write(new, unsafe_allow_html=True)
    st.title("BrahMart! 🛒")
    st.subheader("Your E-commerce AI assistant")

    st.option = st.selectbox('Select the mode of input?', ('Text', 'Voice'))
    if st.option=='Voice':
        st.write('You selected:', st.option)
        st.text_received = ""
        c1, c2 = st.columns(2)
    
        with c1:
            st.write("Tell me, I'm listening:")
        with c2:
            text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
        # st.write(text)
        st.text_received=text
        inp1=st.text_area(label="Your Query",value=text,key="widget", disabled=True)
        st.button(on_click=submit1, label="Submit")
        
        # if text:
        #     st.write("User has sent a prompt", inp)
        #     engine.say(inp)
        # for text in state.text_received:
        #     st.text(text)
            
    elif st.option=='Text':
         inp=st.text_input(label="Your Query",key='widget',on_change=submit, value='')
         
image_data = [
    {"title": "Fuson Back Cover for Samsung Galaxy J7", "image_url": "http://img6a.flixcart.com/image/cases-covers/back-cover/f/d/f/fuson-3d-vk-ipadmini2-d9516-1100x1100-imaedm45vhjfhhys.jpeg"},
    {"title": "Candy House Solid Men's Polo Neck T-Shirt", "image_url": "http://img6a.flixcart.com/image/t-shirt/v/e/h/polo3-red-ylo-gry-christy-world-m-1000x1000-imaegj44wr7cg5fx.jpeg"},
    {"title": "Mynte Solid Women's Cycling Shorts, Gym Shorts, Swim Shorts", "image_url": "http://img6a.flixcart.com/image/watch/t/j/a/77036sm02j-sonata-original-imaecqsfpxqeyhed.jpeg"},
    {"title": "Kiara Jewellery Sterling Silver Cubic Zirconia Rhodium Ring", "image_url": "http://img5a.flixcart.com/image/ring/h/j/8/kir0195-8-kiara-jewellery-ring-original-imadtdy7yhyfgvxw.jpeg"},
    {"title": "kasemantra Back Cover for Apple iPad Mini", "image_url": "http://img6a.flixcart.com/image/cases-covers/back-cover/d/h/z/kasemantra-kasemantra-confused-face-case-for-apple-ipad-mini-1100x1100-imae6pu2z6nheqzt.jpeg"},
    {"title": "Rr Rainbow Grand Dlx 30 L Backpack", "image_url": "http://img5a.flixcart.com/image/backpack/y/r/n/rbbp00104-mb-rr-rainbow-backpack-grand-dlx-original-imaecsv6hyzh7yqz.jpeg"}    
]

# Function to display images in the sidebar
def display_images_in_sidebar(image_data):
    for image_info in image_data:
        st.sidebar.image(image_info["image_url"], caption=image_info["title"], width=150)

# Button to show popular products
if st.sidebar.button('Show Popular Products'):
    display_images_in_sidebar(image_data)
# image_data = [
#     {"title": "Fuson Back Cover for Samsung Galaxy J7", "image_url": "http://img6a.flixcart.com/image/cases-covers/back-cover/f/d/f/fuson-3d-vk-ipadmini2-d9516-1100x1100-imaedm45vhjfhhys.jpeg"},
#     {"title": "Candy House Solid Men's Polo Neck T-Shirt", "image_url": "http://img6a.flixcart.com/image/t-shirt/v/e/h/polo3-red-ylo-gry-christy-world-m-1000x1000-imaegj44wr7cg5fx.jpeg"},
#     {"title": "Mynte Solid Women's Cycling Shorts, Gym Shorts, Swim Shorts", "image_url": "http://img6a.flixcart.com/image/watch/t/j/a/77036sm02j-sonata-original-imaecqsfpxqeyhed.jpeg"},
#     {"title": "Kiara Jewellery Sterling Silver Cubic Zirconia Rhodium Ring", "image_url": "http://img5a.flixcart.com/image/ring/h/j/8/kir0195-8-kiara-jewellery-ring-original-imadtdy7yhyfgvxw.jpeg"},
#     {"title": "kasemantra Back Cover for Apple iPad Mini", "image_url": "http://img6a.flixcart.com/image/cases-covers/back-cover/d/h/z/kasemantra-kasemantra-confused-face-case-for-apple-ipad-mini-1100x1100-imae6pu2z6nheqzt.jpeg"},
#     {"title": "Rr Rainbow Grand Dlx 30 L Backpack", "image_url": "http://img5a.flixcart.com/image/backpack/y/r/n/rbbp00104-mb-rr-rainbow-backpack-grand-dlx-original-imaecsv6hyzh7yqz.jpeg"}    
# ]

# # Function to display images in a grid
# def display_images_in_grid(image_data):
#     # Create 4 columns
#     cols = st.columns(3)
#     # Display images in a 4x2 grid
#     for index, image_info in enumerate(image_data):
#         with cols[index % 3]:
#             st.image(image_info["image_url"], caption=image_info["title"], width=150)

# # Button to show popular products
# if st.button('Show Popular Products'):
#     display_images_in_grid(image_data)
# # st.write("Record your voice, and play the recorded audio:")
# audio = mic_recorder(start_prompt="", stop_prompt="", key='recorder')

# if audio:
#     st.audio(audio['bytes'])

if __name__ == "__main__":
    main()