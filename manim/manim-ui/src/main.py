import os
import subprocess
import streamlit as st
from manim import *
from PIL import Image
from utils import *
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

st.set_page_config(
    page_title="DistilLM",
)

# Add a selectbox for the user to choose the model
st.session_state.model_choice = st.selectbox("Choose the model to use", ["Claude",
                                                        "GPT",
                                                        "Local Model"])

def query_llm(input_text, history=None):
    if st.session_state.model_choice == "Local Model":
        reply_text, history = query_llm(input_text)
    return reply_text, history

def query_llm_api(input_text, history=None):
    if st.session_state.model_choice == "GPT":
        yield query_gpt(openai_client, input_text, history)
    elif st.session_state.model_choice == "Claude":
        yield query_claude(anthropic_client, input_text, history)

# Load API keys from local JSON files
api_keys_file = os.path.join(os.path.dirname(__file__), 'api_keys.json')
if os.path.exists(api_keys_file):
    with open(api_keys_file, 'r') as f:
        api_keys = json.load(f)
else:
    api_keys = {}

if st.session_state.model_choice == "Local Model":
    base_model = "deepseek-ai/deepseek-coder-6.7b-instruct"
    adapter_names = ["pravsels/deepseek-coder-6.7b-instruct_finetuned_manim_pile_context_FIM_1024"]
    adapter_index = 0
    adapter_name = adapter_names[adapter_index]

    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 torch_dtype=torch.bfloat16)
    model.load_adapter(adapter_name)

    query_func = query_llm

elif st.session_state.model_choice == "GPT":
    query_func = query_llm_api
    # Check if OpenAI API key is already saved
    if 'openai_api_key' not in api_keys:
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
        if openai_api_key:
            api_keys['openai_api_key'] = openai_api_key
            with open(api_keys_file, 'w') as f:
                json.dump(api_keys, f)
    else:
        openai_api_key = api_keys['openai_api_key']

    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key to use GPT.")
    else:
        # Use the saved API key to access GPT
        from openai import OpenAI
        openai_client = OpenAI(api_key=openai_api_key)

elif st.session_state.model_choice == "Claude":
    query_func = query_llm_api
    print('IN CLAUDE SECTION')
    # Check if Anthropic API key is already saved
    if 'anthropic_api_key' not in api_keys:
        anthropic_api_key = st.text_input("Enter your Anthropic API Key", type="password")
        if anthropic_api_key:
            api_keys['anthropic_api_key'] = anthropic_api_key
            with open(api_keys_file, 'w') as f:
                json.dump(api_keys, f)
    else:
        anthropic_api_key = api_keys['anthropic_api_key']

    if not anthropic_api_key:
        st.warning("Please enter your Anthropic API Key to use Claude.")
    else:
        # Use the saved API key to access Claude
        import anthropic
        anthropic.api_key = anthropic_api_key

        anthropic_client = anthropic.Anthropic(
                            api_key=anthropic_api_key,
                           )

def toggle():
    st.session_state.animate = not st.session_state.animate

st.session_state.animate = False
generate_video = None 
# generate_video = st.button("Animate", type="primary", on_click=toggle,
#                             disabled=st.session_state.animate)
show_code = True
show_reply = True
code_response = ""

######## CHAT SECTION ########

# initializing chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history 
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

import time 
# user input 
if prompt := st.chat_input("It's manim time!"):
    with st.chat_message('user'):
        st.markdown(prompt)
    # add message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # response from bot 
    with st.chat_message('assistant'):
        response = st.write_stream(query_func(prompt, st.session_state.messages))
    # add bot response to history 
    st.session_state.messages.append({'role': 'assistant', 'content': response})

if generate_video:
    if not prompt:
        st.error("Error: Please write a prompt to generate the video.")
        st.stop()

    # Prompt must be trimmed of spaces at the beginning and end
    prompt = prompt.strip()

    # Remove ", ', \ characters
    prompt = prompt.replace('"', '')
    prompt = prompt.replace("'", "")
    prompt = prompt.replace("\\", "")

    try:
        reply_text, history_new = query_lm(prompt)
        print('REPLY FROM LM FUNC : ', reply_text)
    except:
        st.error(
            "Error: We couldn't animate the generated code. Please reload the page, or try again later")
        st.stop()

    code_response = extract_construct_code(extract_code(reply_text))
    print('code response : ', code_response)

    if show_reply:
        st.text_area(label="Raw reply from LM: ",
                        value=reply_text,
                        key="reply_text")

    if show_code:
        st.text_area(label="Code generated: ",
                    value=code_response,
                    key="code_input")

    print('CURRENT DIR : ', os.path.dirname(__file__))
    if os.path.exists(os.path.dirname(__file__) + '../GenScene.py'):
        os.remove(os.path.dirname(__file__) + '../GenScene.py')

    if os.path.exists(os.path.dirname(__file__) + '../GenScene.mp4'):
        os.remove(os.path.dirname(__file__) + '../GenScene.mp4')

    try:
        with open(os.path.dirname(__file__) + "../GenScene.py", "w") as f:
            f.write(create_file_content(code_response))
    except:
        st.error("Error: We couldn't write the generated code to the Python file. Please reload the page, or try again later")
        st.stop()

    COMMAND_TO_RENDER = "manim GenScene.py GenScene --format=mp4 --media_dir ../"

    problem_to_render = False

    try:
        working_dir = os.path.dirname(__file__) + "../"
        subprocess.run(COMMAND_TO_RENDER, check=True, cwd=working_dir, shell=True)
    except Exception as e:
        problem_to_render = True
        st.error(
            f"Error: LLM generated code that Manim can't process.")

    if not problem_to_render:
        try:
            video_file = open(os.path.dirname(__file__) + '../GenScene.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
        except FileNotFoundError:
            st.error("Error: generated video file couldn't be found. Please reload the page.")
        except:
            st.error(
            "Error: Something went wrong while displaying video. Please reload the page.")
    try:
        python_file = open(os.path.dirname(__file__) + '../GenScene.py', 'rb')
        st.download_button("Download scene in Python",
                            python_file, "GenScene.py", "text/plain")
    except:
        st.error(
            "Error: Something went wrong finding the Python file. Please reload the page.")

    st.session_state.animate = False
