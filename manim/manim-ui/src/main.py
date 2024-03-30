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

model_choices_dict = {
    # "Claude Haiku": "claude-3-haiku-20240307",
    "Claude Sonnet": "claude-3-sonnet-20240229",
    "Claude Opus": "claude-3-opus-20240229",
    "GPT 3.5 Turbo": "gpt-3.5-turbo-0125",
    "GPT 4 Turbo": "gpt-4-turbo-preview",
    "Local Model": "deepseek-ai/deepseek-coder-6.7b-instruct"
}

# Add a selectbox for the user to choose the model
st.session_state.model_choice = st.selectbox("Choose the model to use", 
                                             list(model_choices_dict.keys()))

def query_llm(input_text, model_name, history=None):
    if "Local Model" in model_name:
        reply_text, history = query_llm(input_text)
    return reply_text, history

def query_llm_api(model_name, history=None, stream=False):
    if "gpt" in model_name:
        return query_gpt(openai_client, 
                         model_name,
                         history, 
                         stream)
    elif "claude" in model_name:
        return query_claude(anthropic_client, 
                            model_name,
                            history, 
                            stream)

# Load API keys from local JSON files
api_keys_file = os.path.join(os.path.dirname(__file__), 'api_keys.json')
if os.path.exists(api_keys_file):
    with open(api_keys_file, 'r') as f:
        api_keys = json.load(f)
else:
    api_keys = {}

if "Local Model" in st.session_state.model_choice:
    base_model = model_choices_dict[st.session_state.model_choice]
    adapter_names = ["pravsels/deepseek-coder-6.7b-instruct_finetuned_manim_pile_context_FIM_1024"]
    adapter_index = 0
    adapter_name = adapter_names[adapter_index]

    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 torch_dtype=torch.bfloat16)
    model.load_adapter(adapter_name)

    query_func = query_llm

elif "GPT" in st.session_state.model_choice:
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

elif "Claude" in st.session_state.model_choice:
    query_func = query_llm_api
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

def toggle_code_editor():
    st.session_state.show_code_editor = not st.session_state.show_code_editor

st.session_state.animate = False
st.session_state.show_code_editor = False 
response_dict = ""
code_response = ""
generate_video = ""

######## CHAT SECTION ########

# initializing chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history 
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# user input 
if prompt := st.chat_input("It's manim time!"):
    with st.chat_message('user'):
        st.markdown(prompt)
    # add message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # response from bot 
    with st.chat_message('assistant'):
        response = query_func(model_choices_dict[st.session_state.model_choice], 
                              st.session_state.messages, 
                              stream=True)
        if 'GPT' in st.session_state.model_choice:
            ws_response = st.write_stream(response)
        elif 'Claude' in st.session_state.model_choice:
            # Convert Claude's stream to a generator
            response_generator = claude_stream_to_generator(response)
            ws_response = st.write_stream(response_generator)

    # add bot response to history 
    st.session_state.messages.append({'role': 'assistant', 'content': ws_response})


# Generate animation section 
if len(st.session_state.messages):
    latest_reply = st.session_state.messages[-1]['content']
    code_response = extract_code(latest_reply)

    col1, col2 = st.columns(2)
    with col1:
        generate_video = st.button("Animate", type="primary", 
                                   on_click=toggle,
                                   disabled=st.session_state.animate)
    with col2:
        editing_code = st.button('Edit Code', 
                                 on_click=toggle_code_editor)

    if editing_code:
        st.session_state.editable_code = st.text_area("Edit the generated code", 
                                                      value=code_response, 
                                                      height=400)
    else:
        st.session_state.editable_code = code_response


COMMAND_TO_RENDER = "manim GenScene.py GenScene --format=mp4 --media_dir ."
if generate_video:
    st.session_state.animate = True 

    if os.path.exists(os.path.dirname(__file__) + '/../GenScene.py'):
        os.remove(os.path.dirname(__file__) + '/../GenScene.py')

    if os.path.exists(os.path.dirname(__file__) + '/../GenScene.mp4'):
        os.remove(os.path.dirname(__file__) + '/../GenScene.mp4')

    try:
        code_to_render = st.session_state.editable_code
        with open(os.path.dirname(__file__) + "/../GenScene.py", "w") as f:
            f.write(create_file_content(code_to_render, COMMAND_TO_RENDER))

    except Exception as e:
        st.warning(e)
        st.stop()

    render_issue = False

    try:
        working_dir = os.path.dirname(__file__) + "/../"
        result = subprocess.run(COMMAND_TO_RENDER, 
                                check=True, 
                                cwd=working_dir, 
                                shell=True)
    except Exception as e:
        render_issue = True
        st.warning(e)

    if not render_issue:
        try:
            video_file = open(os.path.dirname(__file__) + '/../videos/GenScene/1080p60/GenScene.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            st.download_button(
                label="Download video of scene",
                data=video_bytes,
                file_name="GenScene.mp4",
                mime="video/mp4"
            )
        except FileNotFoundError as e:
            st.warning(e)
        except Exception as e:
            st.warning(e)

    st.session_state.animate = False
