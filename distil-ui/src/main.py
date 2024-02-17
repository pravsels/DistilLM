import os
import subprocess
import streamlit as st
from manim import *
from PIL import Image
from utils import *

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

# icon = Image.open(os.path.dirname(__file__) + '/icon.png')
st.set_page_config(
    page_title="DistilLM",
    # page_icon=icon,
)

styl = f"""
<style>
  textarea[aria-label="Code generated: "] {{
    font-family: 'Consolas', monospace !important;
  }}
  }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

prompt = st.text_area("Write your math animation concept here. Use simple words.",
                      "Write manim code to draw a blue circle and convert it to a red square", 
                      key="prompt_input")

base_model = 'deepseek-ai/deepseek-coder-6.7b-instruct'
adapter_names = ["pravsels/deepseek-coder-6.7b-instruct-finetuned-manimation"]
adapter_index = 0
adapter_name = adapter_names[adapter_index]

tokenizer = AutoTokenizer.from_pretrained(base_model, 
                                          padding_side='left')
model = AutoModelForCausalLM.from_pretrained(base_model,  
                                            torch_dtype=torch.bfloat16)
model.load_adapter(adapter_name)

def query_lm(input_text, history=None, generate_size=128):
    new_user_input_ids = tokenizer.encode(tokenizer.eos_token + input_text, 
                                          return_tensors='pt')

    bot_input_ids = torch.cat([history, 
                              new_user_input_ids], 
                              dim=-1) if history is not None else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, 
                                      max_length=generate_size, 
                                      pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
                            skip_special_tokens=True), chat_history_ids


st.session_state.task_running = False
generate_video = st.button("Animate", type="primary", 
                            disabled=st.session_state.task_running)
show_code = True
show_reply = True
code_response = ""

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
    st.session_state.task_running = True
    reply_text, history_new = query_lm(prompt)

  except:
    st.error(
        "Error: We couldn't animate the generated code. Please reload the page, or try again later")
    st.stop()

  code_response = extract_construct_code(
      extract_code(reply_text))

  if show_reply:
    st.text_area(label="Raw reply from LM: ",
                 value=reply_text,
                 key="reply_text")

  if show_code:
    st.text_area(label="Code generated: ",
                 value=code_response,
                 key="code_input")

  if os.path.exists(os.path.dirname(__file__) + '/../../GenScene.py'):
    os.remove(os.path.dirname(__file__) + '/../../GenScene.py')

  if os.path.exists(os.path.dirname(__file__) + '/../../GenScene.mp4'):
    os.remove(os.path.dirname(__file__) + '/../../GenScene.mp4')

  try:
    with open("GenScene.py", "w") as f:
      f.write(create_file_content(code_response))
  except:
    st.error("Error: We couldn't create the generated code in the Python file.")
    st.stop()

  COMMAND_TO_RENDER = "manim GenScene.py GenScene --format=mp4 --media_dir . --custom_folders video_dir"

  problem_to_render = False

  try:
    working_dir = os.path.dirname(__file__) + "/../"
    subprocess.run(COMMAND_TO_RENDER, check=True, cwd=working_dir, shell=True)
  except Exception as e:
    problem_to_render = True
    st.error(
        f"Error: Manim can't render the LM generated code.")

  if not problem_to_render:
    try:
      video_file = open(os.path.dirname(__file__) + '/../GenScene.mp4', 'rb')
      video_bytes = video_file.read()
      st.video(video_bytes)
    except FileNotFoundError:
      st.error("Error: I couldn't find the generated video file. I know this is a bug and I'm working on it. Please reload the page.")
    except:
      st.error(
          "Error: Something went wrong showing your video. Please reload the page.")

  try:
    python_file = open(os.path.dirname(__file__) + '/../GenScene.py', 'rb')
    st.download_button("Download scene in Python",
                       python_file, "GenScene.py", "text/plain")
  except:
    st.error(
        "Error: Something went wrong finding the Python file. Please reload the page.")

st.session_state.task_running = False
