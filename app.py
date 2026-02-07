import time
from src.gpt2_small import *
from src.config import GPTSettings
from transformers import GPT2Tokenizer
import streamlit as st

def response_generator(response):
    for word in response.split():
        yield word + ' '
        time.sleep(0.05)

config = GPTSettings(
    vocab_size=50257,
    block_size=1024,
    dropout=0.0
)

model = GPT2(config)
model.load_state_dict(torch.load("weights/gpt_2_pretrained_weights.pt"))
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

st.title("Small GPT2 for text Generation")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.expander('GPT Settings'):
    max_tokens = st.slider("Max Tokens to Generate", 10, 100, 50, 5)
    temperature = st.slider("Set creativity (Temperature)", 0.0, 1.0, 0.5, 0.1)
    top_k = st.slider("Set Sampling (Top_k)", 1, max_tokens, max_tokens, 5)

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input("Start a Sentence", max_chars=max_tokens*3):
    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({'role': 'user', 'content': prompt})

    idx = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    response = tokenizer.decode(out[0])

    with st.chat_message('GPT'):
        st.write_stream(response_generator(response))

    st.session_state.messages.append({'role': 'role', 'content': response})