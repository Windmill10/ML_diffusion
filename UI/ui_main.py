import streamlit as st
import os
from audio_generator import generate_audio
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pydub import AudioSegment

# Define constants
options = ['紅嘴黑鵯', '白頭翁', '大卷尾', '樹鵲', '五色鳥', '麻雀', '綠繡眼']
ch_to_en = {
    "紅嘴黑鵯": "blabul1",
    "白頭翁": "livbul1",
    "大卷尾": "bladro1",
    "樹鵲": "grytre1",
    "五色鳥": "taibar2",
    "麻雀": "eutspa",
    "綠繡眼":"swishe1"
}

# Use absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
image_paths = {

    '紅嘴黑鵯': './picture/blackbulbul.jpeg',
    '白頭翁': './picture/white.jpeg',
    '大卷尾': './picture/bladro.jpeg',
    '樹鵲': './picture/tree.jpeg',
    '五色鳥': './picture/color.jpeg',
    '麻雀': './picture/sparrow.jpeg',
    '綠繡眼': './picture/green.jpeg'
}
audio_paths = {
    '紅嘴黑鵯': './sample_audio/black.mp3',
    '白頭翁': './sample_audio/white.mp3',
    '大卷尾': './sample_audio/bladro.mp3',
    '樹鵲': './sample_audio/tree.mp3',
    '五色鳥': './sample_audio/color.mp3',
    '麻雀': './sample_audio/sparrow.mp3',
    '綠繡眼': './sample_audio/green.mp3'
}

analyzer = Analyzer()

selected_option = st.radio("Choose one bird species and generate its sound", options)

# Display the corresponding image and audio
if selected_option:
    st.image(image_paths[selected_option], caption=selected_option)
    audio_file_path = audio_paths[selected_option]
    with open(audio_file_path, "rb") as audio_file:
        st.audio(audio_file.read(), format='audio/mp3')

# Add number input fields
seed = st.number_input("Enter seed number", value=0)
num_steps = st.number_input("Enter step number", value=0)

ckpt_name = ch_to_en[selected_option]

# Add calculate button
if st.button(f"Generate {selected_option} sound"):
    st.write(f"Generating {selected_option} sound with seed: {seed}, step: {num_steps}")
    generate_audio(ckpt_name, seed, num_steps)
    st.write(f"The sound is: {selected_option}")
    with open("generated.wav", "rb") as audio_file:

