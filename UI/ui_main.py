import streamlit as st
import os

# Define constants
options = ['紅嘴黑鵯', '白頭翁', '大卷尾', '樹鵲', '綠繡眼', '五色鳥', '麻雀']
ch_to_en = {
    "紅嘴黑鵯": "black",
    "白頭翁": "white",
    "大卷尾": "bladro",
    "樹鵲": "tree",
    "綠繡眼": "green",
    "五色鳥": "color",
    "麻雀": "sparrow"
}

# Use absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, 'picture')
audio_dir = os.path.join(base_dir, 'sample_audio')

image_paths = {
    '紅嘴黑鵯': os.path.join(image_dir, 'blackbulbul.jpeg'),
    '白頭翁': os.path.join(image_dir, 'white.jpeg'),
    '大卷尾': os.path.join(image_dir, 'bladro.jpeg'),
    '樹鵲': os.path.join(image_dir, 'tree.jpeg'),
    '綠繡眼': os.path.join(image_dir, 'green.jpeg'),
    '五色鳥': os.path.join(image_dir, 'color.jpeg'),
    '麻雀': os.path.join(image_dir, 'sparrow.jpeg')
}

audio_paths = {
    '紅嘴黑鵯': os.path.join(audio_dir, 'black.wav'),
    '白頭翁': os.path.join(audio_dir, 'white.mp3'),
    '大卷尾': os.path.join(audio_dir, 'bladro.mp3'),
    '樹鵲': os.path.join(audio_dir, 'tree.mp3'),
    '綠繡眼': os.path.join(audio_dir, 'green.mp3'),
    '五色鳥': os.path.join(audio_dir, 'color.mp3'),
    '麻雀': os.path.join(audio_dir, 'sparrow.mp3')
}

selected_option = st.radio("Choose one bird species and generate its sound", options)
# Display the corresponding image and audio
if selected_option:
    st.image(image_paths[selected_option], caption=selected_option)
    audio_file_path = audio_paths[selected_option]
    with open(audio_file_path, "rb") as audio_file:
        st.audio(audio_file.read(), format='audio/wav')
ckpt_name = ch_to_en[selected_option]
seed = st.number_input("Enter seed number (0~2)", value=0, min_value=0, max_value=2)
# Add calculate button
if st.button(f"Generate {selected_option} sound"):
    st.write(f"Generating {selected_option} sound with seed: {seed}")
    st.write(f"The sound is: {selected_option}")
    
    generated_audio_path = os.path.join(base_dir, f"generated_audio/{ckpt_name}_{seed}.wav")
    with open(generated_audio_path, "rb") as audio_file:
        st.audio(audio_file.read(), format='audio/wav')