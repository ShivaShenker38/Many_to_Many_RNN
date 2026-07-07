import streamlit as st
import random
import os
from predict import generate_text

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="📚 AI Text Generator",
    page_icon="🤖",
    layout="wide"
)

# =====================================
# Custom CSS
# =====================================

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    font-size:42px;
    color:#0E76A8;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.box{
    background:#ffffff;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# =====================================
# Title
# =====================================

st.markdown("<div class='title'>📚 AI Text Generator using LSTM</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Many-to-Many RNN (Tiny Shakespeare Dataset)</div>", unsafe_allow_html=True)

st.write("")

# =====================================
# Check Model
# =====================================

if not os.path.exists("text_generator.keras"):

    st.error("Model not found!")

    st.info("Run train.py first.")

    st.stop()

# =====================================
# Sidebar
# =====================================

st.sidebar.header("⚙ Generation Settings")

temperature = st.sidebar.slider(
    "Temperature",
    0.2,
    1.5,
    0.8,
    0.1
)

num_words = st.sidebar.slider(
    "Words to Generate",
    20,
    300,
    100
)

# =====================================
# Random Seed Examples
# =====================================

examples = [

    "to be",

    "romeo",

    "juliet",

    "king",

    "love",

    "war",

    "thy",

    "the queen",

    "my lord",

    "good night",

    "where art thou",

    "the king"

]

# =====================================
# Session State
# =====================================

if "seed" not in st.session_state:

    st.session_state.seed = random.choice(examples)

# =====================================
# Buttons
# =====================================

col1,col2=st.columns(2)

with col1:

    if st.button("🎲 Random Seed"):

        st.session_state.seed=random.choice(examples)

with col2:

    if st.button("🧹 Clear"):

        st.session_state.seed=""

# =====================================
# Seed Text
# =====================================

seed=st.text_area(

    "Enter Seed Text",

    value=st.session_state.seed,

    height=140

)

st.session_state.seed=seed

# =====================================
# Generate
# =====================================

if st.button("🚀 Generate Text"):

    if seed.strip()=="":

        st.warning("Please enter a seed text.")

    else:

        progress=st.progress(0)

        with st.spinner("Generating..."):

            for i in range(100):

                progress.progress(i+1)

            output=generate_text(

                seed_text=seed,

                next_words=num_words,

                temperature=temperature

            )

        st.success("Generation Completed!")

        st.subheader("Generated Text")

        st.text_area(

            "",

            output,

            height=350

        )

        st.download_button(

            "📥 Download Text",

            output,

            file_name="generated_text.txt",

            mime="text/plain"

        )

# =====================================
# Sidebar Info
# =====================================

st.sidebar.markdown("---")

st.sidebar.success("Model : LSTM")

st.sidebar.info("Dataset : Tiny Shakespeare")

st.sidebar.write("Vocabulary : 10,000")

st.sidebar.write("Window Size : 20")

st.sidebar.write("Embedding : 64")

st.sidebar.write("Epochs : 10")

st.sidebar.markdown("---")

st.sidebar.caption("Developed using TensorFlow + Streamlit")

# =====================================
# Footer
# =====================================

st.markdown("---")

st.markdown(
"""
<center>

Made By ShivaShenker 

</center>
""",
unsafe_allow_html=True
)