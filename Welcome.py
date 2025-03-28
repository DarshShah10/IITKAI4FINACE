# 1_🏠_Welcome.py
import streamlit as st

st.set_page_config(
    page_title="Welcome - Easy Company Stories",
    page_icon="🏠",
    layout="wide"
)

st.title("Welcome to the Easy Company Story Explorer! 👋")

st.markdown("""
This tool helps you understand complex company information in simple terms.

*What you can do:*

*   *Explore Company Stories (📊):* Choose a company, get a summary of its documents, and ask questions in plain language. See key financial numbers visualized!
*   *Learn Finance Basics (🎓):* Get simple explanations for common financial terms and concepts.

Navigate using the sidebar on the left!
""")

st.info("Select a page from the sidebar to get started.")

# You could add an image here if you like:
# from PIL import Image
# image = Image.open('path/to/your/image.jpg')
# st.image(image, caption='Understanding Finance Together')