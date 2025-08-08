
import streamlit as st

st.set_page_config(page_title="Colab Streamlit App", layout="centered")

st.title("🚀 My Streamlit App from Google Colab")
st.write("This is running inside Google Colab and accessible via LocalTunnel!")

name = st.text_input("Enter your name:")
if name:
    st.success(f"Hello {name}, welcome to the app!")
