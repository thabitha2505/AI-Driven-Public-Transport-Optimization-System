import streamlit as st

# Function to add background image via CSS
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function
add_bg_from_url()

# Streamlit content
st.title("Welcome to My App")
st.write("This app has a custom background image!")
