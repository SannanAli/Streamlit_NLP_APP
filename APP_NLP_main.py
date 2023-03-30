import streamlit as st

from app_nlp_fileupload import fileupload
from app_nlp_processing import NLP

PAGE_CONFIG = {
    "page_title": "NLP Processing",
    "page_icon": "random",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}
st.set_page_config(
    **PAGE_CONFIG,
)


def main():
    menu = ["NLP", "Upload your File(NLP)", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "NLP":
        NLP()
    elif choice == "Upload your File(NLP)":
        fileupload()
    else:
        st.subheader("ABOUT")


if __name__ == "__main__":
    main()
