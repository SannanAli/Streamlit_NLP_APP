import streamlit as st
import streamlit.components.v1 as stc

import pandas as pd

# NLP package
import spacy

nlp = spacy.load("en_core_web_sm")
from spacy import displacy
from textblob import TextBlob

# Text cleaning package
import neattext as nt
import neattext.functions as nfx

# utils
from collections import Counter
import base64
import time

timestr = time.strftime("%Y%m%d-%H%M%S")

# for plotting
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px

# wordcloud

from wordcloud import WordCloud


# text ananlysis function
def text_analysis(my_text):
    docx = nlp(my_text)
    all_text = [
        (
            token.text,
            token.shape_,
            token.pos_,
            token.tag_,
            token.lemma_,
            token.is_alpha,
            token.is_stop,
        )
        for token in docx
    ]
    text_df = pd.DataFrame(
        all_text, columns=["Token", "Shape", "POS", "Tag", "Lemma", "Isalpha", "Isstop"]
    )
    return text_df


# Getting Entities
def get_entites(my_text):
    docx = nlp(my_text)
    entities = [(ent.text, ent.label_) for ent in docx.ents]
    entities_df = pd.DataFrame(entities)
    return entities_df


# for entities Rendering
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""


def render_entities(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    return result


# for keywords
def get_most_common_tokens(my_text, num=5):
    word_tokens = Counter(my_text.split())
    most_common_tokens = dict(word_tokens.most_common(num))
    return most_common_tokens


# fro sentiment
def get_sentiment(my_text):
    blob = TextBlob(my_text)
    sentiment = blob.sentiment
    return sentiment


# Fth for Wordcloud
def plot_wordcloud(my_text):
    my_wordCloud = WordCloud().generate(my_text)
    fig = plt.figure()
    plt.imshow(my_wordCloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)


# Ftn to downlaod csv
def downlaod_able(data):
    csv_file = data.to_csv(index=("false"))
    b64 = base64.b64encode(csv_file.encode()).decode()
    new_filename = "NLP_RESULT_{}_.csv".format(timestr)
    st.markdown("### ** 📁 📩DOWNLOAD CSV FILE **")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here!</a>'
    st.markdown(href, unsafe_allow_html=True)


def NLP():
    st.title("NLP PROCESSING")

    raw_text = st.text_area("Paste your text below")
    

    if st.button("Analyze"):
        with st.expander("Original Text"):
            st.write(raw_text)

        with st.expander("Text Analysis"):
            token_result_df = text_analysis(raw_text)
            st.dataframe(token_result_df, use_container_width=True)

        with st.expander("Entities"):
            # entities_result_df = get_entites(raw_text)
            # st.dataframe(entities_result_df,use_container_width=True)

            entities_result_render = render_entities(raw_text)
            stc.html(entities_result_render, scrolling=True)

        c1, c2 = st.columns(2)

        with c1:
            with st.expander("Word Statistics"):
                st.info("Word Statistics")
                docx = nt.TextFrame(raw_text)
                stats = docx.word_stats()
                st.dataframe(pd.DataFrame(stats), use_container_width=True)

            with st.expander("Top Keywords"):
                st.info("Top Keywords")
                process = nfx.remove_stopwords(raw_text)
                most_common = get_most_common_tokens(process)
                st.dataframe(pd.DataFrame(most_common, [0]), use_container_width=True)

            with st.expander("Sentiment"):
                st.info("Sentiment Analysis")
                sent_result = get_sentiment(raw_text)
                st.write(sent_result)

        with c2:
            with st.expander("Plot Word Frequency"):
                fig = plt.figure()
                Top_keywords = get_most_common_tokens(
                    process
                )
                plt.bar(most_common.keys(), Top_keywords.values())
                st.pyplot(fig)

            with st.expander("Plot Part Of Speech"):
                fig = px.bar(token_result_df, "POS")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Plot Wordcloud"):
                plot_wordcloud(raw_text)

        with st.expander("Download Annalysis Result Here"):
            downlaod_able(token_result_df)
