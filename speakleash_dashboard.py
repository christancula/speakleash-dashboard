import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import json_normalize
from streamlit_extras.add_vertical_space import add_vertical_space
from speakleash import Speakleash
import os
import random

#Init

base_dir = os.path.join(os.path.dirname(__file__))
replicate_to = os.path.join(base_dir, "datasets")
sl = Speakleash(replicate_to)

#Prepare data

datasets = []
size = []
name = []
category = []
avg_doc_length = []
avg_words_in_sentence = []
avg_sents_in_docs = []
avg_text_dynamics = []
avg_nouns_to_verbs = []
avg_stopwords_to_words = []

total_size_mb = 0

for d in sl.datasets:
    size_mb = round(d.characters/1024/1024)
    datasets.append("Dataset: {0}, size: {1} MB, characters: {2}, documents: {3}".format(d.name, size_mb, d.characters, d.documents))
    size.append(size_mb)
    name.append(d.name)
    category.append(d.category)
    total_size_mb += size_mb
    #Get metrics
    
    avg_doc_length.append(d.words/d.documents)
    try:
      avg_words_in_sentence.append(d.words/d.sentences)
    except:
      avg_words_in_sentence.append(None)
    try:
      avg_sents_in_docs.append(d.sentences/d.documents)
    except:
      avg_sents_in_docs.append(None)
    try: 
      avg_text_dynamics.append(d.verbs/d.words)
    except:
      avg_text_dynamics.append(None)
    try: 
      avg_nouns_to_verbs.append(d.nouns/d.verbs)
    except:
      avg_nouns_to_verbs.append(None)
    try: 
      avg_stopwords_to_words.append(d.stopwords/d.words)
    except:
      avg_stopwords_to_words.append(None)

data = {
  "name": name,
  "category": category,
  "size": size,
  "avg doc length": avg_doc_length,
  "avg sentence length" : avg_words_in_sentence,
  "avg sentences in doc": avg_sents_in_docs,
  "avg text dynamics" : avg_text_dynamics,
  "avg nouns to verbs" : avg_nouns_to_verbs,
  "avg stopwords to words": avg_stopwords_to_words
}

#Using name as indexer for easier navigation
df = pd.DataFrame(data).set_index('name')

#Prepare layout

st.set_page_config(page_title="Speakleash Dashboard", layout="wide")

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 1, 0.2, 1, 0.1)
)

row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))

row1a_1, row1a_2 = st.columns(
    (1, 1)
)

row2_spacer1, row2_1, row2_spacer2 = st.columns((0.1, 3.2, 0.1))

line1_spacer1, line1_1, line1_spacer2, line1_2, line1_spacer2= st.columns((0.1, 1, 0.1, 1, 0.1))

row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

row0_1.title("Speakleash a.k.a. Spichlerz Dashboard")


with row0_2:
    add_vertical_space()

row0_2.subheader(
    "[WWW](https://speakleash.org/) | [GitHub](https://github.com/speakleash) | [Twitter](https://twitter.com/Speak_Leash)"
)



fig1_1 = go.Figure(go.Indicator(
    value = total_size_mb/1024,
    number = {'valueformat':'.2f'},
    mode = "gauge+number",
    title = {'text': "<b>Project data progress</b><br><span style='color: gray; font-size:0.9em'>#GB of 1TB target</span>", 'font': {"size": 14}},
    gauge = {'axis': {'range': [None, 1200]},
            'bar': {'color': "darkblue"},
            'steps' : [
                 {'range': [0, 250], 'color': 'red'},
                 {'range': [250, 500], 'color': "orange"},
                 {'range': [500, 750], 'color': "yellow"},
                 {'range': [750, 1000], 'color': "yellowgreen"},
                 {'range': [1000, 1500], 'color': "green"}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': 1024}}))
fig1a_1 = px.pie(df, values='size', names=df.index)
fig1a_2 = px.pie(df, values='size', names='category')
fig2a_1 = px.bar(df, x=df.index, y='size')



with row1_1:
    st.markdown(
        "An open collaboration project to build a data set for Language Modeling with a capacity of at least 1TB comprised of diverse texts in Polish. Our aim is to enable machine learning research and to train a Generative Pre-trained Transformer Model from collected data"
    )
    
    st.plotly_chart(fig1_1, theme="streamlit", use_container_width=True)


with row1a_1:
    st.plotly_chart(fig1a_1, theme="streamlit", use_container_width=True)

with row1a_2:
    st.plotly_chart(fig1a_2, theme="streamlit", use_container_width=True)


def show_comparison(fig2a_1, filter_choice):
    
    fig2a_1 = px.bar(df, x=df.index, y=filter_choice)
    
    with row4_1:
        st.plotly_chart(fig2a_1, theme="streamlit", use_container_width=True)
    

choice = None


with row2_1:
    choice = st.selectbox(
        "Select one of our datasets",
        datasets)

selected_ds = choice.split(",")[0].split(":")[1].strip()     
       
with line1_1.container():    
  if choice:
      st.header("Dataset: {}".format(selected_ds))
      st.write(sl.get(selected_ds).manifest)
    

with line1_2:
    
    if choice:

        counter = 0
        random_step = random.randrange(1, 10)
        txt = ""
        meta = {}

        ds = sl.get(selected_ds).samples
        for doc in ds:
            txt = doc['text'] 
            meta = doc['meta']
            counter += 1
            if counter == random_step:
                break

        st.subheader("Random document (max 200 chars))")
        st.write(txt[:200])
        st.write(meta)


 

filters = ['size','avg doc length','avg sentence length','avg sentences in doc','avg text dynamics','avg nouns to verbs','avg stopwords to words']
with row3_1:
    filter_choice = st.selectbox(
    "Select filter to compare",
        filters)
    if filter_choice:
      show_comparison(fig2a_1, filter_choice)


with row3_2:
    if choice:
        st.subheader("Selected dataset characteristics")
        st.write("Metrics compared to average metrics in all datasets (average = 1)")



with row4_2:
        if choice:
            theta = ['avg doc length','avg sentence length','avg sentences in doc',
                    'avg text dynamics','avg nouns to verbs', 'avg stopwords to words']
            
            #Selected dataset metrics compared to average metrics in all datasets
            r=df[theta].loc[selected_ds]/(df[theta].sum()/len(sl.datasets))
            radar_df = pd.DataFrame(dict(r=r, theta=theta))
            fig = px.line_polar(radar_df, r=r, theta=theta, line_close=True)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
