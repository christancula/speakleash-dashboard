import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import json_normalize
from streamlit_extras.add_vertical_space import add_vertical_space
from speakleash import Speakleash
import os
import random
from datetime import datetime
import ftfy


st.set_page_config(page_title="Speakleash Dashboard", layout="wide")

def show_comparison(fig2a_1, filter_choice):
    
    fig2a_1 = px.bar(df, x=df.index, y=filter_choice)
    
    with row4_1:
        st.plotly_chart(fig2a_1, theme="streamlit", use_container_width=True)


@st.cache_data()
def prepare_data(date_string):

  #Dummy datetime input string to reset cache daily. Different string triggers cache refresh

  base_dir = os.path.join(os.path.dirname(__file__))
  replicate_to = os.path.join(base_dir, "datasets")
  sl = Speakleash(replicate_to)

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
  total_documents = 0
  total_characters = 0
  total_words = 0
  total_sentences = 0
  total_verbs = 0
  total_nouns = 0
#   total_adverbs = 0
#   total_adjectives = 0
  total_punctuations = 0
  total_symbols = 0
  total_stopwords = 0
#   total_oovs = 0
  total_size_mb = 0

  for d in sl.datasets:
      punctuations = getattr(d, 'punctuations', 0)
      symbols = getattr(d, 'symbols', 0)
    #   oovs = getattr(d, 'oovs', 0)
      size_mb = round(d.characters/1024/1024)
      datasets.append("Dataset: {0}, size: {1} MB, characters: {2}, documents: {3}".format(d.name, size_mb, d.characters, d.documents))
      size.append(size_mb)
      name.append(d.name)
      category.append(d.category)
      total_size_mb += size_mb
      total_documents += getattr(d, 'documents', 0)
      total_characters += getattr(d, 'characters', 0)
      total_sentences += getattr(d, 'sentences', 0)
      total_words += getattr(d, 'words', 0)
      total_verbs += getattr(d, 'verbs', 0)
      total_nouns += getattr(d, 'nouns', 0)
    #   total_adverbs += getattr(d, 'adverbs', 0)
    #   total_adjectives += getattr(d, 'adjectives', 0)

      if isinstance(punctuations, list):
        total_punctuations += len(punctuations)
      else:
        total_punctuations += punctuations
      if isinstance(symbols, list):
        total_symbols += len(symbols)
      else:
        total_symbols += symbols
    #   if isinstance(oovs, list):
    #     total_oovs += len(oovs)
    #   else:
    #     total_oovs += oovs
      
      total_stopwords += getattr(d, 'stopwords', 0)

      try:
        avg_doc_length.append(d.words/d.documents)
      except:
        avg_doc_length.append(0)
      try:
        avg_words_in_sentence.append(d.words/d.sentences)
      except:
        avg_words_in_sentence.append(0)
      try:
        avg_sents_in_docs.append(d.sentences/d.documents)
      except:
        avg_sents_in_docs.append(0)
      try: 
        avg_text_dynamics.append(d.verbs/d.words)
      except:
        avg_text_dynamics.append(0)
      try: 
        avg_nouns_to_verbs.append(d.nouns/d.verbs)
      except:
        avg_nouns_to_verbs.append(0)
      try: 
        avg_stopwords_to_words.append(d.stopwords/d.words)
      except:
        avg_stopwords_to_words.append(0)


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

  return sl, datasets, df, total_size_mb, total_documents, total_characters, total_sentences, total_words, total_verbs, total_nouns, total_punctuations, total_symbols, total_stopwords,

#Init

sl, datasets, df, total_size_mb, total_documents, total_characters, total_sentences, total_words, total_verbs, total_nouns, total_punctuations, total_symbols, total_stopwords = prepare_data(datetime.now().strftime("%m-%d-%Y"))


#Prepare layout



row0_1, row0_2 = st.columns(2)

row1_1 = st.columns(1)[0]

row1a_1, row1a_2 = st.columns(2)

row2_1 = st.columns(1)[0]

line1_1, line1_2, = st.columns(2)

row3_1, row3_2 = st.columns(2)

row4_1, row4_2 = st.columns(2)

row0_1.title("Speakleash a.k.a. Spichlerz Dashboard")


with row0_2:
    add_vertical_space()

row0_2.subheader("[WWW](https://speakleash.org/) | [GitHub](https://github.com/speakleash) | [Twitter](https://twitter.com/Speak_Leash)")


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
    st.markdown("""
    <style>
        .center {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
        }
        .center-text {
            text-align: center;
        }
        .table-responsive {
            font-size: 0.8em;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="center-text"><h3>An open collaboration project to build a data set for Language Modeling with a capacity of at least 1TB comprised of diverse texts in Polish. Our aim is to enable machine learning research and to train a Generative Pre-trained Transformer Model from collected data</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="center-text"><h2>Summary</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="center-text">Join us!, we have so far:</div>', unsafe_allow_html=True)

    # Same code as before to create the table
    table_data = {
        'Total documents': ["{:,}".format(total_documents).replace(",", " ")],
        'Total characters': ["{:,}".format(total_characters).replace(",", " ")],
        'Total sentences': ["{:,}".format(total_sentences).replace(",", " ")],
        'Total words': ["{:,}".format(total_words).replace(",", " ")],
        'Total verbs': ["{:,}".format(total_verbs).replace(",", " ")],
        'Total nouns': ["{:,}".format(total_nouns).replace(",", " ")],
        'Total punctuations': ["{:,}".format(total_punctuations).replace(",", " ")],
        'Total symbols': ["{:,}".format(total_symbols).replace(",", " ")],
        'Total stopwords': ["{:,}".format(total_stopwords).replace(",", " ")]
    }
  
    df2 = pd.DataFrame(table_data)
  
   
    st.markdown('<div style="display: flex; justify-content: center;">' + df2.to_html(col_space="auto",justify='center',classes='table-responsive', index=False).replace('<table', '<table style="white-space: nowrap; text-align: center;"') + '</div>',unsafe_allow_html=True)


    
    st.plotly_chart(fig1_1, theme="streamlit", use_container_width=True)


with row1a_1:
    st.plotly_chart(fig1a_1, theme="streamlit", use_container_width=True)

with row1a_2:
    st.plotly_chart(fig1a_2, theme="streamlit", use_container_width=True)


  

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
        st.write(ftfy.fix_encoding(txt[:200]))
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
