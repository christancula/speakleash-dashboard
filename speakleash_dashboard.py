import os
import random
import ftfy

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_extras.add_vertical_space import add_vertical_space
from datetime import datetime
from speakleash import Speakleash

# Webpage config
st.set_page_config(page_title="Speakleash Dashboard", 
                   layout="wide", 
                   page_icon="📝")

# Functions for figures
def show_comparison(fig2a_1, filter_choice):
    fig2a_1 = px.bar(data_frame=df.groupby('category')[filter_choice].mean().reset_index(), 
                     x='category', 
                     y=filter_choice,)
    st.plotly_chart(fig2a_1, 
                    theme="streamlit", 
                    use_container_width=True)

# Cache the data for future usage
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

  return sl, datasets, df, total_size_mb, total_documents, total_characters, total_sentences, \
    total_words, total_verbs, total_nouns, total_punctuations, total_symbols, total_stopwords,

#Initialize data
sl, datasets, df, total_size_mb, total_documents, total_characters, total_sentences, total_words, \
  total_verbs, total_nouns, total_punctuations, total_symbols, total_stopwords = prepare_data(datetime.now().strftime("%m-%d-%Y"))

# Main plot with data count figure
fig1_1 = go.Figure(go.Indicator(
    value = total_size_mb/1024,
    number = {'valueformat':'.2f'},
    mode = "gauge+number",
    title = {'text': "<b>Project data progress</b><br><span style='color: gray; font-size:0.9em'>GBs of 1TB target</span>", 
             'font': {"size": 20}},
    gauge = {'axis': {'range': [None, 1200]},
            'bar': {'color': "darkblue"},
            'steps' : [
                 {'range': [0, 250], 'color': 'red'},
                 {'range': [250, 500], 'color': "orange"},
                 {'range': [500, 750], 'color': "yellow"},
                 {'range': [750, 1000], 'color': "yellowgreen"},
                 {'range': [1000, 1500], 'color': "green"}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 
                           'thickness': 0.9, 
                           'value': 1024}}))

# Other plots
fig1a_1 = px.pie(df, values='size', names=df.index)
fig1a_2 = px.pie(df, values='size', names='category')
#fig1a_2 = sns.lineplot(data=df, x=df.index, y='sizes')
#fig2a_1 = px.bar(df, x=df.index, y='size')
fig2a_1 = px.bar(df, x='category', y='size')

#Prepare layout
row0 = st.columns(1)[0] # Title
row1 = st.columns(1)[0] # Social URLs
row2 = st.columns(1)[0] # Summary text
row3 = st.columns(1)[0] # Summary table + progress plot

row4 = st.columns(1)[0] # Dataset selector
row4_1, row4_2 = st.columns(2) # Dateset details

row5 = st.columns(1)[0] # Facts and trivia header
row5_1, row5_2 = st.columns(2) # Facts and trivia with photos?

row6 = st.columns(1)[0] # Plots header
row6_1, row6_2 = st.columns(2) # Percentage plots?
row6_3, row6_4 = st.columns(2) # Other plots

# Title row
with row0:
  st.markdown("<h1 style='text-align: center;'>SpeakLeash a.k.a Spichlerz Dashboard</h1>",
              unsafe_allow_html=True)

# Socials row
with row1:
  st.markdown("<h3 style='text-align: center;'>"
              "<a href='https://speakleash.org/'>WWW</a> | "
              "<a href='https://github.com/speakleash'>GitHub</a> | "
              "<a href='https://twitter.com/Speak_Leash'>Twitter</a>"
              "</h3",
              unsafe_allow_html=True)

# Short summary
with row2:
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

    st.markdown('<div class="center-text"><h3>An open collaboration project to build a data set for Large Language Modeling \
                with a volume of at least 1TB comprised of diverse texts in Polish. Our aim is to enable machine learning \
                research in our beautiful, native language and to train a Generative Pre-trained Transformer Model from \
                collected data.</h3></div>', unsafe_allow_html=True)

# Summary table + progress plot 
with row3:
    st.markdown('<div class="center-text"><h2>So far we managed to collect:</h2></div>', 
                unsafe_allow_html=True)

    # Summary table
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
   
    st.markdown('<div style="display: flex; justify-content: center;">' + \
                df2.to_html(col_space="auto",
                            justify='center',
                            classes='table-responsive', 
                            index=False).replace('<table', '<table style="white-space: nowrap; text-align: center;"') + \
                              '</div>',unsafe_allow_html=True)
    
    st.plotly_chart(fig1_1, theme="streamlit", use_container_width=True)

# Dataset selector
choice = None
with row4:
    st.markdown("<h3 style='text-align: left'>Select one of our datasets and check it's metrics below:</h3>",
                unsafe_allow_html=True)
    choice = st.selectbox(
        "Select one of our datasets",
        datasets)

selected_ds = choice.split(",")[0].split(":")[1].strip()     
# Dataset details
with row4_1.container():    
  if choice:
      st.subheader("Dataset: {}".format(selected_ds))
      st.write(sl.get(selected_ds).manifest)
with row4_2:
    
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

        st.subheader("Random document (max 200 chars):")
        st.write(ftfy.fix_encoding(txt[:200]))
        st.write(meta)

# Facts and trivia
with row5:
   st.markdown("#")
   st.markdown("<h3 style='text-align: center'>Facts and trivia about the data we have collected \U0001F60E</h3>",
               unsafe_allow_html=True)
with row5_1:
   st.write(f"In total we have over {total_documents/1e6:.1f}M documents. Knowing :flag-pl: has about 38M inhabitans, \
             we could give each citizen {total_documents/38000000:.2f} documents. Isn't it impressive that we provide \
              something to read for each and every person in :flag-pl:?")
with row5_2:
   st.write(f"If we'd take into account that our dataset have {total_characters/1e9:.1f}B characters, we can safely \
             assume they do cover {total_characters/1800/1e6:.1f}M A4 pages. Each page is 0.05mm thick, so if we'd stack up \
              all the pages, created tower would be {total_characters/1500*0.05/1e4:.2f} meters high! Tallest :office: in \
                the world, Burj Khalifa, is 828 meters tall. Our stack of :page_with_curl: would tower over it quite significantly. And yes, \
                  we'll grow only bigger :nerd_face:")

# Plots header
with row6:
   st.markdown("#")
   st.markdown("<h3 style='text-align: center'>Datasets, but plotted </h3>",
               unsafe_allow_html=True)
   st.markdown("<p style='text-align: center'>As pictures do say more than a thousand words - go ahead \
               and play with our plots. You can easily toggle some of the datasets to compare them!</p>",
               unsafe_allow_html=True)

# In-depth plots with dataset metrics
with row6_1:
    st.subheader("Percentage distribution of datasets")    
    st.plotly_chart(fig1a_1, theme="streamlit", use_container_width=True)
with row6_2:
    st.subheader("Percentage distribution per category of data")
    st.plotly_chart(fig1a_2, theme="streamlit", use_container_width=True)

# Other plots
filters = ['size','avg doc length','avg sentence length','avg sentences in doc',\
           'avg text dynamics','avg nouns to verbs','avg stopwords to words']
 
with row6_3:
    st.subheader('Stats per dataset category')
    filter_choice = st.selectbox(
    "Select filter to compare average values of metrics",
        filters)
    if filter_choice:
      show_comparison(fig2a_1, filter_choice)
with row6_4:
    if choice:
        st.subheader("Selected dataset characteristics")
        st.write("Metrics compared to average metrics in all datasets (average = 1)")

        if choice:
            theta = ['avg doc length','avg sentence length','avg sentences in doc',
                    'avg text dynamics','avg nouns to verbs', 'avg stopwords to words']
            
            #Selected dataset metrics compared to average metrics in all datasets
            r=df[theta].loc[selected_ds]/(df[theta].sum()/len(sl.datasets))
            radar_df = pd.DataFrame(dict(r=r, theta=theta))
            fig = px.line_polar(radar_df, r=r, theta=theta, line_close=True)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)