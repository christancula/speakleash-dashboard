""" SpeakLeash Datasets Dashboard - Info about collected GBs, Quality and Datasets information"""
import os
from datetime import datetime, timedelta, timezone
import json
import time

from speakleash import Speakleash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ftfy
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space


st.set_page_config(page_title="Speakleash Dashboard", layout="wide", page_icon="http://speakleash.org/wp-content/uploads/2022/12/cropped-sl-favico-black-32x32.png")

# print("\n--- START --- START --- START --- START --- START --- START --- START ---")
print(f"{datetime.now()} : // NEW SESSION / RESTART // --- START ---")


@st.cache_data()
def Prepare_Data(date_string):
    """
    Prepare information about datasets from manifests.
    """
    # Dummy datetime input string to reset cache daily. Different string triggers cache refresh.

    start_timer = time.perf_counter()
    print(f"{datetime.now()} : // DEBUG // Func: Prepare_Data()")

    base_dir = os.path.join(os.path.dirname(__file__))
    replicate_to = os.path.join(base_dir, "datasets")
    sl = Speakleash(replicate_to)

    print(f"{datetime.now()} : // DEBUG // Func: Init Speakleash = Before LOOP = {(time.perf_counter() - start_timer):.3f}")

    datasets_dates = pd.read_csv("./static/datasets_dates.csv", sep=",", header=0)
    datasets_dates["Dates"] = pd.to_datetime(datasets_dates["Dates"], utc = True, format = 'ISO8601')

    dataframe_for_all_datasets = pd.DataFrame()
    tags_sum_docs = {}

    # Gather info about every dataset
    for idx, d in enumerate(sl.datasets):

        # print(f"Name: {d.name} | Docs: {d.documents}")
        # print(sl.get(d.name).manifest)

        try:
            punctuations = d.punctuations
            if isinstance(punctuations, list):
                d_punctuations = len(punctuations)
            else:
                d_punctuations = punctuations
        except:
            d_punctuations = 0

        try:
            symbols = d.symbols
            if isinstance(symbols, list):
                d_symbols = len(symbols)
            else:
                d_symbols = symbols
        except:
            d_symbols = 0

        try:
            sentences = d.sentences
            if isinstance(sentences, list):
                d_sentences = len(sentences)
            else:
                d_sentences = sentences
        except:
            d_sentences = 0

        try:
            words = d.words
            if isinstance(words, list):
                d_words = len(words)
            else:
                d_words = words
        except:
            d_words = 0

        try:
            verbs = d.verbs
            if isinstance(verbs, list):
                d_verbs = len(verbs)
            else:
                d_verbs = verbs
        except:
            d_verbs = 0

        try:
            nouns = d.nouns
            if isinstance(nouns, list):
                d_nouns = len(nouns)
            else:
                d_nouns = nouns
        except:
            d_nouns = 0

        try:
            avg_doc_length = d.words / d.documents
        except:
            avg_doc_length = 0
        try:
            avg_words_in_sentence = d.words / d.sentences
        except:
            avg_words_in_sentence = 0
        try:
            avg_sents_in_docs = d.sentences / d.documents
        except:
            avg_sents_in_docs = 0
        try:
            avg_text_dynamics = d.verbs / d.words
        except:
            avg_text_dynamics = 0
        try:
            avg_nouns_to_verbs = d.nouns / d.verbs
        except:
            avg_nouns_to_verbs = 0
        try:
            avg_stopwords_to_words = d.stopwords / d.words
        except:
            avg_stopwords_to_words = 0


        d_manifesto = sl.get(d.name).manifest
        d_tags = {}

        try:
            dici = d_manifesto.get('category=95%')
            dici_sort = dict(sorted(dici.items(), key=lambda x: x[1], reverse=True))

            for key, value in dici_sort.items():
                calc_temp = round(value / d.documents * 100, 2)
                if calc_temp > 1 :
                    d_tags[key] = calc_temp
                else:
                    break

            tags_sum_docs = {k: tags_sum_docs.get(k, 0) + dici_sort.get(k, 0) for k in set(tags_sum_docs) | set(dici_sort)}

        except:
            try:
                d_tags = {"Rożne": d.documents / d.documents * 100}
            except:
                d_tags = {"Rożne": 0.0}

        if d_tags:
            pass
        else:
            try:
                d_tags = {"Rożne": d.documents / d.documents * 100}
            except:
                d_tags = {"Rożne": 0.0}


        try:
            try:
                d_create_date = d_manifesto.get('creation_date', "")

                if d_create_date == "":
                    # d_create_date = datasets_dates[datasets_dates['Name'] == d.name]['Dates'].values[0]
                    try:
                        d_create_date = datasets_dates[datasets_dates['Name'] == d.name]['Dates'].values[0]
                        proper_date = False
                    except:
                        d_create_date = pd.to_datetime(datetime.fromisoformat('2022-01-01'))
                        proper_date = False
                        # print(f"- Got empty string as date -> Can't find date in manifest -> Dataset: {d.name} | Create new date: {d_create_date}")
                else:
                    d_create_date = pd.to_datetime(d_create_date) # "%Y-%m-%d %H:%M:%s"
                    proper_date = True
            except:
                d_create_date = datasets_dates[datasets_dates['Name'] == d.name]['Dates'].values[0]
                proper_date = False
                # print(f"- Something is wrong with date -> Can't find date in manifest -> Dataset: {d.name} | Found date in file: {d_create_date}")
        except:
            d_create_date = pd.to_datetime(datetime.fromisoformat('2022-01-01') + timedelta(days=1, hours=12))
            proper_date = False
            # print(f"- Can't find nothing -> Can't find date in manifest -> Dataset: {d.name} | Create new date: {d_create_date}")

        try:
            d_update_date = d_manifesto.get('update_date',"")
            if d_update_date == "":
                d_update_date = d_create_date
            else:
                d_update_date = pd.to_datetime(d_update_date)
        except:
            d_update_date = d_create_date

        # print(f"Before CONCAT => Data: {d.name} | Date: {d_create_date}\n---")
        # print(f"Before CONCAT => Data: {d.name} | Date: {d_update_date}\n---")

        # Prepare DataFrame with info about Dataset
        dataframe_for_all_datasets = pd.concat([dataframe_for_all_datasets, pd.DataFrame({
                                                    "Dataset": d.name,
                                                    "Size_MB": round(d.characters/1024/1024),
                                                    "Category": d.category,
                                                    "Tags": [d_tags],
                                                    "Documents": d.documents,
                                                    "Characters": d.characters,
                                                    "Sentences": d_sentences,
                                                    "Words": d_words,
                                                    "Verbs": d_verbs,
                                                    "Nouns": d_nouns,
                                                    "Punctuations": d_punctuations,
                                                    "Symbols": d_symbols,
                                                    "Stopwords": d.stopwords,
                                                    "Quality": [d.quality],
                                                    "Description": d.description,
                                                    "License": d.license,
                                                    "Sources": d.sources,
                                                    "Creation_Date": d_create_date,
                                                    "Update_Date": d_update_date,
                                                    "Proper_Date": proper_date,
                                                    "Avg_Doc_Length": avg_doc_length,
                                                    "Avg_Sentence_Length" : avg_words_in_sentence,
                                                    "Avg_Sentences_in_Doc": avg_sents_in_docs,
                                                    "Avg_Text_Dynamics" : avg_text_dynamics,
                                                    "Avg_Nouns_to_Verbs" : avg_nouns_to_verbs,
                                                    "Avg_Stopwords_to_Words": avg_stopwords_to_words,
                                                    "Manifest": [d_manifesto]
                                                }, index=[idx])])

        # loop end


    # Calculations of TOTAL
    total_size_mb = round(dataframe_for_all_datasets["Size_MB"].sum(), 2)
    total_documents = dataframe_for_all_datasets["Documents"].sum()
    total_characters = dataframe_for_all_datasets["Characters"].sum()
    total_sentences = dataframe_for_all_datasets["Sentences"].sum()
    total_words = dataframe_for_all_datasets["Words"].sum()
    total_verbs = dataframe_for_all_datasets["Verbs"].sum()
    total_nouns = dataframe_for_all_datasets["Nouns"].sum()
    total_punctuations = dataframe_for_all_datasets["Punctuations"].sum()
    total_symbols = dataframe_for_all_datasets["Symbols"].sum()
    total_stopwords = dataframe_for_all_datasets["Stopwords"].sum()

    dataframe_show = dataframe_for_all_datasets.copy(deep=True)
    dataframe_show["SELECTED"] = False
    dataframe_show["Tags_p"] = dataframe_show["Tags"].apply(lambda d: [f"{key} - {value}%" for key, value in d.items()])
    dataframe_show["Tags"] = dataframe_show["Tags"].apply(lambda d: list(d.keys()))
    dataframe_show["Quality_HIGH"] = dataframe_show["Quality"].apply(lambda d: d.get('HIGH', 0))
    dataframe_show["Quality"] = dataframe_show["Quality"].apply(lambda d: [int(v * 100) for v in d.values()])

    tags_sum_table = pd.DataFrame({"Tags": tags_sum_docs.keys(), "Docs_sum": tags_sum_docs.values()}).sort_values(by="Docs_sum", ascending=False)

    end_timer = time.perf_counter()
    print(f"{datetime.now()} : // DEBUG // Func: Prepare_Data() = {(end_timer - start_timer):.3f} sec")

    return sl, dataframe_for_all_datasets, dataframe_show, tags_sum_table, total_size_mb, total_documents, total_characters, total_sentences, total_words, total_verbs, total_nouns, total_punctuations, total_symbols, total_stopwords


### Init

sl, dataframe_for_all_datasets, dataframe_show, tags_sum_table, total_size_mb, total_documents, total_characters, total_sentences, total_words, total_verbs, total_nouns, total_punctuations, total_symbols, total_stopwords = Prepare_Data(datetime.now().strftime("%m-%d-%Y"))

# Debug table
# st.dataframe(dataframe_show)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0%;
                    padding-bottom: 0%;
                    padding-left: 3%;
                    padding-right: 3%;
                    scrollbar-width: thin;
                }
        </style>
        """, unsafe_allow_html=True)

### Prepare layout
st.subheader("")

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
        text-align: center;
        font-size: 0.9em;
        margin-left: 0%;
        margin-right: 0%;
        overflow-x: auto;
        -ms-overflow-style: 3px;  /* Internet Explorer 10+ */
        scrollbar-width: thin;  /* Firefox */
    }
    .table-responsive::-webkit-scrollbar { 
        /*display: none;*/  /* Safari and Chrome */
        width: 6px;
    }

    #table_id {
      display: block;
    }

    #table_id th {
      display: inline-block;
    }

    #table_id td {
        padding-left: 0.7rem; 
        padding-right: 0.7rem;
        display: inline-block;
    }
    #table_id td:hover {
        color:#FDA428;
    }

    a:link {color:#A85E00;}      /* unvisited link */
    a:hover {color:#FDA428;}   /* Mouse over link */
    a:visited {color:#A85E00;}  /* visited link */
    a:active {color:#A85E00;}  /* selected link */

    .image-container {
      position: relative;
      display: inline-block;
      transition: transform 0.3s ease;
    }

    .image-container img {
      vertical-align: middle;
    }

    .image-container::after {
      content: "";
      position: absolute;
      left: 0;
      bottom: 0;
      width: 100%;
      height: 2px;
      background-color: #FDA428; /* Change this to your desired color */
      transform: scaleX(0);
      transition: transform 0.3s ease;
    }

    .image-container:hover {
      transform: translateY(-3px); /* Change the value to adjust the upward movement */
    }

    .image-container:hover::after {
      transform: scaleX(1);
    }

/* ---------------------------------------------------------------- */
</style>
""", unsafe_allow_html=True)

# --- Colors info ---
# Primary Color: #FDA428
# Secondary Color: #A85E00
# Grey Color: #7B7B7B
# Background Color: #1C1C1C
# {'LOW': '#7B7B7B', 'MEDIUM': '#A85E00', 'HIGH': '#FDA428'}
# ----------------------------------------------------------

### Row: 1 --> Title + links to SpeakLeash.org website / GitHub / X (Twitter)
st.write(
"""
<html>
<div style="text-align: right; font-size: 1.3em; font-family: 'Inter var', sans-serif; color:#A85E00;">
<a href="https://speakleash.org" target='_blank' rel='noopener noreferrer'>speakleash.org</a>
<a href="https://speakleash.streamlit.app/" target='_blank' rel='noopener noreferrer'><img class='image-container' src="https://streamlit.io/images/brand/streamlit-mark-light.svg" alt="Streamlit App" height="25"></a>&nbsp;
<a href="https://github.com/speakleash" target='_blank' rel='noopener noreferrer'><img class='image-container' src="./app/static/github-mark-white.png" alt="GitHub" width="30" height="30"></a>&nbsp;
<a href="https://twitter.com/Speak_Leash" target='_blank' rel='noopener noreferrer'><img class='image-container' src="./app/static/X-logo-white.png" alt="X (Twitter)" width="28" height="28"></a>
<br>
<!-- <p style="font-size: 0.8em;">Streamlit standalone</p> -->
</div>
</html>
""",unsafe_allow_html=True)

row0_1, row0_2 = st.columns([0.8,0.2])

# Website Title with: SpeakLeash a.k.a. Spichlerz Datasets Dashboard
row0_1.markdown("""<html><a href='https://speakleash.streamlit.app/' target='_blank' rel='noopener noreferrer'><h1 style='color: #FDA428; margin-top: -1rem; font-size: 3.1em;'>SpeakLeash a.k.a. Spichlerz <br>Datasets Dashboard <sub><img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" height="15"></sub></h1></a></html>""", unsafe_allow_html=True) # <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-lighttext.svg" height="25">

### Row: 2 --> Project Info + data acquisition timeline + Project data progress Indicator
row1_1a, row1_1b = st.columns([0.5, 0.5])

# Project Info & timeline Bar Chart
with row1_1a:
    # Info about Project
    st.markdown("""<div style="text-align: center; font-size: 1em; font-family: 'Source Sans Pro', sans-serif; margin-left: 0%; margin-right: 5%;">
                    <i>"SpeakLeash is an open collaboration project to build datasets for Language Modeling with a capacity of at least 1TB containing diverse texts in Polish. Our aim is to enable machine learning research and to train a Generative Pre-trained Transformer Model from collected data."</i></div>""", unsafe_allow_html=True)

    add_vertical_space()

    @st.cache_data()
    def BarChart_Timeline(dataframe_show = dataframe_show):
        """
        Create Timeline with collected datasets per each month.
        """

        start_timer = time.perf_counter()
        print(f"{datetime.now()} : // DEBUG // Func: BarChart_Timeline()")

        dataframe_with_dates = dataframe_show[dataframe_show['Proper_Date'] is True]
        dataframe_with_dates = dataframe_with_dates[["Dataset", "Size_MB", "Category", "Documents", "Creation_Date"]]

        # Calculation for MONTHs aggregation
        dataframe_grouped = dataframe_with_dates.groupby(pd.Grouper(key='Creation_Date', freq='M')).agg({'Dataset': ['count', list], 'Documents': 'sum'})
        dataframe_grouped['Creation_Date_Placeholder'] = dataframe_grouped.index.strftime('%Y-%m-01')
        dataframe_grouped.reset_index(drop=True, inplace=True)
        dataframe_grouped.columns = ['Total_Datasets', 'Datasets', 'Total_Documents', 'Creation_Date_Placeholder']
        dataframe_grouped = dataframe_grouped[['Creation_Date_Placeholder', 'Datasets', 'Total_Documents', 'Total_Datasets']]

        # Get last 8 months for best visuals on website
        result_table = dataframe_grouped.tail(8)

        # Don't touch this
        result_table = result_table.explode("Datasets")
        result_table = result_table.merge(dataframe_with_dates[["Dataset", "Documents", "Creation_Date"]], how="left", left_on="Datasets", right_on="Dataset")
        result_table.drop("Dataset", axis=1, inplace=True)
        result_table = result_table.drop_duplicates()

        # Plotly Bar Chart
        fig_test = px.bar(result_table, x="Creation_Date_Placeholder", y='Documents', color="Documents", hover_name="Datasets", hover_data=["Total_Documents", "Documents", "Creation_Date"],
                     labels={"Creation_Date_Placeholder": 'Months of data collection', 'Documents': 'Documents'}, height=300,
                     title="New datasets every month",
                     color_discrete_sequence=px.colors.sequential.Plasma)

        # ----------------------------------- #
        # Adding a summary value above the bars:
        # 2) Total Datasets in specific time:
        max_y = result_table["Total_Documents"].max()

        for idx, row in result_table[["Creation_Date_Placeholder", "Total_Documents", "Total_Datasets"]].drop_duplicates().iterrows():
            y_coordinate = row["Total_Documents"] + round(max_y * 0.05, 0)
            fig_test.add_annotation(x=row["Creation_Date_Placeholder"], y=y_coordinate, text=f'{(row["Total_Datasets"])}',  showarrow=False)
        # ----------------------------------- #

        # Last changes to make chart nice and clean
        fig_test.update_layout(margin={"t": 50, "b": 10, "r": 1, "l": 5}, title={'x': 0.05, 'y': 0.89})

        # Show chart on Streamlit App
        st.plotly_chart(fig_test, theme="streamlit", use_container_width=True)

        end_timer = time.perf_counter()
        print(f"{datetime.now()} : // DEBUG // Func: BarChart_Timeline() = {(end_timer - start_timer):.3f} sec")

    BarChart_Timeline(dataframe_show)


# Indicator with project progress
with row1_1b:

    @st.cache_data()
    def Gauge_Progress(total_size_mb = total_size_mb):
        """
        Create Gauge indicator with GBs of collected documents.
        """

        start_timer = time.perf_counter()
        print(f"{datetime.now()} : // DEBUG // Func: Gauge_Progress()")
        fig1_1 = go.Figure(go.Indicator(
                value = total_size_mb/1024,
                number = {'valueformat':'.2f'}, # 'font': {'size': 40} 'suffix': " GB"
                mode = "gauge+number",
                title = {'text': f"<b>Project data progress</b><br><span style='color: gray; font-size:1em'>{round(total_size_mb/1024,2)}GB of 1TB target</span>", 'font': {"size": 18}},
                gauge = {'axis': {'range': [None, 1200]},
                        'bar': {'color': "#00488B"},
                        'steps' : [
                             {'range': [0, 250], 'color': '#EE4E26'},
                             {'range': [250, 500], 'color': "#FDA428"},
                             {'range': [500, 750], 'color': "#d8c700"},
                             {'range': [750, 1000], 'color': "#9EC000"},
                             {'range': [1000, 1500], 'color': "#279D00"}],
                        'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': 1024}}))
        fig1_1.update_layout(margin={"l": 20, "r": 35, "b": 10, "t": 80}, height=400)

        st.plotly_chart(fig1_1, theme="streamlit", use_container_width=True)

        end_timer = time.perf_counter()
        print(f"{datetime.now()} : // DEBUG // Func: Gauge_Progress() = {(end_timer - start_timer):.3f} sec")

    Gauge_Progress(total_size_mb)


### Row: 3 --> Table with summary info about collected data
row1_2a = st.columns(1)[0]

with row1_2a:
    st.write('<div style="text-align: center;"><h5 style="">So far we managed to collect:</h5></div>', unsafe_allow_html=True)

    @st.cache_data()
    def Table_Collected(total_documents = total_documents, total_characters = total_characters, total_sentences = total_sentences,
                       total_words = total_words, total_verbs = total_verbs, total_nouns = total_nouns,
                       total_punctuations = total_punctuations, total_symbols = total_symbols, total_stopwords = total_stopwords):
        """
        Create Table (HTML) with selected summary statistics like e.g. total documents...
        """

        start_timer = time.perf_counter()
        print(f"{datetime.now()} : // DEBUG // Func: Table_Collected()")

        # Create dataframe for table with statistics
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

        # Prepare specific table format
        def generate_html_table(dataframe):
            html_table = "<table id='table_id' style='margin-top: -1rem; white-space: nowrap; text-align: center; border-radius: 7px;' class='table-responsive'>"
            html_table += "<tr>"
            for column in dataframe.columns:
                value = dataframe[column].values[0]
                html_table += f"<td><b>{column}</b><hr style='margin: 0.25rem;'>{value}</td>"
            html_table += "</tr>"
            html_table += "</table>"
            return html_table

        # Generate HTML table in streamlit app
        st.markdown('<div style="display: flex; justify-content: center;">' + generate_html_table(df2) + '</div>',unsafe_allow_html=True)

        end_timer = time.perf_counter()
        print(f"{datetime.now()} : // DEBUG // Func: Table_Collected() = {(end_timer - start_timer):.3f} sec")

    add_vertical_space()
    Table_Collected(total_documents = total_documents, total_characters = total_characters, total_sentences = total_sentences,
                       total_words = total_words, total_verbs = total_verbs, total_nouns = total_nouns,
                       total_punctuations = total_punctuations, total_symbols = total_symbols, total_stopwords = total_stopwords)

add_vertical_space()
add_vertical_space()


### Row: 4 --> Expander with some charts
row_expander= st.columns(1)[0]

with row_expander:

    with st.expander("More graphs with datasets distribiutions"):

        row_exp_col1, row_exp_col2 = st.columns(2)

        with row_exp_col1:

            # Chart with: Total Size of datasets by Category
            @st.cache_data()
            def Expander_Chart_1(dataframe_show = dataframe_show):
                """
                Create Bar plot with Total Size of datasets by Category.
                """
                start_timer = time.perf_counter()
                print(f"{datetime.now()} : // DEBUG // Func: Expander_Chart_1() = Total Size of datasets by Category")
                grouped_data = dataframe_show.groupby('Category').sum(numeric_only=True).reset_index()
                grouped_data["Size_GB"] = grouped_data["Size_MB"] / 1000
                fig1a_1 = px.bar(grouped_data, x='Category', y='Size_GB',text_auto='.1f', title="Total Size of datasets by Category")
                fig1a_1.update_traces(textangle=0, textposition="outside", cliponaxis=False) # marker_color='#FDA428'
                fig1a_1.update_layout(margin={"r": 10, "t": 50, "b": 10},
                                      xaxis_title='Category', yaxis_title='Total Size [GB]',
                                      title_x=0.0, title_y=0.93)
                st.plotly_chart(fig1a_1, theme="streamlit", use_container_width=True)
                end_timer = time.perf_counter()
                print(f"{datetime.now()} : // DEBUG // Func: Expander_Chart_1() = {(end_timer - start_timer):.3f} sec")
            Expander_Chart_1(dataframe_show)


        with row_exp_col2:

            # Chart with: Avg words per document by Category
            @st.cache_data()
            def Expander_Chart_2(dataframe_show = dataframe_show):
                """
                Create Box plot with Average words in documents by Category.
                """
                start_timer = time.perf_counter()
                print(f"{datetime.now()} : // DEBUG // Func: Expander_Chart_2() = Avg words in documents by Category")
                fig1a_2 = px.box(dataframe_show, x="Category", y="Avg_Doc_Length", title="Avg words in documents by Category")
                # fig1a_2.update_traces(marker_color='#FDA428')     # If we wanna chart in different color
                fig1a_2.update_layout(margin={"l": 20, "t": 50, "b": 20},title_x=0.0, title_y=0.93)
                st.plotly_chart(fig1a_2, theme="streamlit", use_container_width=True)
                end_timer = time.perf_counter()
                print(f"{datetime.now()} : // DEBUG // Func: Expander_Chart_2() = {(end_timer - start_timer):.3f} sec")
            Expander_Chart_2(dataframe_show)

        st.divider()

        top_chart_int = st.slider(label="Select number of top Tags", min_value=2, max_value=30, value=10, step=1)

        # Chart with: Sum of documents by Tags
        @st.cache_data()
        def Expander_Chart_3(tags_sum_table = tags_sum_table, top_chart_int = top_chart_int):
            """
            Create Bar plot with Sum of documents by Tags.
            """
            start_timer = time.perf_counter()
            print(f"{datetime.now()} : // DEBUG // Func: Expander_Chart_3() = Sum of documents by Tags")
            fig1a_3 = px.bar(tags_sum_table[0:top_chart_int], x='Tags', y='Docs_sum',text_auto='.2s',
                             title=f"Sum of documents by Tags - TOP {top_chart_int} [+ {tags_sum_table.shape[0] - top_chart_int} more Tags]")
            fig1a_3.update_layout(xaxis_title='Tags', yaxis_title='Documents (sum)')
            fig1a_3.update_traces(textangle=0, textposition="outside", cliponaxis=False)
            fig1a_3.update_layout(margin={"r": 10, "t": 60, "b": 10},title_x=0.0, title_y=0.93)
            st.plotly_chart(fig1a_3, theme="streamlit", use_container_width=True)
            end_timer = time.perf_counter()
            print(f"{datetime.now()} : // DEBUG // Func: Expander_Chart_3() = {(end_timer - start_timer):.3f} sec")
        Expander_Chart_3(tags_sum_table, top_chart_int)

        st.divider()

        # Chart with: Average Quality in Categories
        @st.cache_data()
        def Expander_Chart_4(dataframe_show = dataframe_show):
            """
            Create Bar plot with Average Quality in Categories.
            """
            start_timer = time.perf_counter()
            print(f"{datetime.now()} : // DEBUG // Func: Expander_Chart_4() = Average Quality in Categories")
            df_chart = pd.DataFrame()
            df_chart['Category'] = dataframe_show['Category']
            df_chart[['HIGH', 'MEDIUM', 'LOW']] = pd.DataFrame(dataframe_show['Quality'].tolist()) / 100.0
            df_chart_combine = df_chart.groupby('Category').mean()
            df_chart_solo = df_chart[['HIGH', 'MEDIUM', 'LOW']].mean().to_frame('SpeakLeash').T
            df_chart_combine = pd.concat([df_chart_solo, df_chart_combine], axis=0)
            df_chart_combine = df_chart_combine.round(4)
            df_long = df_chart_combine.reset_index().melt(id_vars='index', var_name='Category', value_name='Value')
            df_long.columns = ['Category', 'Quality', 'Value']
            df_swapped = pd.concat([df_long[df_long['Quality'] == 'LOW'],
                                    df_long[df_long['Quality'] == 'MEDIUM'],
                                    df_long[df_long['Quality'] == 'HIGH']])
            color_map = {'LOW': '#7B7B7B', 'MEDIUM': '#A85E00', 'HIGH': '#FDA428'}

            fig1a_0 = px.bar(df_swapped, x='Category', y='Value', color='Quality',
                             text_auto='.0%', title="Average Quality in Categories",
                             color_discrete_map=color_map)
            fig1a_0.update_traces(textangle=0, textposition="inside", cliponaxis=False)
            fig1a_0.update_layout(xaxis_title='Category', yaxis_title='Documents Quality',
                                    margin={"r": 10, "t": 90, "b": 10},title_x=0.0, title_y=0.93,
                                    legend={"orientation": 'h', "yanchor": 'top', "y": 1.14, "x": -0.03}, legend_traceorder="reversed")
            st.plotly_chart(fig1a_0, theme="streamlit", use_container_width=True)
            end_timer = time.perf_counter()
            print(f"{datetime.now()} : // DEBUG // Func: Expander_Chart_4() = {(end_timer - start_timer):.3f} sec")

        Expander_Chart_4(dataframe_show)

add_vertical_space()
st.subheader("", divider="orange")


### Row: 5 --> Tabs with search, compare and RAW info about datasets
st.markdown("<html><h2 style='color: #FDA428;'><b>Search for data you need  <span style='font-size: 0.7em;'>🔍</span></b></h2></html>", unsafe_allow_html=True)

tab_search, tab_compare, tab_RAW = st.tabs(["Search Datasets...", "Selected...", "RAW Table..."])

### Row: 5.1.1 --> Search Tab
with tab_search:

    row_search_1, row_search_2, row_search_3, row_search_4, row_search_5 = st.columns(5)

    with row_search_1:
        search_by_name = st.multiselect(label="Search by name:", placeholder="e.g. wolne_lektury_corpus", options=sorted(dataframe_show["Dataset"]))

        if not search_by_name:
            search_by_name = dataframe_show["Dataset"]

    with row_search_2:
        search_by_category = st.multiselect(label="Search by category:", placeholder="e.g. Forum", options=sorted(set(dataframe_show["Category"])))

        if not search_by_category:
            search_by_category = dataframe_show["Category"]

    with row_search_3:
        search_by_tags = st.multiselect(label="Search by tags:", placeholder="e.g. Bankowość", options=sorted(set(dataframe_show["Tags"].explode())))

        if not search_by_tags:
            search_by_tags = list(set(dataframe_show["Tags"].explode()))

    with row_search_4:
        search_by_quality = st.slider(label="Volume of High-Quality Docs [%]", min_value=0, max_value=100, value=(0,100))


    col_order = ["SELECTED", "Dataset", "Size_MB", "Category", "Tags_p", "Documents", "Characters", "Avg_Doc_Length", "Quality_HIGH", "Update_Date"]
    col_add_options = dataframe_show.columns.difference(col_order)

    with row_search_5:
        search_add_column = st.multiselect(label="Select more columns:", placeholder="e.g. Quality", options=col_add_options)

    col_order.extend(search_add_column)

    # print("--------------------- BY NAME")
    # print(search_by_name)
    # print("--------------------- BY CATEGORY")
    # print(search_by_category)
    # print("--------------------- BY TAGS")
    # print(search_by_tags)
    # print("--------------------- BY QUALITY")
    # print(search_by_quality)
    # print("--------------------- ADD COLUMN")
    # print(search_add_column)

    dataframe_show = st.data_editor(dataframe_show.loc[
                                                        (dataframe_show["Dataset"].isin(search_by_name))
                                                        & (dataframe_show["Category"].isin(search_by_category))
                                                        & (dataframe_show["Tags"].apply(lambda slowa: any(slowo in slowa for slowo in search_by_tags)))
                                                        & (dataframe_show["Quality_HIGH"] > (search_by_quality[0] / 100))
                                                        & (dataframe_show["Quality_HIGH"] < (search_by_quality[1] / 100))],
                    column_config={
                        "Dataset": st.column_config.TextColumn("Dataset", help="Datasets name"),
                        "Size_MB": st.column_config.NumberColumn("Size [MB]",format="%d", help="Dataset size in MB"),  # format="%d"
                        "Category": st.column_config.TextColumn("Category", help="Broader category"),
                        "Tags": st.column_config.ListColumn("Tags", help="Tags related to categorised documents (Tags added if >1% documents)", width='small'),
                        "Tags_p": st.column_config.ListColumn("Tags [%]", help="Dataset Tags related to categorised documents (Tags added if >1% documents) - sorted by percentage", width='small'),
                        "Documents": st.column_config.NumberColumn("Documents", help="Number of documents in Dataset"),
                        "Characters": st.column_config.NumberColumn("Characters", help="Number of charaters in Dataset"),
                        "Avg_Doc_Length": st.column_config.NumberColumn("Avg Docs Length", help="Average is calculated with = words / docs", format="%d"),
                        "Quality_HIGH": st.column_config.ProgressColumn("High Quality Docs", help="Volume of high quality documents in dataset", min_value=0, max_value=1),
                        "Quality": st.column_config.BarChartColumn("Quality = High | Medium | Low", help="Documents quality distribution", y_min=0, y_max=100),
                        "Creation_Date": st.column_config.DateColumn("Creation Date", help="Date of creation Dataset"),
                        "Update_Date": st.column_config.DateColumn("Update Date", help="Date of updating Dataset - should be updated from time to time"),
                        "SELECTED": st.column_config.CheckboxColumn("✔", help="Select Datasets to compare in second tab")
                    },
                    column_order = col_order, # "Quality", "Creation_Date",
                    disabled = dataframe_show.columns.drop("SELECTED"),
                    hide_index=True,
                    use_container_width=True)

    add_vertical_space()
    add_vertical_space()


    ### Row: 5.1.2 --> Get random documents
    st.subheader("* Random document (max 200 characters):", divider="gray")
    search_get_random_docs = st.multiselect(label="Select Dataset to get random document:", placeholder="Select Dataset to view a random document...", options=dataframe_show["Dataset"])

    if search_get_random_docs:
        for dataset_random_doc in search_get_random_docs:
            ds = sl.get(dataset_random_doc).samples
            for idx, doc in enumerate(ds):
                txt = doc['text']
                meta = doc['meta']
                if idx == 0:
                    break

            st.write("<html><u><b>Random document from :</b></u> &nbsp</html>", dataset_random_doc, unsafe_allow_html = True)
            st.write(ftfy.fix_encoding(txt[:200] + " [...]"))
            st.write("<html><u><b>Metadata for this document :</b></u> &nbsp</html>", unsafe_allow_html = True)
            st.json(meta, expanded=False)
            st.write("---")


### Row: 5.2.1 --> Compare Tab
with tab_compare:

    num_rows = dataframe_show.loc[dataframe_show["SELECTED"] is True].shape[0]

    col_table, col_manifest = st.columns([0.99, 0.01])

    with col_table:
        add_vertical_space()
        add_vertical_space()

        st.data_editor(dataframe_show.loc[dataframe_show["SELECTED"] is True],
                    column_config={
                        "Dataset": st.column_config.TextColumn("Dataset", help="Datasets name"),
                        "Size_MB": st.column_config.NumberColumn("Size [MB]", format="%d", help="Dataset size in MB"),  # format="%d"
                        "Category": st.column_config.TextColumn("Category", help="Broader category"),
                        "Tags": st.column_config.ListColumn("Tags", help="Tags related to categorised documents (Tags added if >1% documents)", width='small'),
                        "Tags_p": st.column_config.ListColumn("Tags [%]", help="Dataset Tags related to categorised documents (Tags added if >1% documents) - sorted by percentage", width='small'),
                        "Documents": st.column_config.NumberColumn("Documents", help="Number of documents in Dataset"),
                        "Characters": st.column_config.NumberColumn("Characters", help="Number of charaters in Dataset"),
                        "Avg_Doc_Length": st.column_config.NumberColumn("Avg Words / Doc", help="Average is calculated with = words / docs", format="%d"),
                        "Quality_HIGH": st.column_config.ProgressColumn("High Quality Docs", help="Volume of high quality documents in dataset", min_value=0, max_value=1),
                        "Quality": st.column_config.BarChartColumn("Quality = High | Medium | Low", help="Documents quality distribution", y_min=0, y_max=100),
                        "Creation_Date": st.column_config.DateColumn("Creation Date", help="Date of creation Dataset - should be updated from time to time"),
                        "Update_Date": st.column_config.DateColumn("Update Date", help="Date of updating Dataset - should be updated from time to time"),
                        "SELECTED": st.column_config.CheckboxColumn("Compare", help="Select Datasets to compare in second tab")
                    },
                    column_order = col_order, # "Quality", "Creation_Date",
                    disabled = dataframe_show.columns,
                    hide_index=True,
                    use_container_width=True)

    add_vertical_space()


    with st.expander("Some comparison charts..."):
    # with col_manifest:
        if dataframe_show.loc[dataframe_show["SELECTED"] is True].shape[0] > 0:
            theta = ["Avg_Doc_Length", "Avg_Sentence_Length", "Avg_Sentences_in_Doc", "Avg_Text_Dynamics", "Avg_Nouns_to_Verbs", "Avg_Stopwords_to_Words"]
            comp_df = pd.DataFrame()

            for idx, row in dataframe_show.loc[dataframe_show["SELECTED"] is True].iterrows():
                r_r = row[theta] / (dataframe_show[theta].sum() / len(sl.datasets))
                r_data = pd.DataFrame({"r": r_r[theta].T, "theta": theta}).reset_index(drop=True)
                r_data["Dataset"] = row["Dataset"]
                comp_df = pd.concat([comp_df, r_data])
            comp_df['theta'] = comp_df['theta'].str.replace("_"," ")

            fig_comp = px.line_polar(comp_df, r="r", theta="theta", color="Dataset",
                   color_discrete_sequence= px.colors.sequential.Plasma_r, line_close=True, template="plotly_dark",
                   title="Comparison of selected metrics to the average value in all datasets (avg = 1)")
            # fig_comp.update_layout(legend=dict(orientation="h"))
            # fig_comp.update_layout(margin=dict(l=150, r=150),)
            st.plotly_chart(fig_comp, theme="streamlit", use_container_width=False)

    add_vertical_space()
    add_vertical_space()


    ### Row: 5.2.2 --> Datasets Manifests
    st.subheader("* Datasets manifests:", divider="gray")
    for idx, row in dataframe_show.loc[dataframe_show["SELECTED"] is True].iterrows():
        r1, r2 = st.columns(2)
        with r1:
            st.write("Dataset: ",row["Dataset"])
        with r2:
            st.json(row["Manifest"], expanded=False)

    add_vertical_space()
    add_vertical_space()


    ### Row: 5.2.3 --> Get Random Documents
    st.subheader("* Random document from selected Datasets (max 200 chars):", divider="gray")
    search_get_random_docs_comp = st.multiselect(label="Select Dataset to get random documents:", placeholder="Select Dataset to see a random documents...", options=dataframe_show.loc[dataframe_show["SELECTED"] is True, "Dataset"])

    if search_get_random_docs_comp:
        for dataset_random_doc in search_get_random_docs_comp:
            ds = sl.get(dataset_random_doc).samples
            for idx, doc in enumerate(ds):
                txt = doc['text']
                meta = doc['meta']
                if idx == 0:
                    break

            st.write("<html><u><b>Random document from :</b></u> &nbsp</html>", dataset_random_doc, unsafe_allow_html = True)
            st.write(ftfy.fix_encoding(txt[:200] + " [...]"))
            st.write("<html><u><b>Metadata for this document :</b></u> &nbsp</html>", unsafe_allow_html = True)
            st.json(meta, expanded=False)
            st.write("---")



### Row: 5.3.1 --> RAW Table tab
with tab_RAW:

    # TODO! : revisit column "Tags" - streamlit sort dictinary inside dataframe
    st.dataframe(dataframe_for_all_datasets, column_config={'Tags': st.column_config.Column()}, use_container_width=True)

    empty_quality = dataframe_for_all_datasets[dataframe_for_all_datasets["Quality"] == {}]
    if len(empty_quality) > 0:
        for ind, row in enumerate(empty_quality['Dataset']):
            st.error(f"Empty Quality Index! Check index: {empty_quality.index[ind]} → Dataset: {row}", icon="🚨")

    false_rows = dataframe_for_all_datasets[dataframe_for_all_datasets["Proper_Date"] is False]
    if len(false_rows) > 0:
        empty_dates = ""
        for ind, row in enumerate(false_rows['Dataset']):
            # st.warning(f"Index: {false_rows.index[ind]} | Dataset: {row}", icon="⚠️")
            empty_dates = empty_dates + f" '{row}' |"

        st.warning(f"WARNING! Please ensure to carefully check the manifests before proceeding (based on 'Proper_Date' column): {empty_dates}", icon="⚠️")
        # st.error(f"WARNING! Please ensure to carefully check the manifests before proceeding (based on 'Proper_Date' column): {empty_dates}", icon="🚨")

    ### End of RAW Table


add_vertical_space()
add_vertical_space()


### Row: 6 --> Dot with JSON

### --- JSON for GitHub badge --- ###

file_path = "speakleash_data.json"

@st.cache_data(ttl = 3600)
def get_json_badge(file_path = file_path):
    """
    Create JSON with some values like: datasets count, datasets GBs.
    - for custom badge on GitHub SpeakLeash library repositiory.
    """

    time_now = datetime.now(timezone.utc)

    data_to_json = {
        "datasetsCOUNT": str(int(dataframe_for_all_datasets.shape[0])),
        "datasetsGB": str(int(round(total_size_mb / 1024, 0))),
        "updateUTCdate": str(datetime.strftime(time_now,"%Y-%m-%d %H:%M %z")),
    }

    if os.path.exists(f"./static/{file_path}"):
        with open(f"./static/{file_path}", "r", encoding='Utf-8') as json_file:
            existing_data = json.load(json_file)

        last_update_utc = existing_data.get("updateUTCdate", "")

        if last_update_utc:
            last_update_time = datetime.strptime(last_update_utc, "%Y-%m-%d %H:%M %z")
        else:
            last_update_time = time_now - timedelta(days = 1)

        time_difference = time_now - last_update_time

        if time_difference.total_seconds() > 3600:
            print(f"{time_now} --> Saving JSON file --> Data: {data_to_json}")
            with open(f"./static/{file_path}", "w", encoding='Utf-8') as json_file:
                json.dump(data_to_json, json_file)
    else:
        print(f"{time_now} --> Saving JSON file (1-st time) --> Data: {data_to_json}")
        with open(f"./static/{file_path}", "w", encoding='Utf-8') as json_file:
            json.dump(data_to_json, json_file)

get_json_badge(file_path)
st.markdown(f'<html><a href="./app/static/{file_path}" style="color: #FDA428;">.</a></html>', unsafe_allow_html=True)
