import os
import random
from datetime import datetime, timedelta, timezone
import json
# import time

from speakleash import Speakleash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ftfy
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.grid import grid
from stqdm import stqdm


# Initialization
# if 'data_show' not in st.session_state:
#     st.session_state.data_show = pd.DataFrame()



st.set_page_config(page_title="Speakleash Dashboard", layout="wide", page_icon="http://speakleash.org/wp-content/uploads/2022/12/cropped-sl-favico-black-32x32.png")

print("\n--- START --- START --- START --- START --- START --- START --- START ---\n")


@st.cache_data()
def prepare_data(date_string):

    #Dummy datetime input string to reset cache daily. Different string triggers cache refresh

    base_dir = os.path.join(os.path.dirname(__file__))
    replicate_to = os.path.join(base_dir, "datasets")
    sl = Speakleash(replicate_to)

    dataframe_for_all_datasets = pd.DataFrame()

    # Gather info about every dataset
    for id, d in stqdm(enumerate(sl.datasets)):
        # print(sl.get(d.name).manifest)

        punctuations = d.punctuations
        if isinstance(punctuations, list):
            d_punctuations = len(punctuations)
        else:
            d_punctuations = punctuations

        symbols = d.symbols
        if isinstance(symbols, list):
            d_symbols = len(symbols)
        else:
            d_symbols = symbols

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

        tags = {}
        manifesto = sl.get(d.name).manifest

        try:
            dici = manifesto.get('category=95%')
            dici_sort = sorted(dici.items(), key=lambda x: (x[1],x[0]), reverse=True)

            for x in dici_sort:
                calc_temp = round(x[1] / d.documents * 100, 2)
                if (calc_temp > 1):
                    tags[x[0]] = calc_temp
                else:
                    break
        except:
            tags = {"Inne": d.documents / d.documents * 100}

        if tags:
            pass
        else:
            tags = {"Rożne": d.documents / d.documents * 100}


        # Prepare DataFrame with info about Dataset
        dataframe_for_all_datasets = pd.concat([dataframe_for_all_datasets, pd.DataFrame({
                                                    "Dataset": d.name,
                                                    "Size_MB": round(d.characters/1024/1024),
                                                    "Category": d.category,
                                                    "Tags": [tags],
                                                    "Documents": d.documents,
                                                    "Characters": d.characters,
                                                    "Sentences": d.sentences,
                                                    "Words": d.words,
                                                    "Verbs": d.verbs,
                                                    "Nouns": d.nouns,
                                                    "Punctuations": d_punctuations,
                                                    "Symbols": d_symbols,
                                                    "Stopwords": d.stopwords,
                                                    "Quality": [d.quality],
                                                    "Description": d.description,
                                                    "License": d.license,
                                                    "Sources": d.sources,
                                                    "Creation_Date": datetime.fromisoformat('2022-12-01') + timedelta(days=id),  # d.creation_date,
                                                    "Avg_Doc_Length": avg_doc_length,
                                                    "Avg_Sentence_Length" : avg_words_in_sentence,
                                                    "Avg_Sentences_in_Doc": avg_sents_in_docs,
                                                    "Avg_Text_Dynamics" : avg_text_dynamics,
                                                    "Avg_Nouns_to_Verbs" : avg_nouns_to_verbs,
                                                    "Avg_Stopwords_to_Words": avg_stopwords_to_words,
                                                    "Manifest": [manifesto]
                                                }, index=[id])])


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
    dataframe_show["Tags"] = dataframe_show["Tags"].apply(lambda d: list(d.keys()))
    dataframe_show["Quality_HIGH"] = dataframe_show["Quality"].apply(lambda d: d.get('HIGH', 0))
    dataframe_show["Quality"] = dataframe_show["Quality"].apply(lambda d: [v * 100 for v in d.values()])

    # st.session_state.data_show = dataframe_show

    return sl, dataframe_for_all_datasets, dataframe_show, total_size_mb, total_documents, total_characters, total_sentences, total_words, total_verbs, total_nouns, total_punctuations, total_symbols, total_stopwords


#Init

sl, dataframe_for_all_datasets, dataframe_show, total_size_mb, total_documents, total_characters, total_sentences, total_words, total_verbs, total_nouns, total_punctuations, total_symbols, total_stopwords = prepare_data(datetime.now().strftime("%m-%d-%Y"))

# print(f"TUTAJ PO FUNKCJI: {st.session_state.data_show}")

#Prepare layout

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
        font-size: 0.9em;
        margin-left: auto;
        margin-right: auto;
        max-width: 100%;
        overflow-x: auto;
    }

    a:hover {
        color:#A85E00;
    }   /* Mouse over link */


</style>
""", unsafe_allow_html=True)


# Primary Color: #FDA428
# Secondary Color: #A85E00


row0_1, row0_2 = st.columns([0.6,0.4])

# row0_1.title("Speakleash a.k.a. Spichlerz Datasets Dashboard")
row0_1.markdown("<html><a href='https://speakleash-dashboard-dev-samox-repo-2.streamlit.app/' target='_blank' rel='noopener noreferrer'><h1 style='color: #FDA428;'>Speakleash a.k.a. Spichlerz Datasets Dashboard <sub>STREAMLIT APP</sub></h1></a></html>", unsafe_allow_html=True)


with row0_2:
    add_vertical_space()
    st.write(
    """
    <html>
    <div style="text-align: right; font-size: 1.5em;">
    <a href="https://speakleash.org" target='_blank' rel='noopener noreferrer'>speakleash.org</a>
    | 
    <a href="https://github.com/speakleash" target='_blank' rel='noopener noreferrer'>GitHub</a>
    |
    <a href="https://twitter.com/Speak_Leash" target='_blank' rel='noopener noreferrer'>Twitter</a>
    <br>
    <!-- <p style="font-size: 0.8em;">Streamlit standalone</p> -->
    </div>
    </html>
    """,unsafe_allow_html=True)

# row0_2.subheader("[speakleash.org](https://speakleash.org/) | [GitHub](https://github.com/speakleash) | [Twitter](https://twitter.com/Speak_Leash)")



# background-image: linear-gradient(to right, #d83232, #d76300, #c08f00, #94b600, #32d832);

# fig1a_1 = px.pie(df, values='size', names=df.index)

# fig2a_1 = px.bar(df, x=df.index, y='size')


row1_1a, row1_1b = st.columns([0.5, 0.5])

with row1_1a:
    # with st.container():
    #     st.markdown('<br>', unsafe_allow_html=True)
    # row1_1a_1, row1_1a_2, row1_1a_3 = st.columns([0.1,0.8,0.1])
    # with row1_1a_2:
        # st.write("""<html><div style="text-align: center; ">An open collaboration project to build a data set for Language Modeling with a capacity of at least 1TB comprised of diverse texts in Polish. Our aim is to enable machine learning research and to train a Generative Pre-trained Transformer Model from collected data</div></html>""", unsafe_allow_html=True)
    add_vertical_space()
    add_vertical_space()
    caption_grid = grid([1], vertical_align="center")
    caption_grid.markdown("""_"An open collaboration project to build a data set for Language Modeling with a capacity of at least 1TB comprised of diverse texts in Polish. Our aim is to enable machine learning research and to train a Generative Pre-trained Transformer Model from collected data."_""")
    # st.markdown('<div class="center-text"><h2>Summary</h2></div>', unsafe_allow_html=True)
    
    # streamlit_timeline.timeline(dataframe_for_all_datasets["Creation_Date"])

# TEST MOJ
    weeks_dates = pd.date_range(start=dataframe_show["Creation_Date"].min(), end=dataframe_show["Creation_Date"].max(), freq='W-SUN')
    result_table = pd.DataFrame({"Creation_Date": weeks_dates})

    def filter_and_sum(row):
        mask = (dataframe_show["Creation_Date"] > row["Creation_Date"] - pd.DateOffset(days=7)) & \
               (dataframe_show["Creation_Date"] <= row["Creation_Date"])
        filtered_data = dataframe_show[mask]
        datasets = filtered_data["Dataset"].tolist()
        total_documents = filtered_data["Documents"].sum()  # Sumowanie dokumentów
        return pd.Series({"Datasets": datasets, "Total_Documents": total_documents})

    result_table[["Datasets", "Total_Documents"]] = result_table.apply(filter_and_sum, axis=1)

    result_table = result_table.explode("Datasets")
    result_table = result_table.merge(dataframe_show[["Dataset", "Documents"]], how="left", left_on="Datasets", right_on="Dataset")
    result_table.drop("Dataset", axis=1, inplace=True)
    result_table = result_table.drop_duplicates()

    fig_test = px.bar(result_table, x='Creation_Date', y='Documents', color="Documents", hover_name="Datasets", hover_data=["Total_Documents", "Documents"],
                 labels={'Creation_Date': 'Weeks', 'Documents': 'Total Documents'}, height=300,
                 color_discrete_sequence=px.colors.sequential.Plasma)

    # Dodanie sumarycznej wartości nad słupkami
    # for idx, row in result_table.iterrows():
    #     fig_test.add_annotation(x=row["Creation_Date"], y=(row["Total_Documents"]+3e5), text=f'{(row["Total_Documents"] / 10e5):.1f}M',  showarrow=False)
    
    fig_test.update_layout(margin=dict(t=10, b=10))
    
    st.plotly_chart(fig_test, theme="streamlit", use_container_width=True)


with row1_1b:
    fig1_1 = go.Figure(go.Indicator(
    value = total_size_mb/1024,
    number = {'valueformat':'.2f'}, # 'font': {'size': 40} 'suffix': " GB"
    mode = "gauge+number",
    title = {'text': "<b>Project data progress</b><br><span style='color: gray; font-size:1em'>#GB of 1TB target</span>", 'font': {"size": 15}},
    gauge = {'axis': {'range': [None, 1200]},
            'bar': {'color': "#00488B"},
            'steps' : [
                 {'range': [0, 250], 'color': '#EE4E26'},
                 {'range': [250, 500], 'color': "#FDA428"},
                 {'range': [500, 750], 'color': "#d8c700"},
                 {'range': [750, 1000], 'color': "#9EC000"},
                 {'range': [1000, 1500], 'color': "#279D00"}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': 1024}}))

    fig1_1.update_layout(margin=dict(l=20, r=50, b=10))

    # fig1_1.update_layout(height = 400)
    st.plotly_chart(fig1_1, theme="streamlit", use_container_width=True)



row1_2a = st.columns(1)[0]
with row1_2a:
  st.write('<div style="text-align: center"><h4>So far we managed to collect:</h4></div>', unsafe_allow_html=True)
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
  st.markdown('<div style="display: flex; justify-content: center;">' + df2.to_html(col_space="auto",justify='center',classes='table-responsive', index=False).replace('<table', '<table style="white-space: nowrap; text-align: center; overflow-x: auto;"') + '</div>',unsafe_allow_html=True)
  # columns=['Total documents','Total words','Total characters']
  # st.dataframe(df2, hide_index=True)

add_vertical_space()
add_vertical_space()
row_expander= st.columns(1)[0]

# print(dataframe_show)

with row_expander:
    
    with st.expander("More graphs with datasets distribiutions"):
        # st.markdown('<div class="center-text"><h2>Summary</h2></div>', unsafe_allow_html=True)
        
        row_exp_col1, row_exp_col2 = st.columns(2)
        with row_exp_col1:
            grouped_data = dataframe_show.groupby('Category').sum(numeric_only=True).reset_index()
            grouped_data["Size_GB"] = grouped_data["Size_MB"] / 1000
            fig1a_1 = px.bar(grouped_data, x='Category', y='Size_GB',text_auto='.2f', title="Total Size of all datasets by Category")
            fig1a_1.update_layout(xaxis_title='Category', yaxis_title='Total Size [GB]')
            fig1a_1.update_traces(textangle=0, textposition="outside", cliponaxis=False)
            fig1a_1.update_layout(margin=dict(r=10, t=25, b=10),title_x=0.1)
            st.plotly_chart(fig1a_1, theme="streamlit", use_container_width=True)

        with row_exp_col2:
            # tag_counts = dataframe_show["Tags"].apply(lambda x: ', '.join(x))
            # tag_counts = tag_counts.str.split(', ', expand=True).stack().value_counts()
            # print(tag_counts)
            # fig1a_2 = px.pie(tag_counts, values=tag_counts.values, names=tag_counts.index, title='Procentowy udział tagów')
            fig1a_2 = px.box(dataframe_show, x="Category", y="Avg_Doc_Length", title="Average words per document by Category")
            fig1a_2.update_layout(margin=dict(l=20, t=25, b=20),title_x=0.1)
            st.plotly_chart(fig1a_2, theme="streamlit", use_container_width=True)  # use_container_width=True


add_vertical_space()
st.subheader("", divider="orange")

st.markdown("<html><h2 style='color: #FDA428;'><b>Search for data you need  <span style='font-size: 0.7em;'>🔍</span></b></h2></html>", unsafe_allow_html=True)



tab_search, tab_compare, tab_RAW = st.tabs(["Search Datasets...", "Comparing Datasets...", "RAW Table..."])

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


    col_order = ["SELECTED", "Dataset", "Size_MB", "Category", "Tags", "Documents", "Characters", "Avg_Doc_Length", "Quality_HIGH"]
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

    # dataframe_show_grid = grid([1], vertical_align="center")

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
                        "Tags": st.column_config.ListColumn("Tags", help="Tags with more than 1% of documents - sorted by percentage", width='small'),
                        "Documents": st.column_config.NumberColumn("Documents", help="Number of documents in Dataset"),
                        "Characters": st.column_config.NumberColumn("Characters", help="Number of charaters in Dataset"),
                        "Avg_Doc_Length": st.column_config.NumberColumn("Avg Docs Length", help="Average is calculated with = words / docs", format="%d"),
                        "Quality_HIGH": st.column_config.ProgressColumn("High Quality Docs", help="Volume of high quality documents in dataset", min_value=0, max_value=1),
                        "Quality": st.column_config.BarChartColumn("Quality = High | Medium | Low", help="Documents quality distribution", y_min=0, y_max=100),
                        "Creation_Date": st.column_config.DateColumn("Creation Date", help="Date of creation Dataset - should be updated from time to time"),
                        "SELECTED": st.column_config.CheckboxColumn("Compare", help="Select Datasets to compare in second tab")
                    },
                    column_order = col_order, # "Quality", "Creation_Date",
                    disabled = dataframe_show.columns.drop("SELECTED"),
                    hide_index=True)
    
    # st.write("---")
    # st.subheader("", divider="orange")

    add_vertical_space()
    add_vertical_space()

    # Part -> Get Random Documents
    st.subheader("* Random document (max 200 chars):", divider="gray")
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


with tab_compare:

    num_rows = dataframe_show.loc[dataframe_show["SELECTED"] == True].shape[0]

    col_table, col_manifest = st.columns([0.99, 0.01])

    with col_table:
        add_vertical_space()
        add_vertical_space()
        st.data_editor(dataframe_show.loc[dataframe_show["SELECTED"] == True], 
                    column_config={
                        "Dataset": st.column_config.TextColumn("Dataset", help="Datasets name"),
                        "Size_MB": st.column_config.NumberColumn("Size [MB]", format="%d", help="Dataset size in MB"),  # format="%d"
                        "Category": st.column_config.TextColumn("Category", help="Broader category"),
                        "Tags": st.column_config.ListColumn("Tags", help="Tags with more than 1% of documents - sorted by percentage", width='small'),
                        "Documents": st.column_config.NumberColumn("Documents", help="Number of documents in Dataset"),
                        "Characters": st.column_config.NumberColumn("Characters", help="Number of charaters in Dataset"),
                        "Avg_Doc_Length": st.column_config.NumberColumn("Avg Words / Doc", help="Average is calculated with = words / docs", format="%d"),
                        "Quality_HIGH": st.column_config.ProgressColumn("High Quality Docs", help="Volume of high quality documents in dataset", min_value=0, max_value=1),
                        "Quality": st.column_config.BarChartColumn("Quality = High | Medium | Low", help="Documents quality distribution", y_min=0, y_max=100),
                        "Creation_Date": st.column_config.DateColumn("Creation Date", help="Date of creation Dataset - should be updated from time to time"),
                        "SELECTED": st.column_config.CheckboxColumn("Compare", help="Select Datasets to compare in second tab")
                    },
                    column_order = col_order, # "Quality", "Creation_Date",
                    disabled = dataframe_show.columns, 
                    hide_index=True)
    

    add_vertical_space()
    
    with st.expander("Some comparison charts..."):
    # with col_manifest:
        if dataframe_show.loc[dataframe_show["SELECTED"] == True].shape[0] > 0:
            theta = ["Avg_Doc_Length", "Avg_Sentence_Length", "Avg_Sentences_in_Doc", "Avg_Text_Dynamics", "Avg_Nouns_to_Verbs", "Avg_Stopwords_to_Words"]
            comp_df = pd.DataFrame()
            
            for idx, row in dataframe_show.loc[dataframe_show["SELECTED"] == True].iterrows():
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

    st.subheader("* Datasets manifests:", divider="gray")
    for idx, row in dataframe_show.loc[dataframe_show["SELECTED"] == True].iterrows():
        r1, r2 = st.columns(2)
        with r1:
            st.write("Dataset: ",row["Dataset"])
        with r2:
            st.json(row["Manifest"], expanded=False)

    add_vertical_space()
    add_vertical_space()

    # Part -> Get Random Documents
    st.subheader("* Random document from selected Datasets (max 200 chars):", divider="gray")
    search_get_random_docs_comp = st.multiselect(label="Select Dataset to get random documents:", placeholder="Select Dataset to see a random documents...", options=dataframe_show.loc[dataframe_show["SELECTED"] == True, "Dataset"])
    
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


    # if num_rows < 5:
    #     for _ in range(2 - num_rows):
    #         add_vertical_space()


with tab_RAW:
    st.dataframe(dataframe_for_all_datasets)


add_vertical_space()
add_vertical_space()

### --- JSON for GitHub badge --- ###

# file_path = "speakleash_data.json"
#
# with open(f"./static/{file_path}", "r") as json_file:
#     kappa = json.load(json_file)
# # print(f"{kappa=}")
# 
# time_now = datetime.now(timezone.utc)
# data_to_json = {"datasetsCOUNT": str(int(dataframe_for_all_datasets.shape[0])), "datasetsGB": str(int(round(total_size_mb/1024,0))), "updateUTCdate": datetime.strftime(time_now,"%Y-%m-%d %H:%M %z")}
# # print(f"{data_to_json=}")
# 
# with open(f"./static/{file_path}", "w") as json_file:
#     json.dump(data_to_json, json_file)
# 
# st.markdown(f'<html><a href="./app/static/{file_path}" style="color: #FDA428;">.</a></html>', unsafe_allow_html=True)


file_path = "speakleash_data.json"
time_now = datetime.now(timezone.utc)

data_to_json = {
    "datasetsCOUNT": str(int(dataframe_for_all_datasets.shape[0])),
    "datasetsGB": str(int(round(total_size_mb / 1024, 0))),
    "updateUTCdate": str(datetime.strftime(time_now,"%Y-%m-%d %H:%M %z")),
}

if os.path.exists(f"./static/{file_path}"):
    with open(f"./static/{file_path}", "r") as json_file:
        existing_data = json.load(json_file)

    last_update_utc = existing_data.get("updateUTCdate", "")

    if last_update_utc:
        last_update_time = datetime.strptime(last_update_utc, "%Y-%m-%d %H:%M %z")
    else:
        last_update_time = time_now - timedelta(days = 1)

    time_difference = time_now - last_update_time

    if time_difference.total_seconds() > 3600:
        print(f"{time_now} --> Saving JSON file")
        with open(f"./static/{file_path}", "w") as json_file:
            json.dump(data_to_json, json_file)
else:
    print(f"{time_now} --> Saving JSON file")
    with open(f"./static/{file_path}", "w") as json_file:
        json.dump(data_to_json, json_file)

st.markdown(f'<html><a href="./app/static/{file_path}" style="color: #FDA428;">.</a></html>', unsafe_allow_html=True)