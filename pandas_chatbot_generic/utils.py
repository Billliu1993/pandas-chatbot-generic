import chardet
import pandas as pd
import streamlit as st
import json
import pdb

@st.cache_data
def load_csv_data(uploaded_file):
    # Read a portion of the file to detect encoding
    rawdata = uploaded_file.read(10000)
    result = chardet.detect(rawdata)
    encoding = result['encoding']

    # Reset the file pointer to the beginning of the file
    uploaded_file.seek(0)

    # Read the CSV file with the detected encoding
    df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=True)
    return df

@st.cache_data
def get_summary(df):
    info_dict = {
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum(),
        "Dtype": df.dtypes.astype(str)
    }
    info_df = pd.DataFrame(info_dict).reset_index(drop=True)
    return info_df

@st.cache_data
def load_json_data(uploaded_file):
    file_content = uploaded_file.read()

    # Parse the JSON data
    json_data = json.loads(file_content)

    return json_data