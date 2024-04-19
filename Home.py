import streamlit as st

st.set_page_config(
    page_title="Elliot's Portfolio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="auto"
)

st.header("Elliot Chung")
st.subheader("Penultimate Student at Singapore Management University")
st.write("Currently looking for internship opportunities in the field of data science and quantitative finance.")
col1, col2 = st.columns(2)
with col1:
  st.write("Feel free to contact me through email:")
with col2:  
  st.code("elliotcky@gmail.com")
st.divider()

st.subheader("Favourite Courses Taken")
st.write("1. Game Theory")
st.write("2. Quantitative Finance")
st.write("3. Computational Thinking")
st.write("4. Macroeconomics")
st.divider()

col3, col4 = st.columns(2)
with col3:
  st.subheader("Completed Projects")
  st.write("""
           1. Mean-Variance Optimization using historical data
           2. algoline: A simple algorithmic trendline drawing tool
           """)

with col4:
  st.subheader("Ongoing Projects")
  st.write('''
              1. Portfolio Optimization with Machine Learning  
              2. FOMC Meeting Minutes Sentiment Analysis
              ''')
