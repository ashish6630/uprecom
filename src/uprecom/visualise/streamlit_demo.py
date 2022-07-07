import streamlit as st
from uprecom.api.request_handler import product_recommender

form = st.form(key="query_text")
query_text = form.text_input(label="Enter job related keywords")
submit_button = form.form_submit_button("Submit")

if submit_button:
    job_result = product_recommender(query_text=query_text)
    st.dataframe(job_result)
