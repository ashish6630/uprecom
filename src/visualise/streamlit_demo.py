import streamlit as st

form = st.form(key="query_text")
query_text = form.text_input(label="Enter job related keywords")
submit_button = form.form_submit_button("Submit")

if submit_button:
    print(query_text)
