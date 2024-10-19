import streamlit as st
import requests

API_URL = st.secrets["API_URL"]

st.title("Ask Anything")

query = st.text_input(label="Your query", placeholder="Why Marvel sucks recently?")
if st.button("Ask Me"):
    if query:
        try:
            response = requests.post(f"{API_URL}/query", data={"query": query})
            # print(response.json())
            if response.status_code == 200:
                st.success(f"{response.json()}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query before submitting.")
