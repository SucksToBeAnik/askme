import streamlit as st
import requests

API_URL = st.secrets["API_URL"]

st.title("ASK PDF")


uploaded_file = st.file_uploader("Upload PDF", type="pdf")


if st.button("Upload") and uploaded_file is not None:
    try:
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}

        response = requests.post(f"{API_URL}/file", files=files)

        if response.status_code == 200:
            st.toast("File uploaded successfully", icon=":material/check:")
            st.write(response.json()["message"])
        else:
            st.toast(
                f"Error: {response.status_code} - {response.text}",
                icon=":material/error:",
            )
    except Exception as e:
        st.toast(f"An error occurred: {e}", icon=":material/error:")

query = st.text_input("Your query",placeholder="I am an expert research assistant. Ask me about your paper!")

if st.button("Ask Me"):
    # Only proceed if the input is not empty

    if query:
        try:
            response = requests.post(f"{API_URL}/file/query", data={"query": query})
            # Check if the request was successful
            if response.status_code == 200:
                # Display the response from the FastAPI backend
                st.success(f"{response.json()["answer"]}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query before submitting.")
