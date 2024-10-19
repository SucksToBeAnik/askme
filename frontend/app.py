import streamlit as st

# pages
pdf_query = st.Page(
    page="routes/pdf_query.py",
    title="Ask PDF",
    icon=":material/description:",
    default=True,
)
general_query = st.Page(
    page="routes/general_query.py",
    title="Ask Anything",
    url_path="/ask",
    icon=":material/person_raised_hand:",
)


pg = st.navigation({"Projects": [pdf_query, general_query]})

pg.run()

# # Title for the Streamlit app
# st.title("Welcome to AskMe")
