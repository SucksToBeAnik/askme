from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from settings import env_store


llm = ChatGroq(
    model="gemma2-9b-it", stop_sequences=None, api_key=env_store.GROQ_API_KEY
)


def answer_random_query(query: str):
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are like a reliable information soruces who will give reliable answers to the questions asked. If you are unable to answer a question, respectfully mention that.",
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": query})

    return response


def answer_query_from_pdf(query: str):
    pass
