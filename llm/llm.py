import os
from langchain_google_genai import ChatGoogleGenerativeAI


def prepare_llm():
    """Create and return the Gemini LLM."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    return llm


def call_llm(llm, prompt):
    """Send a prompt to the LLM and return the response."""
    response = llm.invoke(prompt)

    return response.content

