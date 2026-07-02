import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ATTRIBUTION_PROVIDER = os.getenv("ATTRIBUTION_PROVIDER", "ollama").lower()
ATTRIBUTION_MODEL = os.getenv("ATTRIBUTION_MODEL", "qwen2.5:3b")
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "180"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))


def get_embeddings():
    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def get_chat_llm(temperature: float = 0):
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=OLLAMA_NUM_PREDICT,
        client_kwargs={"timeout": OLLAMA_REQUEST_TIMEOUT},
    )


def get_attribution_llm(temperature: float = 0):
    """Return the separately configured model used to verify citations."""
    if ATTRIBUTION_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=ATTRIBUTION_MODEL, temperature=temperature)

    if ATTRIBUTION_PROVIDER != "ollama":
        raise ValueError(
            "ATTRIBUTION_PROVIDER must be either 'ollama' or 'openai'"
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=ATTRIBUTION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=OLLAMA_NUM_PREDICT,
        client_kwargs={"timeout": OLLAMA_REQUEST_TIMEOUT},
    )
