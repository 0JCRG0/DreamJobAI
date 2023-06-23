import re
from chromadb.utils import embedding_functions
import tiktoken
import logging
from dotenv import load_dotenv
import os

load_dotenv(".env")
LOGGER_PATH = os.getenv("LOGGER_PATH")

def clean_rows(s):
    if not isinstance(s, str):
        print(f"{s} is not a string! Returning unmodified")
        return s
    s = re.sub(r'\(', '', s)
    s = re.sub(r'\)', '', s)
    s = re.sub(r"'", '', s)
    s = re.sub(r",", '', s)
    return s

def openai_ef(OPENAI_API_KEY):
    openai_embedding = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=OPENAI_API_KEY,
                    model_name="text-embedding-ada-002"
                )
    return openai_embedding

def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = False,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string

def num_tokens(text: str, model: str ="gpt-3.5-turbo") -> int:
    #Return the number of tokens in a string.
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def LoggingMain():
    # Define a custom format with bold text
    log_format = '%(asctime)s %(levelname)s: \n%(message)s\n'

    # Configure the logger with the custom format
    logging.basicConfig(filename=LOGGER_PATH,
                        level=logging.INFO,
                        format=log_format)
