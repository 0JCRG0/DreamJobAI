import os
import openai
import pandas as pd
import tiktoken  # for counting tokens
import ast  # for converting embeddings saved as strings back to arrays
from scipy import spatial
import pretty_errors
from displayfunction import display
from dotenv import load_dotenv

"""
Load the variables
"""

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
SAVE_PATH = os.getenv("SAVE_PATH")

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
#GPT_MODEL = "gpt-4"

"""
Load the embedded file
"""

embeddings_path = SAVE_PATH + "/jobs_test.csv"

df = pd.read_csv(embeddings_path)

# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

#print(df.head())
"""
## 2. Search

Now we'll define a search function that:
- Takes a user query and a dataframe with text & embedding columns
- Embeds the user query with the OpenAI API
- Uses distance between query embedding and text embeddings to rank the texts
- Returns two lists:
    - The top N texts, ranked by relevance
    - Their corresponding relevance scores

"""

# search function
def ids_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    #Modify this to get more jobs
    top_n: int = 20
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    #This is the query (e.g., Data Engineer) that the model will find relatednessnes
    query_embedding = query_embedding_response["data"][0]["embedding"]
    ids_and_relatednesses = [
        (row["id"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    ids_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    ids, relatednesses = zip(*ids_and_relatednesses)
    return ids[:top_n], relatednesses[:top_n]

# examples


#tiktoken function -> to count tokens
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    ids, relatednesses = ids_ranked_by_relatedness(query, df)
    #Basically giving the most relevant IDs from the previous function
    introduction = 'These are all the available jobs. Use them to find the jobs that match the skills of the user. If you cannot find a single job that might match then say "I could not find a suitable job."'
    question = f"\n\Skills of the user: {query}"
    message = introduction
    for id in ids:
        next_id = f'\n\nJob\'s ID:\n"""\n{id}\n"""'
        if (
            num_tokens(message + next_id + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_id
    return message + question


#print(query_message("I am looking for Data Engineer jobs",df,GPT_MODEL,1000))


def ask(
    #This query is your question, only parameter to fill in function
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096,
    print_message: bool = False,
) -> str:
    #Answers a query using GPT and a dataframe of relevant texts and embeddings.
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You help people find jobs that match their skills.\
        \The user will provide his 1.Experience, 2.Skills and 3.Desired Role.\
        \Ideally you will output 5-10 suitable jobs. If you find less than that still \
        \\"},
        {"role": "user", "content": message}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    response_message = response["choices"][0]["message"]["content"]
    #Tokens
    print("'\nTOTAL TOKENS USED:\n'", response['usage']['total_tokens'])
    
    #relatednesses
    ids, relatednesses = ids_ranked_by_relatedness(query=query, df=df)
    print(f"\nTHE IDs ARE RANKED BY RELEVANCE:\n")
    for id, relatedness in zip(ids, relatednesses):
        print(f"ID: {id} has the following {relatedness=:.3f}")
    return response_message

print(ask('3 años de experiencia profesional. Skills: Python, SQL Server, PostgreSQL, MySQL, Tableau, Git, Github y PowerBI. Puesto deseado: Software Engineer, Data Scientist, Data Engineer, Data Analyst'))

#{"role": "system", "content": "You answer questions to users who want to find their dreamed job"},
#{"role": "system", "content": "You answer questions to users who want to find their dreamed job. Output 5 jobs only, include their ID and title"},
"""
Cuento con 3 años de experiencia profesional. Conocimiento en manipulación, análisis y visualización de datos mediante el uso de Python, SQL Server, PostgreSQL, MySQL, Tableau y PowerBI. En busca de oportunidades dentro de las áreas de Derecho de la Tecnología, Protección de Datos Personales, Ciencia de Datos, Ingeniería de Datos e Inteligencia Artificial.
"""