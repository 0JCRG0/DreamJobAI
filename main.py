import os
import openai
import psycopg2
import pandas as pd
import tiktoken  # for counting tokens
from scipy import spatial
import pretty_errors
import timeit
import logging
import time
import asyncio
from openai.error import OpenAIError
import json
from typing import Callable
from utils.preprocess import individual_preprocess
from dotenv import load_dotenv
from utils.prompts import *
from utils.SummariseJob import summarise_job_gpt
from utils.AsyncSummariseJob import async_summarise_description
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from utils.handy import e5_base_v2_query, LoggingGPT4, filter_last_two_weeks, df_to_parquet
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")
SAVE_PATH = os.getenv("SAVE_PATH")
E5_BASE_V2_DATA = os.getenv("E5_BASE_V2_DATA")


#Start the timer
start_time = timeit.default_timer()

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
#GPT_MODEL = "gpt-3.5-turbo"
GPT_MODEL = "gpt-4"
#GPT_MODEL = "gpt-3.5-turbo-16k"
""""
Load the embedded file
"""

logging.basicConfig(filename='/Users/juanreyesgarcia/Library/CloudStorage/OneDrive-FundacionUniversidaddelasAmericasPuebla/DEVELOPER/PROJECTS/DreamedJobAI/logs/LoggingGPT4.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


embeddings_path = E5_BASE_V2_DATA

df_unfiltered = pd.read_parquet(embeddings_path)

df = filter_last_two_weeks(df_unfiltered)


def ids_ranked_by_relatedness_e5(query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    #Modify this to get more jobs
    top_n: int = 10
) -> tuple[list[str], list[float]]:
    
    #the query is embedded using e5
    query_embedding = e5_base_v2_query(query=query)

    ids_and_relatednesses = [
        (row["id"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    ids_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    ids, relatednesses = zip(*ids_and_relatednesses)
    return ids[:top_n], relatednesses[:top_n]     
    #Returns a list of strings and relatednesses, sorted from most related to least.

#tiktoken function -> to count tokens
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

async def async_query_summary(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    #Return a message for GPT, with relevant source texts pulled from a dataframe.
    ids, relatednesses = ids_ranked_by_relatedness_e5(query, df)
    #Basically giving the most relevant IDs from the previous function
    introduction = introduction_prompt
    query_user = f"{query}"
    message = introduction
    # Create a list of tasks
    tasks = [async_summarise_description(df[df['id'] == id]['original'].values[0]) for id in ids]

    # Run the tasks concurrently
    results = await asyncio.gather(*tasks)
    job_summaries = []
    total_cost_summaries = 0    

    for id, result in zip(ids, results):
        job_description_summary, cost, elapsed_time = result
        
        # Append the summary to the list
        job_summaries.append({
            "id": id,
            "summary": job_description_summary
        })
        #Append total cost
        total_cost_summaries += cost

        next_id = f'\nID:<{id}>\nJob Description:---{job_description_summary}---\n'
        if (
            num_tokens(message + next_id + query_user, model=model)
            > token_budget
        ):
            break
        else:
            message += next_id
    return query_user, message, job_summaries, total_cost_summaries

#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def ask(
    #This query is your question, only parameter to fill in function
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 8192,
    log_gpt_messages: bool = True
) -> str:
    #Answers a query using GPT and a dataframe of relevant texts and embeddings.
    query_user, job_id_description, job_summaries, total_cost_summaries = await async_query_summary(query, df, model=model, token_budget=token_budget)

    #Save summaries in a df & then parquet
    df_summaries = pd.DataFrame(job_summaries)
    df_to_parquet(df_summaries, "summaries")

    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{delimiters}{query_user}{delimiters}"},
        {"role": "assistant", "content": job_id_description}
    ]
    if log_gpt_messages:
        logging.info(messages)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    
    #if print_cost_and_relatednesses:
    total_tokens = response['usage']['total_tokens']
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    logging.info(f"OPERATION: GPT-3.5 TURBO SUMMARISING. \n TOTAL COST: {total_cost_summaries}")
    #logging.info(f"OPERATION: {GPT_MODEL} CLASSIFYING \nPROMPT TOKENS USED:{prompt_tokens}\n COMPLETION TOKENS USED:{completion_tokens}\n \nTOTAL TOKENS USED:{total_tokens}\n)

    #Approximate cost
    if GPT_MODEL == "gpt-4":
        prompt_cost = round((prompt_tokens / 1000) * 0.03, 3)
        completion_cost = round((completion_tokens / 1000) * 0.06, 3)
        cost_classify = prompt_cost + completion_cost
        logging.info(f"OPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\n COMPLETION TOKENS USED:{completion_tokens}\n \nTOTAL TOKENS USED:{total_tokens}\n COST FOR CLASSIFYING: ${cost_classify} USD")
    elif GPT_MODEL == "gpt-3.5-turbo":
        prompt_cost = round((prompt_tokens / 1000) * 0.0015, 3)
        completion_cost = round((completion_tokens / 1000) * 0.002, 3)
        cost_classify = prompt_cost + completion_cost
        logging.info(f"OPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\n COMPLETION TOKENS USED:{completion_tokens}\n \nTOTAL TOKENS USED:{total_tokens}\n COST FOR CLASSIFYING: ${cost_classify} USD")
    elif GPT_MODEL == "gpt-3.5-turbo-16k":
        prompt_cost = round((prompt_tokens / 1000) * 0.003, 3)
        completion_cost = round((completion_tokens / 1000) * 0.004, 3)
        cost_classify = prompt_cost + completion_cost
        logging.info(f"OPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\n COMPLETION TOKENS USED:{completion_tokens}\n \nTOTAL TOKENS USED:{total_tokens}\n COST FOR CLASSIFYING: ${cost_classify} USD")

    #relatednesses
    ids, relatednesses = ids_ranked_by_relatedness_e5(query=query, df=df)
    for id, relatedness in zip(ids, relatednesses):
        logging.info(f"ID: {id} has the following {relatedness=:.3f}")
    
    elapsed_time = timeit.default_timer() - start_time
    logging.info(f"\n DreamedJobAI finished! all in: {elapsed_time:.2f} seconds \n")
    
    return response_message

async def check_output_GPT4(input_cv: str) -> str:
    default = '[{"id": "", "suitability": "", "explanation": ""}]'
    default_json = json.loads(default)
    
    for _ in range(6):
        i = _ + 1
        try:
            python_string = await ask(input_cv)
            try:
                data = json.loads(python_string)
                logging.info(f"Response is a valid json object. Done in loop number: {i}")
                return data
            except json.JSONDecodeError:
                pass
        except OpenAIError as e:
            logging.warning(f"{e}. Retrying in 10 seconds. Number of retries: {i}")
            time.sleep(10)
            pass
        except Exception as e:
            logging.warning(f"{e}. Retrying in 5 seconds. Number of retries: {i}")
            time.sleep(5)
            pass

    logging.error("Check logs!!!! Main function was not callable. Setting json to default")
    return default_json


async def main():


    checked_json = await check_output_GPT4(abstract_cv)

    def ids_json_loads(data: list[dict[str, str, str]] = None) -> str:
        if data is None:
            data = checked_json
            logging.info(f"type of the json object: {type(data)} Data: {data}")
            #print(type(exp), exp)
        
        ids = ""
        for item in data:
            if "id" in item:
                if ids:
                    ids += ", "
                ids += f"'{item['id']}'"

        return f"({ids})"

    ids_ready = ids_json_loads()
    logging.info(f"Getting the ids from the json object: {type(ids_ready)}, {ids_ready}")


    def find_jobs_per_ids(ids:str, table: str = "test") -> pd.DataFrame:
        conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)
        # Create a cursor object
        cur = conn.cursor()
        #TABLE SHOULD EITHER BE "main_jobs" or "test"
        cur.execute( f"SELECT id, title, link, location FROM {table} WHERE id IN {ids}")

        # Fetch all rows from the table
        rows = cur.fetchall()

        # Separate the columns into individual lists
        all_ids = [row[0] for row in rows]
        all_titles = [row[1] for row in rows]
        all_links = [row[2] for row in rows]
        all_locations = [row[3] for row in rows]

        df = pd.DataFrame({
            'id': all_ids,
            'title': all_titles,
            'link': all_links,
            'location': all_locations
        })


        # Close the database connection
        cur.close()
        conn.close()

        return df

    df_postgre = find_jobs_per_ids(ids=ids_ready)
    #Read the parquet
    df_summaries = pd.read_parquet(SAVE_PATH + "/summaries.parquet")

    df = df_postgre.merge(df_summaries, on='id', how='inner')

    logging.info(f"RELATED JOBS & THEIR SUMMARIES: \n {df}")

    def adding_all_data(df: pd.DataFrame, suitable_jobs: list) -> pd.DataFrame:
        for index, row in df.iterrows():
            entry_id = row['id']
            for json_item in suitable_jobs:
                if int(json_item['id']) == entry_id:
                    suitability = json_item['suitability']
                    explanation = json_item['explanation']
                    df.at[index, 'suitability'] = suitability
                    df.at[index, 'explanation'] = explanation
                    break
        return df

    updated_data = adding_all_data(df=df, suitable_jobs=checked_json)

    logging.info(f"ALL COLUMNS: \n {updated_data}")

    def sort_df_by_suitability(df: pd.DataFrame = df) -> pd.DataFrame:
        custom_order = {
            'Highly Suitable': 1,
            'Moderately Suitable': 2,
            'Potentially Suitable': 3,
            'Marginally Suitable': 4,
            'Not Suitable': 5
        }
        df['suitability_rank'] = df['suitability'].map(custom_order)
        sorted_df = df.sort_values(by='suitability_rank')
        sorted_df = sorted_df.drop(columns='suitability_rank')
        return sorted_df

    sorted_df = sort_df_by_suitability()

    filename = "/final_user_df"
    
    sorted_df.to_parquet(SAVE_PATH + f"{filename}.parquet", index=False)


    logging.info(f"SORTED DF:\n {sorted_df}. \n This df has been saved in ...{filename}.parquet")

if __name__ == "__main__":
	asyncio.run(main())
        