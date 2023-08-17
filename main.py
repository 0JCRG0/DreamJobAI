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
from utils.handy import e5_base_v2_query, filter_last_two_weeks, append_parquet, num_tokens, set_dataframe_display_options, filter_df_per_country
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

logging.basicConfig(filename='/Users/juanreyesgarcia/Library/CloudStorage/OneDrive-FundacionUniversidaddelasAmericasPuebla/DEVELOPER/PROJECTS/DreamedJobAI/logs/LoggingGPT4.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main(user_id:str, user_country:str):

    df_unfiltered = pd.read_parquet(E5_BASE_V2_DATA)

    df_two_weeks = filter_last_two_weeks(df_unfiltered)

    df = filter_df_per_country(df=df_two_weeks, user_desired_country=user_country)

    def ids_ranked_by_relatedness_e5(query: str,
        df: pd.DataFrame,
        min_n: int,
        top_n: int,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    ) -> tuple[list[str], list[float]]:
        
        #the query is embedded using e5
        query_embedding = e5_base_v2_query(query=query)

        ids_and_relatednesses = [
            (row["id"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df.iterrows()
        ]
        ids_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        ids, relatednesses = zip(*ids_and_relatednesses)
        return ids[min_n:top_n], relatednesses[min_n:top_n]     
        #Returns a list of strings and relatednesses, sorted from most related to least.

    async def async_query_summary(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int,
        min_n: int,
        top_n: int
    ) -> str:
        #Return a message for GPT, with relevant source texts pulled from a dataframe.
        ids, relatednesses = ids_ranked_by_relatedness_e5(query, df, min_n=min_n, top_n=top_n)
        #Basically giving the most relevant IDs from the previous function
        introduction = introduction_prompt
        query_user = f"{query}"
        message = introduction
        # Create a list of tasks
        tasks = [async_summarise_description(df[df['id'] == id]['description'].values[0]) for id in ids]

        # Run the tasks concurrently
        results = await asyncio.gather(*tasks)
        job_summaries = []
        total_cost_summaries = 0    

        for id, result in zip(ids, results):
            job_description_summary, cost, elapsed_time = result
            
            # Append summary to the list
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
        min_n: int,
        top_n: int,
        df: pd.DataFrame = df,
        model: str = GPT_MODEL,
        token_budget: int = 8192,
        log_gpt_messages: bool = True,
    ) -> str:
        #Answers a query using GPT and a dataframe of relevant texts and embeddings.
        query_user, job_id_description, job_summaries, total_cost_summaries = await async_query_summary(query, df, model=model, token_budget=token_budget, min_n=min_n, top_n=top_n)

        #Save summaries in a df & then parquet -> append data if function called more than once
        df_summaries = pd.DataFrame(job_summaries)
        append_parquet(df_summaries, 'summaries')
        
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
        logging.info(f"\nOPERATION: GPT-3.5 TURBO SUMMARISING. \nTOTAL COST: ${total_cost_summaries} USD")

        #Approximate cost
        if GPT_MODEL == "gpt-4":
            prompt_cost = round((prompt_tokens / 1000) * 0.03, 3)
            completion_cost = round((completion_tokens / 1000) * 0.06, 3)
            cost_classify = prompt_cost + completion_cost
            logging.info(f"\nOPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\nCOMPLETION TOKENS USED:{completion_tokens}\nTOTAL TOKENS USED:{total_tokens}\nCOST FOR CLASSIFYING: ${cost_classify} USD")
        elif GPT_MODEL == "gpt-3.5-turbo":
            prompt_cost = round((prompt_tokens / 1000) * 0.0015, 3)
            completion_cost = round((completion_tokens / 1000) * 0.002, 3)
            cost_classify = prompt_cost + completion_cost
            logging.info(f"\nOPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\nCOMPLETION TOKENS USED:{completion_tokens}\nTOTAL TOKENS USED:{total_tokens}\nCOST FOR CLASSIFYING: ${cost_classify} USD")
        elif GPT_MODEL == "gpt-3.5-turbo-16k":
            prompt_cost = round((prompt_tokens / 1000) * 0.003, 3)
            completion_cost = round((completion_tokens / 1000) * 0.004, 3)
            cost_classify = prompt_cost + completion_cost
            logging.info(f"\nOPERATION: {GPT_MODEL} CLASSIFICATION \nPROMPT TOKENS USED:{prompt_tokens}\nCOMPLETION TOKENS USED:{completion_tokens}\nTOTAL TOKENS USED:{total_tokens}\nCOST FOR CLASSIFYING: ${cost_classify} USD")

        #relatednesses
        ids, relatednesses = ids_ranked_by_relatedness_e5(query=query, df=df, min_n=min_n, top_n=top_n)
        for id, relatedness in zip(ids, relatednesses):
            logging.info(f"ID: {id} has the following {relatedness=:.3f}")
        
        elapsed_time = (timeit.default_timer() - start_time) / 60
        logging.info(f"\nGPT-3.5 TURBO & GPT-4 finished summarising and classifying! all in: {elapsed_time:.2f} minutes \n")
        
        return response_message

    async def check_output_GPT4(input_cv: str, min_n:int, top_n:int) -> str:
        default = '[{"id": "", "suitability": "", "explanation": ""}]'
        default_json = json.loads(default)
        
        for _ in range(6):
            i = _ + 1
            try:
                python_string = await ask(query=input_cv, min_n=min_n, top_n=top_n)
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

    #Modify df options - useful for logging
    set_dataframe_display_options()

    #Define the rows to classify
    min_n=0
    top_n=10

    # Define the suitable categories
    suitable_categories = ['Highly Suitable', 'Moderately Suitable', 'Potentially Suitable']

    # Initialize the dataframe
    df_appended = pd.DataFrame()

    # Continue to call the function until we have 10 suitable jobs
    counter = 0
    while True:
        checked_json = await check_output_GPT4(input_cv=abstract_cv, min_n=min_n, top_n=top_n)
        
        # Convert the JSON to a dataframe and append it to the existing dataframe
        df_original = pd.read_json(json.dumps(checked_json))
        df_appended = pd.concat([df_appended, df_original], ignore_index=True)
        
        counter += 1
        logging.info(f"Looking for suitable jobs. Current loop: {counter}")

        logging.info(f"Current min_n: {min_n}. Current top_n: {top_n}")

        # Increment the counters
        min_n += 10
        top_n += 10

        # Filter the dataframe to only include the suitable jobs
        df_most_suitable = df_appended[df_appended['suitability'].isin(suitable_categories)] if 'suitability' in df_appended.columns else pd.DataFrame()
        
        df_appended.to_parquet(SAVE_PATH + "/df_appended.parquet", index=False)
        df_most_suitable.to_parquet(SAVE_PATH + "/df_most_suitable.parquet", index=False)

        # Break the loop if we have 10 suitable jobs
        if len(df_most_suitable) >= 10:
            break

    logging.info(f"\nDF APPENDED:\n{df_appended}")
    
    #Get the ids
    def ids_df_most_suitable(df: pd.DataFrame = df_most_suitable) -> str:
        ids = ""
        for _, row in df.iterrows():
            if "id" in row:
                if ids:
                    ids += ", "
                ids += f"'{row['id']}'"

        return f"({ids})"

    ids_most_suitable = ids_df_most_suitable()
    logging.info(f"Getting the ids from the json object: {type(ids_most_suitable)}, {ids_most_suitable}")

    def find_jobs_per_ids(ids:str, table: str = "main_jobs") -> pd.DataFrame:
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

    df_postgre = find_jobs_per_ids(ids=ids_most_suitable)

    #Read the parquet with ids & summaries
    df_summaries = pd.read_parquet(SAVE_PATH + "/summaries.parquet")
    #Merge it with the data in postgre
    df_postgre_summaries = df_postgre.merge(df_summaries, on='id', how='inner')
    #Merge with most suitable df so you have all the rows
    df = df_postgre_summaries.merge(df_most_suitable, on="id", how='inner')

    logging.info(f"\nALL ROWS:\n{df}")


    def sort_df_by_suitability(df: pd.DataFrame = df) -> pd.DataFrame:
        custom_order = {
            'Highly Suitable': 1,
            'Moderately Suitable': 2,
            'Potentially Suitable': 3
        }
        df['suitability_rank'] = df['suitability'].map(custom_order)
        sorted_df = df.sort_values(by='suitability_rank')
        sorted_df = sorted_df.drop(columns='suitability_rank')
        return sorted_df

    sorted_df = sort_df_by_suitability()

    filename = "/final_user_df"
    
    sorted_df.to_parquet(SAVE_PATH + f"{filename}.parquet", index=False)

    logging.info(f"\nSORTED DF:\n{sorted_df}.\n\nThis df has been saved in ...{filename}.parquet\n\n\n")


if __name__ == "__main__":
	asyncio.run(main("", "Mexico"))
