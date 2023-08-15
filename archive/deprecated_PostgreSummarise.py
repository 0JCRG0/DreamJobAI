import psycopg2
import os
from dotenv import load_dotenv
import pretty_errors
import openai
import logging
import chromadb
import timeit
import time
from datetime import datetime, date, timedelta
from aiohttp import ClientSession
import asyncio
import time
from openai.error import ServiceUnavailableError
import pandas as pd
from datetime import datetime, timedelta
from utils.handy import num_tokens, count_words, LoggingMain, truncated_string, save_df_to_csv, summary_specs_txt_file, original_specs_txt_file, clean_rows
from utils.AsyncSummariseJob import async_summarise_job_gpt
from EmbeddingsOpenAI import embeddings_openai
from EmbeddingsE5 import embedding_e5_base_v2

"""
Env variables
"""

load_dotenv('.env')
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")
SAVE_PATH = os.getenv("SAVE_PATH")
#Setting API key
openai.api_key = os.getenv("OPENAI_API_KEY")

"""
Model
"""

LoggingMain()

model = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use


#Uncomment after first call

with open(SAVE_PATH + '/max_id.txt', 'r') as f:
	max_id = int(f.read())


def postgre_to_df(table_name:str, max_id:int = 0) -> list :
	conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)

	# Create a cursor object
	cur = conn.cursor()

	# Fetch new data from the table where id is greater than max_id
	cur.execute(f"SELECT id, title, description, location, timestamp FROM {table_name} WHERE id > {max_id}")
	new_data = cur.fetchall()

	# If new_data is not empty, update max_id with the maximum id from new_data
	if new_data:
		max_id = max(row[0] for row in new_data)

	# Close the database connection
	conn.commit()
	cur.close()
	conn.close()
	
	# Separate the columns into individual lists
	ids = [row[0] for row in new_data]
	titles = [row[1] for row in new_data]
	descriptions = [row[2] for row in new_data]
	locations = [row[3] for row in new_data]
	timestamp = [row[4] for row in new_data]

	return ids, titles, locations, descriptions, timestamp, max_id


##Comment after first call
#max_id = 0

ids, titles, locations, descriptions, timestamps, max_id = postgre_to_df("test", max_id)
print(max_id, len(ids))


def rows_to_nested_list(title_list: list, location_list: list, description_list: list) -> list:
	
	#Titles
	formatted_titles = ["####title: {}####".format(title) for title in title_list]
	cleaned_titles = [clean_rows(title) for title in formatted_titles]
	#Locations
	formatted_locations = ["####location: {}####".format(location) for location in location_list]
	cleaned_locations = [clean_rows(location) for location in formatted_locations]
	#Descriptions
	formatted_descriptions = ["####description: {}####".format(description) for description in description_list]
	cleaned_descriptions = [clean_rows(description) for description in formatted_descriptions]

	#NEST THE LISTS
	jobs_info = [[title, location, description] for title, location, description in zip(cleaned_titles, cleaned_locations, cleaned_descriptions)]

	return jobs_info

jobs_info= rows_to_nested_list(titles, locations, descriptions)


def raw_descriptions_to_batches(max_tokens: int, embedding_model: str, print_messages: bool = True) -> list:
	batches = []
	total_tokens = 0
	truncation_counter = 0  # Counter for truncations

	for i in jobs_info:
		text = " ".join(i)  # Join the elements of the list into a single string
		tokens_description = num_tokens(text)
		if tokens_description <= max_tokens:
			batches.append(text)
		else:
			#TRUNCATE IF STRING MORE THAN x TOKENS
			job_truncated = truncated_string(text, model=model, max_tokens=max_tokens)
			batches.append(job_truncated)
			truncation_counter += 1

		total_tokens += num_tokens(text)  # Update the total tokens by adding the tokens of the current job

	#Get approximate cost for embeddings
	if embedding_model == "openai":
		approximate_cost = round((total_tokens / 1000) * 0.0004, 4)
	elif embedding_model == "e5":
		approximate_cost = 0

	average_tokens_per_batch = total_tokens / len(batches)
	content = f"TOTAL NUMBER OF BATCHES: {len(batches)}\n" \
			f"TOTAL NUMBER OF TOKENS: {total_tokens}\n" \
			f"NUMBER OF TRUNCATIONS: {truncation_counter}\n" \
			f"AVERAGE NUMBER OF TOKENS PER BATCH: {average_tokens_per_batch}\n" \
			f"APPROXIMATE COST OF EMBEDDING: ${approximate_cost} USD\n"
	
	#TODO: This should be log not txt file
	original_specs_txt_file(content)

	if print_messages:
		for i, batch in enumerate(batches, start=1):
			print(f"Batch {i}:")
			print("".join(batch))
			print(f"Tokens per batch:", num_tokens(batch))
			print("\n")

		print(content)
	
	return batches


raw_batches = raw_descriptions_to_batches(max_tokens=1000, embedding_model="e5", print_messages = True)

df_raw_batches = pd.DataFrame({
	"ids": ids,
	"descriptions":raw_batches})

df_raw_batches.to_csv(SAVE_PATH + "/raw_batches.csv", index=False)


async def summarise_descriptions(descriptions: list) -> list:
	#start timer
	start_time = asyncio.get_event_loop().time()
	total_cost = 0

	async def process_description(session, i, text):
		attempts = 0
		while attempts < 5:
			try:
				words_per_text = count_words(text)
				if words_per_text > 50:
					description_summary, cost = await async_summarise_job_gpt(session, text)
					print(f"Description with index {i} just added.")
					logging.info(f"Description's index {i} just added.")
					return i, description_summary, cost
				else:
					logging.warning(f"Description with index {i} is too short for being summarised. Number of words: {words_per_text}")
					print(f"Description with index {i} is too short for being summarised. Number of words: {words_per_text}")
					return i, text, 0
			except (Exception, ServiceUnavailableError) as e:
				attempts += 1
				print(f"{e}. Retrying attempt {attempts}...")
				logging.warning(f"{e}. Retrying attempt {attempts}...")
				await asyncio.sleep(5**attempts)  # exponential backoff
		else:
			print(f"Description with index {i} could not be summarised after 5 attempts.")
			return i, text, 0

	async with ClientSession() as session:
		tasks = [process_description(session, i, text) for i, text in enumerate(descriptions)]
		results = await asyncio.gather(*tasks)

	# Sort the results by the index and extract the summaries and costs
	results.sort()
	descriptions_summarised = [result[1] for result in results]
	costs = [result[2] for result in results]
	total_cost = sum(costs)

	#await close_session()
	#processed_time = timeit.default_timer() - start_time
	elapsed_time = asyncio.get_event_loop().time() - start_time

	return descriptions_summarised, total_cost, elapsed_time


async def main(embedding_model:str):
	raw_summarised_batches, raw_total_cost, raw_processed_time = await summarise_descriptions(raw_batches)

	def passage_e5_format(summaries):
		formatted_summary = ["passage: {}".format(summary) for summary in summaries]
		return formatted_summary
	
	#SAVE THE DATA...

	formatted_summarised_e5_batches = passage_e5_format(raw_summarised_batches)

	save_df_to_csv(ids, raw_batches, formatted_summarised_e5_batches)	

	summary_specs_txt_file(raw_total_cost, raw_processed_time)

	#Embedding starts
	if embedding_model == "openai":
		embeddings_openai(batches_to_embed= formatted_summarised_e5_batches, batches_ids=ids, original_timestamps=timestamps, original_descriptions=raw_batches, db="parquet", filename="openai_embeddings_summary")
	elif embedding_model == "e5":
		embedding_e5_base_v2(batches_to_embed = formatted_summarised_e5_batches, batches_ids=ids, original_timestamps=timestamps, original_descriptions=raw_batches, chunk_size=15)

#At the end of the script, save max_id to the file
with open(SAVE_PATH + '/max_id.txt', 'w') as f:
	f.write(str(max_id))

if __name__ == "__main__":
	asyncio.run(main("e5"))