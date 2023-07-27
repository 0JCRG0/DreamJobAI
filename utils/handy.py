import re
from chromadb.utils import embedding_functions
import tiktoken
import pandas as pd
import logging
import datetime
import pyarrow.parquet as pq
from dotenv import load_dotenv
import os

load_dotenv(".env")
LOGGER_PATH = os.getenv("LOGGER_PATH")
SAVE_PATH = os.getenv("SAVE_PATH")

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

def original_specs_txt_file(content: str): 
	timestamp = datetime.datetime.now()
	with open(SAVE_PATH + '/specs.txt', 'a') as file:
		file.write(f"At {timestamp}\n")
		file.write("RAW BATCHES SPECS: \n")
		file.write(content)

def summary_specs_txt_file(total_cost: float, processed_time: float): 
	with open(SAVE_PATH + '/specs.txt', 'a') as file:
		file.write("\nSUMMARISED BATCHES SPECS: \n")
		file.write(f"Total Cost: {total_cost:.2f}\n")
		file.write(f"Processed Time: {processed_time:.2f} seconds\n\n")

def save_df_to_csv(id, original, summary):
	df_raw_summarised_batches = pd.DataFrame({
		"id": id,
		"original": original,
		"summary": summary})

	df_raw_summarised_batches.to_csv(SAVE_PATH + "/raw_summarised_batches.csv", index=False)

def count_words(text: str) -> int:
	# Remove leading and trailing whitespaces
	text = text.strip()

	# Split the text into words using whitespace as a delimiter
	words = text.split()

	# Return the count of words
	return len(words)

def save_embeddings_to_parquet(data):
	df = pd.DataFrame(data)
	df.to_parquet(SAVE_PATH+ f"/e5_base_v2_data.parquet", engine='pyarrow')
	print(f"Saved embeddings to ../e5_base_v2_data.parquet")

def append_parquet(new_df: pd.DataFrame):
	# Load existing data
	df = pd.read_parquet('/Users/juanreyesgarcia/Library/CloudStorage/OneDrive-FundacionUniversidaddelasAmericasPuebla/DEVELOPER/PROJECTS/DreamedJobAI/data/e5_base_v2_data.parquet')

	df = pd.concat([df, new_df], ignore_index=True)
		# Remove duplicates based on 'id' column
	df = df.drop_duplicates(subset='id')

	# Write back to Parquet
	df.to_parquet('/Users/juanreyesgarcia/Library/CloudStorage/OneDrive-FundacionUniversidaddelasAmericasPuebla/DEVELOPER/PROJECTS/DreamedJobAI/data/e5_base_v2_data.parquet', engine='pyarrow')
	logging.info("e5_base_v2_data.parquet has been updated")