from EmbeddingsOpenAI import embeddings_openai
import pandas as pd
import pretty_errors

df = pd.read_csv("/Users/juanreyesgarcia/Library/CloudStorage/OneDrive-FundacionUniversidaddelasAmericasPuebla/DEVELOPER/PROJECTS/DreamedJobAI/data/raw_summarised_batches.csv")

raw_summarised_batches = df["summary"]
ids = df["id"]
raw_batches = df["original"]

embeddings_openai(batches_to_embed= raw_summarised_batches, batches_ids=ids, original_descriptions=raw_batches, db="parquet", filename="openai_embeddings_summary")