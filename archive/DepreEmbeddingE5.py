import torch.nn.functional as F

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pretty_errors
import pandas as pd
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from utils.handy import *
import pretty_errors
from dotenv import load_dotenv
import os

load_dotenv('.env')
SAVE_PATH = os.getenv("SAVE_PATH")
E5_BASE_PREEXISTING_JOBS = os.getenv("E5_BASE_PREEXISTING_JOBS")
E5_BASE_TODAY_JOBS = os.getenv("E5_BASE_RECENT_JOBS")
E5_BASE_TOTAL_JOBS = os.getenv("E5_BASE_TOTAL_JOBS")

LoggingMain()

#TODO: MODIFY SO IT CAN BE CALLED BY postgresummary

def embedding_e5_family(embedding_tokenizer: str, embedding_model: str, all_jobs: bool, chunk_size: int, print_warning: bool) -> list:
    if all_jobs:
        jobs_in_batches = all_jobs_to_batches(512, "e5", print_warning)
        jobs_text = all_jobs_for_GPT(512, "e5", print_warning)
        jobs_ids = all_jobs_ids
        file_name = "e5_base_preexisting.parquet"
    else:
        jobs_in_batches = recent_jobs_to_batches(512,"e5", print_warning)
        jobs_text = recent_jobs_for_GPT(512, "e5", print_warning)
        jobs_ids = recent_jobs_ids
        file_name = "e5_base_today.parquet"
    
    INPUT_TEXT = jobs_in_batches
    TOKENIZER = AutoTokenizer.from_pretrained(embedding_tokenizer)
    MODEL = AutoModel.from_pretrained(embedding_model)

    CHUNK_SIZE = chunk_size

    def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def save_embeddings_to_parquet(embeddings, parquet_filename):
        df_data = {
        'ids': jobs_ids,
        'text_data': jobs_text,
        'embeddings': list(embeddings)
    }
        df = pd.DataFrame(df_data)
        df.to_parquet(SAVE_PATH+ f"/{parquet_filename}", engine='pyarrow')
        print(f"Saved embeddings to {parquet_filename}")

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.texts = texts
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]

    def collate_fn(batch, tokenizer):
        batch_dict = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return batch_dict

    # Create a dataset and a dataloader
    dataset = TextDataset(INPUT_TEXT, TOKENIZER)
    dataloader = DataLoader(dataset, batch_size=CHUNK_SIZE, collate_fn=lambda b: collate_fn(b, TOKENIZER))

    # Process the data using the dataloader
    embeddings_list = []

    with torch.no_grad():
        for batch_dict in tqdm(dataloader, desc="Processing batches"):
            # Get embeddings
            outputs = MODEL(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).detach().numpy()

            # Add batch embeddings to the list
            embeddings_list.append(batch_embeddings)

    # Concatenate embeddings and save to a single Parquet file
    final_embeddings = np.vstack(embeddings_list)
    save_embeddings_to_parquet(final_embeddings, file_name)

    #If you are embedding jobs from today concat it with all_jobs
    if all_jobs == False:
        def concat_parquet_E5():
            if embedding_model == "intfloat/e5-base-v2":
                today = E5_BASE_TODAY_JOBS
                preexisting = E5_BASE_PREEXISTING_JOBS
                total = E5_BASE_TOTAL_JOBS
            else:
                print("NOT DEFINED")
                logging.error("NOT DEFINED")
            # Read the Parquet files
            today_jobs = pq.read_table(today)
            preexisting_jobs = pq.read_table(preexisting)
            
            # Append the tables
            appended_jobs = pa.concat_tables([today_jobs, preexisting_jobs])
            
            # Write the appended table to a new Parquet file
            pq.write_table(appended_jobs, total)
            now = datetime.now()
            logging.info(f"{embedding_model} embeddings have been appended at {now}")
        concat_parquet_E5()

if __name__ == "__main__":
    embedding_e5_family(embedding_tokenizer="intfloat/e5-base-v2", embedding_model="intfloat/e5-base-v2", all_jobs=False, chunk_size=15, print_warning=False)

    """
    tokeinizer= intfloat/e5-large-v2
    model = intfloat/e5-large-v2

    tokenizer= intfloat/e5-base-v2
    model= intfloat/e5-base-v2

    """