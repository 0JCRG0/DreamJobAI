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
from utils.handy import LoggingMain, append_parquet
import pretty_errors
from dotenv import load_dotenv
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


load_dotenv('.env')
SAVE_PATH = os.getenv("SAVE_PATH")
E5_BASE_PREEXISTING_JOBS = os.getenv("E5_BASE_PREEXISTING_JOBS")
E5_BASE_TODAY_JOBS = os.getenv("E5_BASE_RECENT_JOBS")
E5_BASE_TOTAL_JOBS = os.getenv("E5_BASE_TOTAL_JOBS")

LoggingMain()

#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def embedding_e5_base_v2(batches_to_embed: list[str], batches_ids: list[str], batches_locations: list[str], original_timestamps: list[str], chunk_size: int) -> list:
    
    INPUT_TEXT = batches_to_embed
    INPUT_IDS = batches_ids
    INPUT_LOCATIONS = batches_locations
    INPUT_TIMESTAMPS = original_timestamps
    TOKENIZER = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
    MODEL = AutoModel.from_pretrained("intfloat/e5-base-v2")
    CHUNK_SIZE = chunk_size

    def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

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
    EMBEDDINGS = np.vstack(embeddings_list)

    #Timestamps
    """
    TIMESTAMPS = []
    timestamp = datetime.now()
    TIMESTAMPS.extend([timestamp] * len(INPUT_IDS)) """

    df_data = {
        'id': INPUT_IDS,
        'description': INPUT_TEXT,
        'location': INPUT_LOCATIONS,
        'embedding': list(EMBEDDINGS),
        'timestamp': INPUT_TIMESTAMPS
        }

    new_data = pd.DataFrame(df_data)

    #new_data.to_parquet(SAVE_PATH + f'/e5_base_v2_data.parquet', engine='pyarrow')

    append_parquet(new_data, "e5_base_v2_data")



