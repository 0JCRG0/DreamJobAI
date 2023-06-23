import torch.nn.functional as F

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pretty_errors
import pandas as pd
import numpy as np
from utils.handy import *
import pretty_errors
from dotenv import load_dotenv
import os
from postgre2embedding import jobs_to_batches, jobs_ids, jobs_for_GPT

load_dotenv('.env')
SAVE_PATH = os.getenv("SAVE_PATH")

def embedding_e5_family(embedding_tokenizer: str, embedding_model: str):
    INPUT_TEXT = jobs_to_batches(max_tokens=512, embedding_model="e5-large", print_warning=False)
    TOKENIZER = AutoTokenizer.from_pretrained(embedding_tokenizer)
    MODEL = AutoModel.from_pretrained(embedding_model)

    CHUNK_SIZE = 64

    def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def save_embeddings_to_parquet(embeddings, parquet_filename):
        df_data = {
        'ids': jobs_ids,
        'text_data': jobs_for_GPT(700, "e5-large", False),
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
    all_embeddings = np.vstack(embeddings_list)
    save_embeddings_to_parquet(all_embeddings, f'{embedding_model}_embeddings.parquet')

if __name__ == "__main__":
    embedding_e5_family("intfloat/e5-large-v2", "intfloat/e5-large-v2")

    """
    tokeinizer= intfloat/e5-large-v2
    model = intfloat/e5-large-v2

    tokenizer= intfloat/e5-base-v2
    model= intfloat/e5-base-v2

    """