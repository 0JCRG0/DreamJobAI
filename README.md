## PostgreSummarise.py

### Fetching Data

It then defines a function named fetch_data_from_table(table_name: str) -> list to retrieve data from a specified database table. The function connects to the database, fetches rows that meet specific criteria, separates column values into individual lists, and returns these lists.

### Data Processing

The script fetches data using the fetch_data_from_table() function for a table named "no_usa" and separates the data into ids, titles, locations, and descriptions lists.
The function rows_to_nested_list(title_list: list, location_list: list, description_list: list) -> list is defined to format and clean the data into a nested list structure, where each sub-list contains job-related information.

### Generating Batches

The raw_descriptions_to_batches(max_tokens: int, embedding_model: str, print_messages: bool = True) -> list function creates batches of descriptions by combining the cleaned titles, locations, and descriptions into strings.
It checks if the total tokens of a batch are within the specified limit (max_tokens). If they are, the batch is added as-is; otherwise, the batch is truncated using the truncated_string() function.

The function calculates the total tokens, approximate cost, and other statistics related to the batches.

### Summarizing Descriptions

The async summarise_descriptions(descriptions: list) -> list function asynchronously summarizes descriptions using external services.

It processes each description using the async_summarise_job_gpt() function, considering word count and handling potential exceptions with retries.
The function returns summaries, costs, and elapsed time.

### Main Function

The main(embedding_model:str) function is the main entry point of the script.

It asynchronously summarizes the descriptions using the summarise_descriptions() function, then formats and saves the summarized descriptions.

Based on the specified embedding_model, it either generates OpenAI embeddings or uses the "e5" model for embedding.