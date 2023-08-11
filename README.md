# PostgreSummarise.py

This script is designed to fetch job data from a PostgreSQL database, process the data, and then generate summaries of the job descriptions using an AI model. The script also calculates the cost of generating these summaries and saves the results to a CSV file.

## Functions

### fetch_data_from_table(table_name:str) -> list

This function connects to a PostgreSQL database and fetches job data from a specified table. The data includes job ids, titles, locations, and descriptions. The function returns these data as separate lists.

### rows_to_nested_list(title_list: list, location_list: list, description_list: list) -> list

This function takes lists of job titles, locations, and descriptions, formats them, and then nests them into a single list.

### raw_descriptions_to_batches(max_tokens: int, embedding_model: str, print_messages: bool = True) -> list

This function takes the nested list of job data and splits it into batches. Each batch contains a maximum number of tokens specified by the max_tokens parameter. The function also calculates the cost of generating summaries for these batches.

### summarise_descriptions(descriptions: list) -> list

This function takes a list of job descriptions and generates summaries for them using an gpt-3.5 model. The function returns a list of summaries, the total cost of generating the summaries, and the time taken to generate the summaries.

### main(embedding_model:str)

This function is the main entry point of the script. It calls the other functions to fetch and process the job data, generate summaries, and save the results to a CSV file. The embedding_model parameter specifies the AI model to use for generating the summaries.