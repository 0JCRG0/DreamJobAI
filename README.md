Based on an already existing crawler which scrapes remote jobs in 50+ job boards. This repo contains the embedding process of these scraped jobs which are then feed into GPT-4 to match the ideal job with the user's dreamed role.

Overall description of the repo:

1. Either all the jobs (LoadAllJobs.py) or most recent (LoadRecentJobs [default is 3hrs]) are loaded into a pandas df for further processing (transformed into batches based on max number of tokens).

2. Two embedding models are available, either OpenAI or e5 family,