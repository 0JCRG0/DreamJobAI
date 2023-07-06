import os
import openai
from dotenv import load_dotenv
import pandas as pd
import pretty_errors

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL= "gpt-3.5-turbo-16k"

delimiters = "####"

system_query = f""" 

Your task is to extract "relevant information" of a job opening/
posted by a company, with the aim of effectively matching /
potential candidates for the position./

From the job opening below, delimited by {delimiters} characters, extract 
"relevant information" in at most 200 words. /
Your main focus is the category of Job Information./

Relevant information to extract: 

Job Information (150 words max):  extract any essential details about the specific position being advertised. Focus on job title, /
job objective, responsibilities/key duties, qualifications/requirements and preferred skills/experience. /

About the company and Compensation and Benefits (50 words max): extract any information about the hiring company,/
such as company name, location, culture, values, or mission. /
Also, extract any financial aspects and additional perks associated with the job./
/ 

"""

assistant_query=""" Remember to think step by step.
"""

def summarise_job_gpt(job_description):
    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'user', 'content': system_query},
            {'role': 'user', 'content': f"Job Opening: {delimiters}{job_description}{delimiters}"},
        ],
        model=MODEL,
        temperature=0,
        max_tokens=4000
    )
    response_message = response['choices'][0]['message']['content']
    total_cost = 0
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    #print(f"\nSUMMARISE JOBS FUNCTION\n", f"\nPROMPT TOKENS USED:{prompt_tokens}\n", f"COMPLETION TOKENS USED:{completion_tokens}\n" )
    #Approximate cost
    if MODEL == "gpt-3.5-turbo":
        prompt_cost = round((prompt_tokens / 1000) * 0.0015, 3)
        completion_cost = round((completion_tokens / 1000) * 0.002, 3)
        total_cost = prompt_cost + completion_cost
        #print(f"COST FOR SUMMARISING: ${total_cost} USD")
    elif MODEL == "gpt-3.5-turbo-16k":
        prompt_cost = round((prompt_tokens / 1000) * 0.003, 3)
        completion_cost = round((completion_tokens / 1000) * 0.004, 3)
        total_cost = prompt_cost + completion_cost
        #print(f"COST FOR SUMMARISING: ${total_cost} USD")
    return response_message, total_cost

if __name__ == "__main__":
    x = "Hey there Were AKKO  a venture funded  fastgrowing insurtech company with a passionate solutionforward and collaborative team focused on disrupting the device protection space Our mission is simple  Leveraging technology to make protecting peoples tech easier and more affordable Most peoples lives have become techcentric and AKKO provides customers with peace of mind that their tech is safe guarded in a transparent and seamless way Backed by amazing investors such as Mundi Fika and Pear we are just getting started THE DAYTODAY Develop and manage data systems and databases Extract data from primary and secondary sources and removing corrupted data Establish KPIs that provide actionable insights Use data to analyze trends to help inform business policies and decisions Collaborate with engineers and developers to develop and streamline data governance strategies Ensure data is accurate and highquality Analyze data for trends and patterns and interpret data with a clear objective in mind Communicate analytic solutions to stakeholders and implement improvements and solutions WHAT MAKES YOU QUALIFIED Minimum one year experience working as a data analyst or similar role preferably for a startup with a bachelors degree in Statistics Computer Science Mathematics or equivalent practical experience Knowledge of relational and nonrelational NoSQL databases Python NumPy SciPy or R for statistical analysis Data visualization Looker Studio Power BI Tableau Pythons Matplotlib and Seaborn or similar Proficiency in Excel Actuarial knowledge or experience is a plus All around team player fast and selflearning individual Proactive and solutionforward bias for action Ability to drive forward work independently while communicating and working across functions Passion for data Experience working on highscale productiongrade projects WHY YOULL LOVE IT HERE Unlimited vacation and paid sick time Competitive health benefits including medical dental and vision insurance Robust 401k program  to invest in your future Monthly wellness stipend eg gym yoga meditation etc  we value your wellbeing Monthly treat yourself stipend  dinner on us Remote workspace stipend  Work from home or from a shared workspace  you decide Paid volunteer time  giving back to our community is important to us Annual learning credit explore personal interests that excite you and so much more The base salary for this position ranges from 65000 to 100000 Compensation varies based on jobrelated factors including business needs experience level of responsibility and qualifications WHAT ELSE ARE WE LOOKING FOR Our team is fostered around our core values Collaborate Work together to be more effective lift up others and win together Aim High Set ambitious goals Embrace Diversity Seek different perspectives bring our true self to work Customer Love Serve the end user and listen to them Nurture Empathy Listen and strive to truly understand others Take Action Be proactive be an owner value speed Maintain Integrity Build the AKKO you are proud to work at Data Driven Use data to iterate find truth CCPA disclosure notice at getakkocomlegal"
    summarise_job_gpt(job_description=x)
