delimiters = "####"

system_prompt=f"""

You are a job recruiter for a large recruitment agency./
You will be provided with a candidate's CV./
The CV will be delimited with {delimiters} characters./
You will also be provided with the Job IDs (delimited by angle brackets) /
and corresponding descriptions (delimited by triple dashes)/
for the available job openings./

Perform the following steps:/

Step 1 - Classify the provided CV into a suitability category for each job opening./
Step 2 - For each ID briefly explain in one sentence your reasoning behind the chosen suitability category./
Step 3 - Only provide your output in json format with the keys: id, suitability and explanation./

Do not classify a CV into a suitability category until you have classify the CV yourself.

Suitability categories: Highly Suitable, Moderately Suitable, Potentially Suitable, Marginally Suitable and Not Suitable./

Highly Suitable: CVs in this category closely align with the job opening, demonstrating extensive relevant experience, skills, and qualifications. The candidate possesses all or most of the necessary requirements and is an excellent fit for the role./
Moderately Suitable: CVs falling into this category show a reasonable match to the job opening. The candidate possesses some relevant experience, skills, and qualifications that align with the role, but there may be minor gaps or areas for improvement. With some additional training or development, they could become an effective candidate./
Potentially Suitable: CVs in this category exhibit potential and may possess transferable skills or experience that could be valuable for the job opening. Although they may not meet all the specific requirements, their overall profile suggests that they could excel with the right support and training./
Marginally Suitable: CVs falling into this category show limited alignment with the job opening. The candidate possesses a few relevant skills or experience, but there are significant gaps or deficiencies in their qualifications. They may require substantial training or experience to meet the requirements of the role./
Not Suitable: CVs in this category do not match the requirements and qualifications of the job opening. The candidate lacks the necessary skills, experience, or qualifications, making them unsuitable for the role./
"""

introduction_prompt = """


\n Available job openings:\n

"""


abstract_cv_past = """Data Analyst: Cleansed, analyzed, and visualized data using Python, SQL Server, and Power BI.
Legal Assistant: Drafted legal documents, collaborated on negotiation outlines, and handled trademark registrations.
Data Analyst Jr.: Implemented A/B testing, utilized data analysis tools, and developed real-time visualizations.
Special Needs Counselor: Led and assisted individuals with disabilities, provided personal care, and facilitated camp activities.
Total years of professional experience: 3 years."""

abstract_cv = """('Qualifications: \n- LLB Law degree from Universidad de las Américas Puebla (UDLAP) with an accumulated average of 9.4/10.\n- Currently on an international exchange at the University of Bristol for the final year of studying Law.\n- Member of the Honours Program at UDLAP, conducting research on FinTech, Financial Inclusion, Blockchain, Cryptocurrencies, and Smart Contracts.\n\nPrevious job titles:\n- Data Analyst at Tata Consultancy Services México, where I cleansed, interpreted, and analyzed data using Python and SQL Server to produce visual reports with Power BI.\n- Legal Assistant at BLACKSHIIP Venture Capital, responsible for proofreading and drafting legal documents, as well as assisting with negotiations of International Share Purchase Agreements.\n\nResponsibilities/Key Duties:\n- Developed and introduced A/B testing to make data-driven business decisions as a Data Analyst Jr. at AMATL GRÁFICOS.\n- Taught mental arithmetic as a Mathematics Instructor at ALOHA Mental Arithmetic.\n- Led and assisted individuals with physical and mental disabilities as a Special Needs Counsellor at Camp Merrywood and YMCA Camp Independence.\n\nSkills:\n- Proficient in Python, SQL Server, Tableau, Power BI, Bash/Command Line, Git & GitHub, and Office 365.\n- Strong written and verbal communication skills, teamwork, ability to work under pressure, attention to detail, and leadership skills.\n- Knowledge in machine learning, probabilities & statistics, and proofreading.\n\nOther Achievements:\n- Published paper on "Smart Legal Contracts: From Theory to Reality" and participated in the IDEAS Summer Program on Intelligence, Data, Ethics, and Society at the University of California, San Diego."""