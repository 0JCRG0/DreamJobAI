import psycopg2, pandas as pd
from datetime import datetime, timedelta
import pretty_errors
from dotenv import load_dotenv
import os

load_dotenv('.env')
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")

def filter_last_two_weeks_postgre(table_name:str= "main_jobs") -> pd.DataFrame :
	conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)

	# Create a cursor object
	cur = conn.cursor()

	# Get the current date
	current_date = datetime.now().date()
	
	# Calculate the date two weeks ago from the current date
	two_weeks_ago = current_date - timedelta(days=14)
	
	# Fetch rows from the table with the specified conditions
	cur.execute(f"SELECT * FROM {table_name} WHERE timestamp >= %s", (two_weeks_ago,))

	# Fetch all rows from the table
	rows = cur.fetchall()

	# Close the database connection
	cur.close()
	conn.close()

	# Separate the columns into individual lists
	ids = [row[0] for row in rows]
	titles = [row[1] for row in rows]
	links = [row[2] for row in rows]
	descriptions = [row[3] for row in rows]
	pubdates = [row[4] for row in rows]
	locations = [row[5] for row in rows]
	timestamps = [row[6] for row in rows]

	df = pd.DataFrame({
		'id': ids,
		'title': titles,
		'link': links,
		'description': descriptions,
		'pubdate': pubdates,
		'location': locations,
		'timestamp': timestamps
	})

	# Sort the DataFrame by the 'timestamp' column
	df.sort_values(by='timestamp', inplace=True)

	# Reset the index if you want
	df.reset_index(drop=True, inplace=True)

	return df

df = filter_last_two_weeks_postgre()

print(f"df.info(): {df.info()}.\ndf.head(): {df.head()}\ndf.tail():{df.tail()}")


