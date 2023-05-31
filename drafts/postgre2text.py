import psycopg2

def fetch_data_from_table():
    conn = psycopg2.connect(user='postgres', password='3312', host='localhost', port='5432', database='postgres')

    # Create a cursor object
    cur = conn.cursor()

    # Replace "your_table_name" with the name of the table you want to fetch
    cur.execute("SELECT id, title, description, location FROM personal")

    # Fetch all rows from the table
    rows = cur.fetchall()

    # Close the database connection
    cur.close()
    conn.close()

    return rows


def table_to_text(rows):
    text = ""
    for row in rows:
        # Convert each row to a string and join the elements with a separator (e.g., tab, comma)
        row_text = "\t".join(str(x) for x in row)

        # Add the row to the text, followed by a newline character
        text += row_text + "\n"

    return text

#To save it as a .txt
def save_text_to_file(text, file_name):
    with open(file_name, 'w', encoding='utf8') as file:
        file.write(text)


if __name__ == "__main__":
    # Fetch data from the PostgreSQL table
    table_data = fetch_data_from_table()

    # Convert the table data to text
    text = table_to_text(table_data)

    # Print the text
    save_text_to_file(text, 'personal_partial.txt')
