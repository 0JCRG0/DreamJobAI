import os
import openai
import pretty_errors
from dotenv import load_dotenv

load_dotenv('.env')

openai.api_key = os.getenv("OPENAI_API_KEY")
response=openai.Image.create(
    prompt="people smoking vapes in a park",
    n=2,
    size="1024x1024"
)

# Print the image URLs
for i, data in enumerate(response['data']):
    print(f"Image {i + 1}: {data['url']}")