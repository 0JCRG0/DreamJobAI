import boto3
import botocore
import pretty_errors
import os
from dotenv import load_dotenv

load_dotenv(".env")

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION")
E5_BASE_V2_DATA = os.environ.get("E5_BASE_V2_DATA")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
FILE_NAME = "E5_BASE_V2_DATA.parquet"



def create_default_s3_client() -> object:
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )

    return s3

s3 = create_default_s3_client()

#Upload the file
#s3.upload_file(E5_BASE_V2_DATA, BUCKET_NAME, FILE_NAME)

#Get the file

try:
    # Read the file from S3
    response = s3.get_object(Bucket=BUCKET_NAME, Key=FILE_NAME)
    print(response)
    content = response['Body'].read()

    # Now, 'content' contains the binary data of the file.
    # You can process it as needed.

except Exception as e:
    if e.response['Error']['Code'] == "NoSuchKey":
        print("The object does not exist.")
    else:
        raise