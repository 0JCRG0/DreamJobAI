import pretty_errors
import re
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer


# Load stopwords for English and Spanish
stopwords_eng = set(stopwords.words("english"))
stopwords_spa = set(stopwords.words("spanish"))

# Initialize lemmatizers
lemmatizer_eng = WordNetLemmatizer()
stemmer_spa = SnowballStemmer("spanish")

def individual_preprocess(text):
    # Detect language
    lang = detect(text)

    # Remove special characters, numbers, and extra spaces
    text = re.sub(r"[^a-zA-Z\s:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize/stem based on language
    if lang == "en":
        tokens = [lemmatizer_eng.lemmatize(token) for token in tokens if token not in stopwords_eng]
    elif lang == "es":
        tokens = [stemmer_spa.stem(token) for token in tokens if token not in stopwords_spa]

    return " ".join(tokens)

"""
EXAMPLE

# Example usage
#job_description = "IT Operations Engineer You will be responsible for overseeing and maintaining IT assets software and hardware and IT procedures at the company aligned to current business objectives… Remoto in Ciudad de México"

#job_description = jobs_to_batches(500)

"""

if __name__ == "main":
    individual_preprocess()