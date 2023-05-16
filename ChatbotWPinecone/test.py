import openai
import pinecone
import time
import PyPDF2

from openai.embeddings_utils import get_embedding
from config import *

openai.api_key = openai_apikey
pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)
index = pinecone.Index("office")

pdf_file_path = b"C:\Users\\acer\Downloads\subdomain_finder.pdf"

with open(pdf_file_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text = page.extract_text()
        if len(text.strip()) > 0:
            my_inp_embed = get_embedding(text, engine="text-embedding-ada-002")

            index.upsert(
                vectors=[{
                    "id": str(time.time()),
                    "values": my_inp_embed,
                    "metadata": {"text": text}
                }]
            )
