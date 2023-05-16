import openai
import pinecone
import time

from openai.embeddings_utils import get_embedding
from config import *

openai.api_key = openai_apikey
pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)
index = pinecone.Index("office")

while True:
    my_inp = input("Enter Information: ")
    if len(my_inp)== 0:
        break

    my_inp_embed = get_embedding(my_inp, engine="text-embedding-ada-002")

    index.upsert(
        vectors =[{
            "id": str(time.time()),
            "values": my_inp_embed,
            "metadata": {"text": my_inp}
        }]
    )

