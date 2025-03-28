import os
import pinecone
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


pc = Pinecone(api_key="pcsk_6WTbNw_Tj9nVaxsJuyruu4WbfKBCsCHAqvhiYHgAHrkwAEkop5dM8GDkgzWhd6GJomRx2z")

# Index details
index_name = "financial-reports"

# ✅ Ensure the index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=len(df["embedding"][0]),
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(index_name)

# ✅ Prepare data for insertion
vectors = [
    (str(i), df["embedding"][i],
     {"jargon": df["jargon"][i], "actual_def": df["actual_definition"][i], "simple_def": df["simplified_definition"][i]}
    )
    for i in range(len(df))
]

index.upsert(vectors)
print("✅ Financial jargon stored in Pinecone successfully!")
