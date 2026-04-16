from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding_model = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Virat Kohli is an Indian cricketer known for his agressive attitude",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills",
    "Sachin Tendulkar (known as God of cricket) holds many records",
    "Rohit Sharma is known for his elegant batting",
    "Jasprit Bumrah is an Indian fastest bowler for his unorthodox actions and yorkers"
]

doc_embedding = embedding_model.embed_documents(documents)

query = "Tell me about MS Dhoni"

query_embedding = embedding_model.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embedding)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)

print(documents[index])