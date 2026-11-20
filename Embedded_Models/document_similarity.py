from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ['HF_HOME'] = 'D:/hugging_face_cache'
embedding = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2' ,
)

documents = [
    "Virat Kholi is an Indian Cricketer and he plays cover drive perfectly" ,
    "Rohit Sharma is an Indian cricketer and he is known for his pull shot",
    "MS Dhoni is a former Indian captain and he is famous for his finishing skills",
    "Jasprit Bumrah is an Indian fast bowler and he is known for his unique bowling action",
    "Hardik Pandya is an Indian all-rounder who can contribute with both bat and ball",
    "Shubman Gill is an Indian batsman known for his elegant batting style",
    "Ravindra Jadeja is an Indian all-rounder known for his fielding and spin bowling",
    "Sachin Tendulkar is a legendary Indian cricketer known as the Master Blaster",
    "Suryakumar Yadav is an Indian batsman known for his innovative shots",
    "Yuvraj Singh is a former Indian cricketer famous for hitting six sixes in an over",
]
query = "tell me about bumrah"
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
scores = cosine_similarity([query_embedding] , doc_embeddings)[0]
# index , score = sorted(list(enumerate(scores)) , key = lambda x : x[1][-1])
index = scores.argmax()
score = scores[index]
print(query)
print(documents[index])
print("similarity score is :- " , score)