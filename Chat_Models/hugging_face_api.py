from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Reson8/Rumble-2-Flash",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("what is the capital of India")

print(result)


# import os
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="remon-rakibul/TinyLama-1.1B-Chat-V0.3-AWQ",
#     task="text-generation",
#     # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
#     max_new_tokens=100
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India?")

# print(result.content)
