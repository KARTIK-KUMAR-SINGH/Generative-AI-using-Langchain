from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-4-hydrogen')

result = model.invoke("what is the temprature today at bangalore")

print(result)
