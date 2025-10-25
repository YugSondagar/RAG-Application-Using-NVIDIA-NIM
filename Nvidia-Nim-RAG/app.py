from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv

load_dotenv()

client = ChatNVIDIA(
  model="meta/llama3-70b-instruct",
  api_key=os.getenv("NVIDIA_API_KEY"), 
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
)

for chunk in client.stream([{"role":"user","content":"What is Machine Learning?"}]): 
  print(chunk.content, end="")