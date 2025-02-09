# This example walks through building a retrieval augmented generation (RAG) application using Ollama and embedding models.
import chromadb
from ollama import chat, embeddings
from ollama import ChatResponse
import ollama

# response: ChatResponse = chat(model='llama3.2', messages=[
#   {
#     'role': 'user',
#     'content': 'So if we shine a bright light from earth to the sky will not be blue?',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)
# documents = [
#   "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
#   "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
#   "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
#   "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
#   "Llamas are vegetarians and have very efficient digestive systems",
#   "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
# ]

documents = [
   "John is married to charity olong",
   "John has 5 brothers and come from a royal family in Kogi state",
   "John studied at the university of Kogi state and graduated with a second class upper in computer science",
   "John is a software engineer and has worked with Fnz group for 3 years",
]

client = chromadb.Client()
collection = client.create_collection(name="docs")

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = embeddings(model="mxbai-embed-large", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )

  # an example prompt
prompt = "what do you know about John life and his career and education?"

# generate an embedding for the prompt and retrieve the most relevant doc
response = embeddings(
  prompt=prompt,
  model="mxbai-embed-large"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)
data = results['documents'][0][0]

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="llama3.2",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])