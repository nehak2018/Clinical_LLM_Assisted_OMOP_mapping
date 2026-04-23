from ollama import chat

response = chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Hello, what is the day today!'}],
)
print(response.message.content)

'''
from ollama import chat

response = chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response.message.content)
'''