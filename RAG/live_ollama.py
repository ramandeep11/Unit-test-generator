from langchain_community.llms import Ollama
from langchain.callbacks.base import BaseCallbackHandler

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.partial_output = ""

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.partial_output += token
        print(token, end="", flush=True)

llm = Ollama(model="llama3.1", callbacks=[StreamingCallbackHandler()])

response = llm.invoke("What is the capital of France?")

print(response)

