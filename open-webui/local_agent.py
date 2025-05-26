from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import uvicorn

app = FastAPI()

def get_model():
    llm = 'gpt-4o'
    print(llm)
    base_url = 'https://api.openai.com/v1'
    print(base_url)
    api_key = ''
    print(api_key)
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

primary_agent = Agent(
    get_model(),
    system_prompt="""
    you are a good local agent that processes the input message and returns a response.
    """
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.7

@app.get("/v1/models")
def list_models():
    return JSONResponse(content={
        "data": [
            {
                "id": "local-model",
                "object": "model",
                "created": 0,
                "owned_by": "you",
                "permission": [],
            }
        ],
        "object": "list"
    })

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    user_message = request.messages[-1].content
    # Your local agent logic here:
    agent_response = await run_local_agent(user_message)

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 0,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": agent_response
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

async def run_local_agent(message):
    # Replace this with your custom logic
    retult = await primary_agent.run(message)
    return retult.data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# docker run -p 3000:3000 -e OPENAI_API_KEY=your-key -e OPENAI_API_BASE=http://host.docker.internal:8000 openwebui/openwebui

# ollama installed locally
# docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main