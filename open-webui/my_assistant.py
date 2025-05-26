from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import uvicorn
import json
import asyncio
from supabase import create_client, Client
import os
from openai import AsyncOpenAI
from typing import Any, Dict, List
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

def get_model():
    llm = 'gpt-4o'
    print(llm)
    base_url = 'https://api.openai.com/v1'
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

primary_agent = Agent(
    get_model(),
    system_prompt="""
    you are a good local agent that processes the input message and returns a response.
    """
)

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await openai_client.embeddings.create(
            model= embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@primary_agent.tool_plain
async def search_my_knowledge_data(query: str) -> dict[str, str]:
    """
    Search for the knowledge data in supabase
    Use this tool when the user needs to find personal knowledge information.

    Args:
        ctx: The run context.
        query: The search query or instruction.

    Returns:
        The data from searched from supabase.
    """
    print(f"Calling primary agent with query: {query}")

    supabase: Client = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )

    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(query)
        
        # Query Supabase for relevant documents
        result = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_count': 10,
            }
        ).execute()
        
        if not result.data:
            return """
            ## No Knowledge Data Found
            
            I couldn't find any Knowledge information matching your query. Please try:
            - Using different search terms
            - Checking for typos in your query
            - Providing more specific details about what you're looking for
            
            How would you like to proceed?
            """
            
        # Format the results as a table
        formatted_response = """
        ## Knowledge Information Found
        
        I found the following patient records based on your query:
        
        | Content |
        |---------|
        """
        
        for doc in result.data:
            # Escape any pipe characters in the strings to avoid breaking the table
            content = str(doc.get('content', ''))
                
            # Add the row to the table
            formatted_response += f"| {content} |\n"
            
        formatted_response += """
        
        ## Response Instructions
        When responding to the user:
        1. Present the table of content information
        2. Make sure there are 1 column: Content.
        """
        
        return {"result": formatted_response}
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        error_message = f"""
        ## Error Retrieving Patient Data
        
        I encountered an error while searching for patient information: {str(e)}
        
        Please apologize to the user for the technical difficulty and suggest they:
        - Try again in a few moments
        - Contact technical support if the problem persists
        - Try a different search query
        """
        return error_message

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.7
    stream: bool = False

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

    print(request.stream)
    
    if True:
        # Return streaming response
        print("Streaming response enabled")
        return StreamingResponse(
            stream_chat_response(user_message, request.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        # Return non-streaming response (original behavior)
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

async def stream_chat_response(message: str, model: str):
    """Generate streaming chat response in OpenAI format"""
    try:
        # Start the streaming response
        chunk_id = "chatcmpl-local-stream"
        
        # Send initial chunk
        initial_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": ""
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"
        
        # Stream the agent response
        accumulated_content = ""
        async with primary_agent.run_stream(message) as response:

            data = await response.get_data()
            # Create a streaming chunk with the new content
            stream_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": data
                        },
                        "finish_reason": None
                    }
                ]
            }
            
            yield f"data: {json.dumps(stream_chunk)}\n\n"
        
        # Send final chunk to indicate completion
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        # Send error chunk
        error_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk", 
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": f"Error: {str(e)}"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

async def run_local_agent(message):
    """Non-streaming agent run for backward compatibility"""
    result = await primary_agent.run(message)
    return result.data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# docker run -p 3000:3000 -e OPENAI_API_KEY=your-key -e OPENAI_API_BASE=http://host.docker.internal:8000 openwebui/openwebui

# ollama installed locally
# docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main