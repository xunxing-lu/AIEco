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
from pydantic_ai.mcp import MCPServerStdio
from contextlib import asynccontextmanager
import logging
import aiofiles
import base64
from openai import AsyncOpenAI
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

def get_model():
    llm = 'gpt-4o'
    logger.info(f"Using model: {llm}")
    base_url = 'https://api.openai.com/v1'
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

# Global agent variable
primary_agent = None

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await openai_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error
    
async def read_and_analyze_image(image_path: str, prompt: str = "What do you see in this image?"):
    """
    Async function to read an image file and send it to GPT-4o Vision for analysis using OpenAI client.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Question/prompt to ask about the image
        api_key (str): OpenAI API key
    
    Returns:
        dict: Response from GPT-4o Vision API
    """
    
    try:
        # Initialize async OpenAI client
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Read image file asynchronously
        async with aiofiles.open(image_path, 'rb') as image_file:
            image_data = await image_file.read()
        
        # Convert to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine image format
        image_path_obj = Path(image_path)
        image_format = image_path_obj.suffix.lower().replace('.', '')
        if image_format == 'jpg':
            image_format = 'jpeg'
        
        # Send request to OpenAI API using the official client
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model
        }
                    
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"Image file not found: {image_path}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}"
        }

async def search_my_knowledge_data(query: str) -> dict[str, str]:
    """
    Search for the knowledge data in supabase
    Use this tool when the user needs to find personal knowledge information.

    Args:
        query: The search query or instruction.

    Returns:
        The data from searched from supabase.
    """
    logger.info(f"Searching knowledge data with query: {query}")

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
            return {
                "result": """
                    ## No Knowledge Data Found

                    I couldn't find any Knowledge information matching your query. Please try:
                    - Using different search terms
                    - Checking for typos in your query
                    - Providing more specific details about what you're looking for

                    How would you like to proceed?
                    """
            }
            
        # Format the results as a table
        formatted_response = """
                            ## Knowledge Information Found

                            I found the following records based on your query:

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
                            2. Make sure there is 1 column: Content.
                            """
        
        return {"result": formatted_response}
        
    except Exception as e:
        logger.error(f"Error retrieving documentation: {e}")
        error_message = f"""
                        ## Error Retrieving Data

                        I encountered an error while searching for information: {str(e)}

                        Please try again in a few moments or contact support if the problem persists.
                        """
        return {"result": error_message}
    


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the FastAPI app and MCP servers"""
    global primary_agent
    
    # Startup
    logger.info("Starting up FastAPI application...")
    
    # Validate required environment variables
    required_env_vars = ["BRAVE_API_KEY", "OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    mcp_context = None
    try:
        # Create Brave Search MCP server
        logger.info("Setting up Brave Search MCP server...")
        brave_server = MCPServerStdio(
            'npx', 
            ['-y', '@modelcontextprotocol/server-brave-search'],
            env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
        )

        whatsapp_server = MCPServerStdio(
            'python',
            [r'C:\Projects\WhatsApp\whatsapp-mcp\whatsapp-mcp-server\main.py']
        )

        # Create the agent
        logger.info("Creating primary agent...")
        primary_agent = Agent(
            get_model(),
            system_prompt="""
                You are a helpful AI agent with access to web search capabilities, WhatsApp functionality, personal knowledge data, image vision.

                When users ask questions:
                1. First check if you need to search personal knowledge using the search_my_knowledge_data tool
                2. If you need current information or web searches, use the Brave search capabilities
                3. If you need to send WhatsApp messages or interact with WhatsApp, use the WhatsApp MCP server tools
                4. Always cite your sources when using search results
                5. Provide accurate, helpful, and well-formatted responses
                6. use read_and_analyze_image tool to analyze images
                """,
            mcp_servers=[brave_server, whatsapp_server]
        )
        
        # Register the knowledge search tool
        primary_agent.tool_plain(search_my_knowledge_data)
        primary_agent.tool_plain(read_and_analyze_image)
        
        # Start MCP servers using the context manager properly
        logger.info("Starting MCP servers...")
        mcp_context = primary_agent.run_mcp_servers()
        await mcp_context.__aenter__()
        logger.info("MCP servers started successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Try to create agent without MCP server as fallback
        logger.info("Falling back to agent without MCP server...")
        primary_agent = Agent(
            get_model(),
            system_prompt="""
You are a helpful AI agent with access to personal knowledge data.

When users ask questions:
1. Check if you need to search personal knowledge using the search_my_knowledge_data tool
2. Provide accurate, helpful, and well-formatted responses
3. Note that web search capabilities are currently unavailable
"""
        )
        primary_agent.tool_plain(search_my_knowledge_data)
        logger.info("Fallback agent created successfully!")
        yield
    finally:
        # Shutdown
        logger.info("Shutting down application...")
        if mcp_context:
            try:
                await mcp_context.__aexit__(None, None, None)
                logger.info("MCP servers closed successfully!")
            except Exception as e:
                logger.error(f"Error closing MCP servers: {e}")
        elif primary_agent:
            logger.info("Cleaning up fallback agent...")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

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
    if not primary_agent:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent not initialized. Please check server logs."}
        )
    
    user_message = request.messages[-1].content
    
    if request.stream:
        # Return streaming response
        logger.info("Streaming response enabled")
        return StreamingResponse(
            stream_chat_response(user_message, request.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        # Return non-streaming response
        try:
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
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal server error: {str(e)}"}
            )

async def stream_chat_response(message: str, model: str):
    """Generate streaming chat response in OpenAI format"""
    chunk_id = "chatcmpl-local-stream"
    
    try:
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
        async with primary_agent.run_stream(message) as response:
            data = await response.get_data()
            # Create a streaming chunk with the content
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
        logger.error(f"Error in streaming response: {e}")
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
    try:
        result = await primary_agent.run(message)
        return result.data
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_initialized": primary_agent is not None,
        "timestamp": asyncio.get_event_loop().time()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Usage with Docker:
# docker run -p 3000:3000 -e OPENAI_API_KEY=your-key -e OPENAI_API_BASE=http://host.docker.internal:8000 openwebui/openwebui

# For local ollama:
# docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
