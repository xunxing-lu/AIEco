from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
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
from typing import Any, Dict, List, Optional,Union
from dotenv import load_dotenv
from pydantic_ai.mcp import MCPServerStdio
from contextlib import asynccontextmanager
import logging
import aiofiles
import base64
from openai import AsyncOpenAI
from pathlib import Path
import PyPDF2
import docx2txt
import tempfile
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def get_model():
    llm = 'gpt-4o'
    logger.info(f"Using model: {llm}")
    base_url = 'https://api.openai.com/v1'
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

# Global agent variable
primary_agent = None
file_agent = Agent(
                get_model(),
                system_prompt="""
                    You are a helpful AI agent who can tell the type of the file, and read its content.
                """,
            )

github_agent = Agent(
                get_model(),  
                system_prompt="""
                    You are a GitHub specialist. Help users interact with GitHub repositories and features.
                """,
            )

crawai_agent = Agent(
                get_model(),
                system_prompt="""
                    You are a web crawl specialist. Help users interact with website crawl requiremnt.
                """,
            )

brave_agent = Agent(
                get_model(),
                system_prompt="""
                    You are a web content research specialist. Help users interact with web content research requiremnt.
                """,
            )


# Store uploaded files metadata
uploaded_files = {}

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

async def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return the file path."""
    # Create unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(upload_file.filename).suffix
    unique_filename = f"{timestamp}_{upload_file.filename}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await upload_file.read()
        await f.write(content)
    
    # Store file metadata
    uploaded_files[str(file_path)] = {
        "original_name": upload_file.filename,
        "content_type": upload_file.content_type,
        "size": len(content),
        "upload_time": datetime.now().isoformat()
    }
    
    return str(file_path)


async def read_and_analyze_file(query: str):
    """
    read and analyze file use file agent
    Use this tool when the user needs read content from files

    Args:
        query: The full file path to the file agent.

    Returns:
        The response from the file agent.
    """
    print(f"Calling file agent with query: {query}")
    result = await file_agent.run(query)
    return {"result": result.data}

async def use_github_agent(query: str) -> dict[str, str]:
    """
    Interact with GitHub through the GitHub subagent.
    Use this tool when the user needs to access repositories, issues, PRs, or other GitHub resources.

    Args:
        ctx: The run context.
        query: The instruction for the GitHub agent.

    Returns:
        The response from the GitHub agent.
    """
    print(f"Calling GitHub agent with query: {query}")
    result = await github_agent.run(query)
    return {"result": result.data}

async def use_craw_ai_agent(query: str) -> dict[str, str]:
    """
    Interact with web crawl through the craw ai subagent.
    Use this tool when the user needs to crawl website.

    Args:
        ctx: The run context.
        query: The query for the craw ai agent.

    Returns:
        The response from the craw ai agent.
    """
    print(f"Calling Craw AI agent with query: {query}")
    result = await crawai_agent.run(query)
    return {"result": result.data}

async def use_brave_search_agent(query: str) -> dict[str, str]:
    """
    Interact with web content research through the brave search subagent.
    Use this tool when the user needs to research on web.

    Args:
        ctx: The run context.
        query: The query for the brave search agent.

    Returns:
        The response from the brave search agent.
    """
    print(f"Calling brave search agent with query: {query}")
    result = await  brave_agent.run(query)
    return {"result": result.data}

@file_agent.tool_plain
async def read_and_analyze_pdf_file(file_path: str):
    """
    Async function to read content from a pdf file when asked
    
    Args:
        file_path (str): Path to the pdf file
    
    Returns:
        json
    """

    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            # result = {
            #     "text": text_content.strip(),
            #     "page_count": len(pdf_reader.pages),
            #     "metadata": dict(pdf_reader.metadata) if pdf_reader.metadata else {},
            #     "is_encrypted": pdf_reader.is_encrypted
            # }
            
            return {
                "success": True,
                "error": None,
                "result": text_content.strip()
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result": None
        }
    
@file_agent.tool_plain
async def read_and_analyze_txt_file(file_path: str):
    """
    Async function to read content from a txt file when asked
    
    Args:
        file_path (str): Path to the txt file
    
    Returns:
        json
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return {
            "success": True,
            "error": None,
            "result": content
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result": None
        }

@file_agent.tool_plain
async def read_and_analyze_word_file(file_path: str):
    """
    Async function to read content from a word file when asked
    
    Args:
        file_path (str): Path to the word file
    
    Returns:
        json
    """
    try:
        # Use docx2txt for simple text extraction
        text_content = docx2txt.process(file_path)
        
        return {
            "success": True,
            "error": None,
            "result": text_content
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result": None
        }



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
    
async def process_image_content(image_data: str, prompt: str = "What do you see in this image?") -> str:
    """Process image data from data URL"""
    try:
        if not image_data.startswith("data:image/"):
            return "Invalid image data format"
            
        # Extract base64 data and save as temp file
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        # Determine file extension from header
        if "jpeg" in header or "jpg" in header:
            ext = ".jpg"
        elif "png" in header:
            ext = ".png"
        elif "gif" in header:
            ext = ".gif"
        elif "webp" in header:
            ext = ".webp"
        else:
            ext = ".jpg"  # Default
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Use your existing image analysis function
            image_result = await read_and_analyze_image(temp_file_path, prompt)
            if image_result.get("success"):
                return image_result.get("content", "Could not analyze image")
            else:
                return f"Error analyzing image: {image_result.get('error', 'Unknown error')}"
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error processing image content: {e}")
        return f"Error processing image: {str(e)}"

async def process_file_content(file_url: str, file_type: str, file_name: str, prompt: str) -> str:
    """Process file content from URL"""
    try:
        if file_url.startswith("data:"):
            # Handle data URL (base64 encoded file)
            header, encoded = file_url.split(",", 1)
            file_bytes = base64.b64decode(encoded)
            
            # Determine file extension
            if "pdf" in header:
                ext = ".pdf"
            elif "msword" in header or "officedocument.wordprocessingml" in header:
                ext = ".docx"
            elif "text/plain" in header:
                ext = ".txt"
            else:
                # Try to get extension from file_name
                ext = Path(file_name).suffix or ".bin"
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name
            
            try:
                return await analyze_file_by_extension(temp_file_path, ext, prompt)
            finally:
                os.unlink(temp_file_path)
        else:
            return "URL-based file processing not implemented"
            
    except Exception as e:
        logger.error(f"Error processing file content: {e}")
        return f"Error processing file: {str(e)}"


async def process_document_content(doc_content: str, mime_type: str, doc_name: str, prompt: str) -> str:
    """Process document content from base64"""
    try:
        # Decode base64 content
        file_bytes = base64.b64decode(doc_content)
        
        # Determine file extension from MIME type
        if mime_type == "application/pdf":
            ext = ".pdf"
        elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            ext = ".docx"
        elif mime_type == "text/plain":
            ext = ".txt"
        else:
            # Try to get extension from doc_name
            ext = Path(doc_name).suffix or ".bin"
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name
        
        try:
            return await analyze_file_by_extension(temp_file_path, ext, prompt)
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error processing document content: {e}")
        return f"Error processing document: {str(e)}"


async def analyze_file_by_extension(file_path: str, extension: str, prompt: str) -> str:
    """Analyze file based on its extension"""
    try:
        if extension.lower() == ".pdf":
            # Use your existing PDF reading function
            result = await read_and_analyze_pdf_file(file_path)
            if result.get("success"):
                content = result.get("result", "")
                return f"PDF Content: {content[:1000]}..." if len(content) > 1000 else f"PDF Content: {content}"
            else:
                return f"Error reading PDF: {result.get('error', 'Unknown error')}"
                
        elif extension.lower() in [".docx", ".doc"]:
            # Use your existing Word reading function
            result = await read_and_analyze_word_file(file_path)
            if result.get("success"):
                content = result.get("result", "")
                return f"Word Document Content: {content[:1000]}..." if len(content) > 1000 else f"Word Document Content: {content}"
            else:
                return f"Error reading Word document: {result.get('error', 'Unknown error')}"
                
        elif extension.lower() == ".txt":
            # Use your existing text reading function
            result = await read_and_analyze_txt_file(file_path)
            if result.get("success"):
                content = result.get("result", "")
                return f"Text File Content: {content[:1000]}..." if len(content) > 1000 else f"Text File Content: {content}"
            else:
                return f"Error reading text file: {result.get('error', 'Unknown error')}"
                
        else:
            return f"Unsupported file type: {extension}"
            
    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")
        return f"Error analyzing file: {str(e)}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the FastAPI app and MCP servers"""
    global primary_agent
    global github_agent
    
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

        github_mcp_server = MCPServerStdio(
            'npx',
            ['-y', '@modelcontextprotocol/server-github'],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_ACCESS_TOKEN")}
        )

        craw_ai_server = MCPServerStdio(
            'C:\\Projects\\crawl4ai-mcp\\venv\\Scripts\\python.exe',
            ['C:\\Projects\\crawl4ai-mcp\\crawl_mcp.py']
        )

        github_agent._mcp_servers = [github_mcp_server]
        crawai_agent._mcp_servers = [craw_ai_server]
        brave_agent._mcp_servers = [brave_server]

        # Create the agent
        logger.info("Creating primary agent...")
        primary_agent = Agent(
            get_model(),
            system_prompt="""
                You are a helpful AI agent with access to web search capabilities, WhatsApp functionality, personal knowledge data, image vision.

                When users ask questions:
                1. First check if you need to search personal knowledge using the search_my_knowledge_data tool
                2. If you need to send WhatsApp messages or interact with WhatsApp, use the WhatsApp MCP server tools
                3. Always cite your sources when using search results
                4. Provide accurate, helpful, and well-formatted responses
                5. use read_and_analyze_image tool to analyze images
                6. use read_and_analyze_file tool to read content from file
                """,
            mcp_servers=[whatsapp_server]
        )
        
        # Register the knowledge search tool
        primary_agent.tool_plain(search_my_knowledge_data)
        primary_agent.tool_plain(read_and_analyze_image)
        primary_agent.tool_plain(read_and_analyze_file)
        primary_agent.tool_plain(use_github_agent)
        primary_agent.tool_plain(use_craw_ai_agent)
        primary_agent.tool_plain(use_brave_search_agent)
        
        # Start MCP servers using the context manager properly
        logger.info("Starting MCP servers...")
        mcp_context = primary_agent.run_mcp_servers()
        await mcp_context.__aenter__()
        mcp_context_github = github_agent.run_mcp_servers()
        await mcp_context_github.__aenter__()
        mcp_context_craw_ai = crawai_agent.run_mcp_servers()
        await mcp_context_craw_ai.__aenter__()
        mcp_context_brave = brave_agent.run_mcp_servers()
        await mcp_context_brave.__aenter__()
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

# class Message(BaseModel):
#     role: str
#     content: str
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]  # Keep it simple - just allow dicts

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    stream: bool = False

# New file upload endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file"""
    try:
        # Check file size (limit to 50MB)
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")
        
        # Reset file pointer
        await file.seek(0)
        
        # Save file
        file_path = await save_uploaded_file(file)
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "file_path": file_path,
            "content_type": file.content_type,
            "size": len(content)
        }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload/multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Upload multiple files"""
    uploaded_files_info = []
    
    for file in files:
        try:
            # Check file size
            content = await file.read()
            if len(content) > 50 * 1024 * 1024:  # 50MB
                uploaded_files_info.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": "File too large. Maximum size is 50MB."
                })
                continue
            
            # Reset file pointer
            await file.seek(0)
            
            # Save file
            file_path = await save_uploaded_file(file)
            
            uploaded_files_info.append({
                "filename": file.filename,
                "status": "success",
                "file_path": file_path,
                "content_type": file.content_type,
                "size": len(content)
            })
        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {e}")
            uploaded_files_info.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "message": f"Processed {len(files)} files",
        "files": uploaded_files_info
    }

@app.post("/chat/with-files")
async def chat_with_files(
    message: str = Form(...),
    files: Optional[List[UploadFile]] = File(None)
):
    """Chat endpoint that accepts both message and files"""
    try:
        # Upload files if provided
        file_paths = []
        if files:
            for file in files:
                file_path = await save_uploaded_file(file)
                file_paths.append(file_path)
        
        # Enhance message with file information
        enhanced_message = message
        if file_paths:
            enhanced_message += f"\n\nI have uploaded the following files: {', '.join([Path(fp).name for fp in file_paths])}"
            enhanced_message += f"\nFile paths: {', '.join(file_paths)}"
            enhanced_message += "\nPlease analyze these files and respond to my message."
        
        # Run agent
        if not primary_agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        result = await primary_agent.run(enhanced_message)
        
        return {
            "response": result.data,
            "uploaded_files": file_paths
        }
        
    except Exception as e:
        logger.error(f"Error in chat with files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_uploaded_files():
    """List all uploaded files"""
    return {
        "files": [
            {
                "file_path": file_path,
                "original_name": metadata["original_name"],
                "content_type": metadata["content_type"],
                "size": metadata["size"],
                "upload_time": metadata["upload_time"]
            }
            for file_path, metadata in uploaded_files.items()
        ]
    }

@app.delete("/files/{file_path:path}")
async def delete_file(file_path: str):
    """Delete an uploaded file"""
    try:
        full_path = Path(file_path)
        if full_path.exists():
            full_path.unlink()
            if str(full_path) in uploaded_files:
                del uploaded_files[str(full_path)]
            return {"message": f"File {file_path} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    
    # user_message = request.messages[-1].content
    last_message = request.messages[-1]
    
    # Handle multimodal content (text + files)
    if isinstance(last_message.content, list):
        text_content = ""
        processed_files = []
        
        for content_item in last_message.content:
            content_type = content_item.get("type", "")
            
            if content_type == "text":
                text_content = content_item.get("text", "")
                
            elif content_type == "image_url":
                image_data = content_item.get("image_url", {}).get("url", "")
                if image_data:
                    image_result = await process_image_content(image_data, text_content or "What do you see in this image?")
                    processed_files.append(f"Image analysis: {image_result}")
                    
            elif content_type == "file_url":
                file_data = content_item.get("file_url", {})
                file_url = file_data.get("url", "")
                file_type = file_data.get("type", "")
                file_name = file_data.get("name", "unknown")
                
                if file_url:
                    file_result = await process_file_content(file_url, file_type, file_name, text_content)
                    processed_files.append(f"File analysis ({file_name}): {file_result}")
                    
            elif content_type == "document":
                doc_data = content_item.get("document", {})
                doc_content = doc_data.get("content", "")
                doc_type = doc_data.get("mime_type", "")
                doc_name = doc_data.get("name", "unknown")
                
                if doc_content:
                    file_result = await process_document_content(doc_content, doc_type, doc_name, text_content)
                    processed_files.append(f"Document analysis ({doc_name}): {file_result}")
        
        # Combine text and file analysis results
        if processed_files:
            user_message = f"{text_content}\n\nFile Analysis Results:\n" + "\n".join(processed_files)
        else:
            user_message = text_content
    else:
        # Handle simple text content
        user_message = last_message.content
    
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
