# Your existing imports...
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
from typing import Any, Dict, List, Optional, Union
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
import sys
from graphiti_core import Graphiti
import signal
import platform
from datetime import datetime, timezone
from graphiti_core.nodes import EpisodeType

# Add the AIEco directory to Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
print(parent_dir)
sys.path.insert(0, parent_dir)

# Import the fixed manager
from mcputil.sync_mcp_manager import SyncMCPManager, AsyncMCPManager

# Neo4j connection parameters
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def get_model():
    llm = 'o3'
    logger.info(f"Using model: {llm}")
    base_url = 'https://api.openai.com/v1'
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

brave_server = MCPServerStdio(
    'npx', 
    ['-y', '@modelcontextprotocol/server-brave-search'],
    env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
)

graphiti_agent = Agent(
    get_model(),
    system_prompt="""
        You are a professional neo4j AI agent who is specialized in analyzing data, creating and managing graph databases.
        First, you will analyze the provided data to list all episodes with related information, here are some examples of episodes for your reference: 
        Episode sample 1:
                {
                    'content': 'GPT-4.1 was released by OpenAI on April 14, 2025. It features improved capabilities in coding, instruction following, and long-context processing with a knowledge cutoff of June 2024.',
                    'type': EpisodeType.text,
                    'description': 'LLM research report',
                }
        Episode sample 2:
                {
                    'content': {
                        'name': 'Gemini 2.5 Pro',
                        'creator': 'Google',
                        'release_date': 'April 29, 2025',
                        'key_features': [
                            'Thinking mode',
                            'Video-to-code capabilities',
                            'Superior front-end web development'
                        ],
                        'ranking': 1,
                        'assessment': 'Currently the best LLM on the market'
                    },
                    'type': EpisodeType.json,
                    'description': 'LLM metadata',
                }
        Episode sample 3:
                {	
                    'content': 'Anthropic has just released Claude 4, their most advanced AI assistant to date. Claude 4 represents a significant leap forward in capabilities, outperforming all previous models including Gemini 2.5 Pro and GPT-4.1.',
                    'type': EpisodeType.text,
                    'description': 'LLM announcement',
                }
        Episode sample 4:
                {
                    'content': {
                        'name': 'Claude 4',
                        'creator': 'Anthropic',
                        'release_date': 'May 15, 2025',
                        'key_features': [
                            'Advanced reasoning engine',
                            'Multimodal processing',
                            'Improved factual accuracy',
                            'Tool use framework'
                        ],
                        'ranking': 1,
                        'assessment': 'Currently the best LLM on the market'
                    },
                    'type': EpisodeType.json,
                    'description': 'LLM metadata',
                }
        ...
    
        You will then create episodes based on content you have analyzed, and return the episodes in the format specified below.

        The required format of each episode is as follows:
            {   
                'content': 'Episode content goes here, this can be a string or a json with detailed information',
                'type': EpisodeType.text or EpisodeType.json
                'description': 'A brief description of the episode content'
            }

        Note:    
        1, There are no limitations on the number of episodes you can created.
        2, Don't complicate content for each episode, either type of text or json.
        3, For 'content' field: if it is a json, make sure it has name field.
        
        The format of the response:
        {
            'subject': 'A brief description of the user query',
            'episodes': 'array of episodes in the format specified above'
        }
    """,
    mcp_servers=[brave_server]
)

query = "do research on average aged care house price in Sydney, use brave search to find the latest data, and create a graph with the data you found, then return the graph in the format of an array of episodes."

class MCPServerManager:
    """Enhanced MCP Server Manager with proper cleanup"""
    
    def __init__(self):
        self.contexts = []
        self.cleanup_tasks = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def start_mcp_servers(self, agents):
        """Start MCP servers for given agents"""
        for agent in agents:
            try:
                context = agent.run_mcp_servers()
                await context.__aenter__()
                self.contexts.append(context)
                logger.info(f"Started MCP server for agent")
            except Exception as e:
                logger.error(f"Failed to start MCP server: {e}")
                raise
        return self.contexts
    
    async def cleanup(self):
        """Properly cleanup all MCP servers"""
        logger.info("Starting MCP server cleanup...")
        
        # Close all contexts
        for context in self.contexts:
            try:
                await context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during context cleanup: {e}")
        
        # Wait for cleanup tasks to complete
        if self.cleanup_tasks:
            try:
                await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error during cleanup tasks: {e}")
        
        # Give some time for subprocess cleanup
        await asyncio.sleep(0.1)
        
        logger.info("MCP server cleanup completed")

async def add_episodes(graphiti, result):
    """Add episodes to the graph with a given prefix."""
    prefix = result['subject']
    episodes = result['episodes']
    
    for i, episode in enumerate(episodes):
        # The episode type is already an enum object, not a string
        episode_type = episode['type']
        
        # Only convert if it's somehow a string (shouldn't happen with your current setup)
        if isinstance(episode_type, str):
            if episode_type == "EpisodeType.text":
                episode_type = EpisodeType.text
            elif episode_type == "EpisodeType.json":
                episode_type = EpisodeType.json
            else:
                episode_type = EpisodeType.text  # Default fallback
        
        await graphiti.add_episode(
            name=f'{prefix} {i}',
            episode_body=episode['content']
            if isinstance(episode['content'], str)
            else json.dumps(episode['content']),
            source=episode_type,
            source_description=episode['description'],
            reference_time=datetime.now(timezone.utc),
        )
        print(f'Added episode: {prefix} {i} ({episode_type})')

async def run_agent_with_proper_cleanup():
    """Run the agent with proper MCP server management"""
    async with MCPServerManager() as manager:
        try:
            # Start MCP servers
            await manager.start_mcp_servers([graphiti_agent])
            logger.info("MCP servers started successfully")
            
            # Run the query
            logger.info("Executing query...")
            result = await graphiti_agent.run(query)
            assess_result = result.data
            logger.info("Query completed successfully")
            
            return assess_result
            
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            raise

def handle_subprocess_cleanup():
    """Handle subprocess cleanup for Windows"""
    if platform.system() == "Windows":
        # Set subprocess creation flags for Windows
        import subprocess
        subprocess._PLATFORM_DEFAULT_CLOSE_FDS = False

async def main():
    """Main function with enhanced error handling and cleanup"""
    handle_subprocess_cleanup()
    
    try:
        result = await run_agent_with_proper_cleanup()
        print("Final result:", result)

        # Since result is already a Python dict, don't try to parse it as JSON
        if isinstance(result, dict):
            # Use the result directly
            await add_episodes(graphiti, result)
        elif isinstance(result, str):
            try:
                parsed_result = json.loads(result)
                await add_episodes(graphiti, parsed_result)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                return
        else:
            print(f"Unexpected result type: {type(result)}")
            return
        
        return result
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Final cleanup - wait a bit for any remaining subprocess cleanup
        await asyncio.sleep(0.2)
        logger.info("Application shutdown complete")

def run_main():
    """Run main with proper event loop handling"""
    try:
        # Use asyncio.run which handles event loop lifecycle properly
        return asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)

# Alternative approach using asyncio event loop policy (for Windows)
def run_main_with_policy():
    """Alternative approach with explicit event loop policy"""
    if platform.system() == "Windows":
        # Set the event loop policy for Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        # Properly close the loop
        try:
            # Cancel all running tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Wait for tasks to be cancelled
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Close the loop
            loop.close()
        except Exception as e:
            logger.warning(f"Error during loop cleanup: {e}")

if __name__ == '__main__':
    # Choose one of these approaches:
    
    # Approach 1: Standard asyncio.run (recommended for most cases)
    run_main()
    
    # Approach 2: Explicit event loop policy (use if Approach 1 doesn't work)
    # run_main_with_policy()