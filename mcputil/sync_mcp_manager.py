import asyncio
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class SyncMCPManager:
    def __init__(self):
        self.contexts = []
        self.loop = None
        self.started = False
        
    def start_servers(self, agents):
        """Start MCP servers for the given agents synchronously"""
        if self.started:
            logger.warning("Servers already started, stopping previous ones first")
            self.stop_servers()
            
        # Create a new event loop for this context
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.contexts = self.loop.run_until_complete(self._start_servers_async(agents))
            self.started = True
            return self.contexts
        except Exception as e:
            logger.error(f"Failed to start servers: {e}")
            # Clean up the loop if startup failed
            self.loop.close()
            self.loop = None
            raise
    
    async def _start_servers_async(self, agents):
        """Async implementation of server startup"""
        contexts = []
        try:
            for agent in agents:
                logger.info(f"Starting MCP servers for agent")
                context = agent.run_mcp_servers()
                await context.__aenter__()
                contexts.append(context)
                logger.info(f"Successfully started MCP servers for agent")
            return contexts
        except Exception as e:
            # Clean up any contexts that were successfully started
            logger.error(f"Error starting servers, cleaning up: {e}")
            for context in contexts:
                try:
                    await context.__aexit__(None, None, None)
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup: {cleanup_error}")
            raise
    
    def stop_servers(self):
        """Stop all MCP servers synchronously"""
        if not self.started or not self.contexts:
            logger.info("No servers to stop")
            return
            
        try:
            self.loop.run_until_complete(self._stop_servers_async())
        except Exception as e:
            logger.error(f"Error stopping servers: {e}")
        finally:
            self.started = False
            self.contexts = []
            if self.loop and not self.loop.is_closed():
                self.loop.close()
            self.loop = None
    
    async def _stop_servers_async(self):
        """Async implementation of server shutdown"""
        for i, context in enumerate(self.contexts):
            try:
                logger.info(f"Stopping MCP server context {i}")
                await context.__aexit__(None, None, None)
                logger.info(f"Successfully stopped MCP server context {i}")
            except Exception as e:
                logger.error(f"Error stopping context {i}: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_servers()


# Alternative approach using a context manager
@asynccontextmanager
async def mcp_server_context(agents):
    """Async context manager for MCP servers"""
    contexts = []
    try:
        for agent in agents:
            context = agent.run_mcp_servers()
            await context.__aenter__()
            contexts.append(context)
        yield contexts
    finally:
        i=1
        # for context in contexts:
        #     try:
        #         await context.__aexit__(None, None, None)
        #     except Exception as e:
        #         logger.error(f"Error during MCP server cleanup: {e}")


# Synchronous wrapper for the async context manager
class AsyncMCPManager:
    """A cleaner approach using proper async context management"""
    
    @staticmethod
    def run_with_mcp_servers(agents, query_func):
        """
        Run a function with MCP servers active
        
        Args:
            agents: List of agents that need MCP servers
            query_func: Async function to run with servers active
        """
        async def _run():
            async with mcp_server_context(agents) as contexts:
                return await query_func()
        
        return asyncio.run(_run())
    
    @staticmethod
    def run_sync_query(agent, query):
        """
        Convenience method to run a sync query with MCP servers
        
        Args:
            agent: Single agent to start servers for
            query: Query string to run
        """
        async def _query():
            return agent.run_sync(query)
        
        return AsyncMCPManager.run_with_mcp_servers([agent], _query)