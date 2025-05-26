"""A primary agent with subagents for various tasks, using the pydantic_ai framework.

This file demonstrates a structure for an 'army' of AI subagents, each specialized
in integrating with a specific third-party service or functionality:

    1. AirtableSubagent
    2. BraveSearchSubagent
    3. FileSystemSubagent
    4. GitHubSubagent
    5. SlackSubagent
    6. FirecrawlSubagent

We also declare a PrimaryAgent that uses these subagents to handle specialized tasks.

ENVIRONMENT VARIABLES (see .env.example):
    AIRTABLE_API_KEY
    BRAVE_API_KEY
    SLACK_BOT_TOKEN
    GITHUB_TOKEN
    FIRECRAWL_API_KEY

Do NOT actually start the MCP servers each time a tool is called. Instead, the
PrimaryAgent provides async startup and teardown support with AsyncExitStack to
minimize overhead. Tools in the PrimaryAgent simply trigger the subagents'
respective functionalities. The docstrings for each subagent note when you would
call them (Slack for messaging, Brave for web search, etc.).

Make sure you have installed and configured the 'pydantic_ai' framework in your environment.
"""

import os
import asyncio
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# pydantic_ai imports
try:
    from pydantic_ai import AIModel, AIConfig, Tool, AIBotResponse
    from pydantic_ai.mcp import MCPServer, MCPClient
except ImportError:
    # Placeholder fallback if pydantic_ai isn't installed
    class AIModel:
        pass
    class AIConfig:
        pass
    def Tool(name: str, description: str):
        def decorator(func):
            return func
        return decorator
    class AIBotResponse:
        pass
    class MCPServer:
        pass
    class MCPClient:
        pass


# MCP Server base class for our subagents
class SubagentMCPServer:
    """Base class for MCP servers used by subagents."""
    
    def __init__(self, name: str, env_var: Optional[str] = None):
        self.name = name
        self.env_var = env_var
        self.api_key = os.getenv(env_var) if env_var else None
        self.server = None
        
    async def __aenter__(self):
        """Start the MCP server when entering the async context."""
        # This would be replaced with actual MCP server initialization
        print(f"Starting {self.name} MCP server...")
        self.server = MCPServer(name=self.name)
        # Configure with API key if available
        if self.env_var and self.api_key:
            self.server.configure(api_key=self.api_key)
        await self.server.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop the MCP server when exiting the async context."""
        if self.server:
            print(f"Stopping {self.name} MCP server...")
            await self.server.stop()
            self.server = None

class AirtableMCPServer(SubagentMCPServer):
    """MCP server for Airtable operations."""
    
    def __init__(self):
        super().__init__("airtable", "AIRTABLE_API_KEY")
    
    async def list_bases(self) -> List[Dict[str, Any]]:
        """List all Airtable bases the user has access to."""
        # This would be implemented with actual Airtable API calls
        return [{"id": "appXXXXXXXXXXXXXX", "name": "Example Base"}]
    
    async def list_tables(self, base_id: str) -> List[Dict[str, Any]]:
        """List all tables in a specific base."""
        return [{"id": "tblXXXXXXXXXXXXXX", "name": "Example Table"}]
    
    async def get_records(self, base_id: str, table_id: str, max_records: int = 100) -> List[Dict[str, Any]]:
        """Get records from a specific table."""
        return [{"id": "recXXXXXXXXXXXXXX", "fields": {"Name": "Example Record"}}]
    
    async def create_record(self, base_id: str, table_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new record in a specific table."""
        return {"id": "recYYYYYYYYYYYYYY", "fields": fields}


class AirtableSubagent(AIModel):
    """Subagent specialized in interacting with Airtable.
    Use this subagent to read and write records in Airtable.

    Expects:
        AIRTABLE_API_KEY in environment variables.

    Will connect to an MCP server providing Airtable actions,
    but does NOT start the server automatically.
    """

    class Config(AIConfig):
        pass
    
    def __init__(self):
        super().__init__()
        self.mcp_server = AirtableMCPServer()
        self.client = None
    
    async def __aenter__(self):
        """Enter the async context, starting the MCP server."""
        await self.mcp_server.__aenter__()
        self.client = MCPClient(server_name=self.mcp_server.name)
        await self.client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, stopping the MCP server."""
        if self.client:
            await self.client.disconnect()
        await self.mcp_server.__aexit__(exc_type, exc_val, exc_tb)

    async def run(self, operation: str, base_id: Optional[str] = None,
                 table_id: Optional[str] = None, **kwargs) -> AIBotResponse:
        """Perform an Airtable-related operation.
        
        Args:
            operation: The operation to perform (list_bases, list_tables, get_records, create_record)
            base_id: The ID of the Airtable base
            table_id: The ID of the Airtable table
            **kwargs: Additional arguments for the operation
            
        Returns:
            AIBotResponse with the operation result
        """
        if not self.client:
            return AIBotResponse(message="Error: Airtable client not initialized")
        
        try:
            if operation == "list_bases":
                result = await self.mcp_server.list_bases()
                return AIBotResponse(message=f"Found {len(result)} Airtable bases", data=result)
            
            elif operation == "list_tables" and base_id:
                result = await self.mcp_server.list_tables(base_id)
                return AIBotResponse(message=f"Found {len(result)} tables in base {base_id}", data=result)
            
            elif operation == "get_records" and base_id and table_id:
                max_records = kwargs.get("max_records", 100)
                result = await self.mcp_server.get_records(base_id, table_id, max_records)
                return AIBotResponse(message=f"Retrieved {len(result)} records from table {table_id}", data=result)
            
            elif operation == "create_record" and base_id and table_id:
                fields = kwargs.get("fields", {})
                result = await self.mcp_server.create_record(base_id, table_id, fields)
                return AIBotResponse(message=f"Created record in table {table_id}", data=result)
            
            else:
                return AIBotResponse(message=f"Invalid operation or missing parameters: {operation}")
        
        except Exception as e:
            return AIBotResponse(message=f"Airtable operation error: {str(e)}")


class BraveSearchMCPServer(SubagentMCPServer):
    """MCP server for Brave Search operations."""
    
    def __init__(self):
        super().__init__("brave_search", "BRAVE_API_KEY")
    
    async def search(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """Perform a web search using Brave Search API."""
        # This would be implemented with actual Brave Search API calls
        return [
            {
                "title": f"Result for {query} #{i}",
                "url": f"https://example.com/result{i}",
                "description": f"This is a sample result for the query: {query}"
            }
            for i in range(1, count + 1)
        ]


class BraveSearchSubagent(AIModel):
    """Subagent specialized in performing Brave web searches.
    Use this subagent to conduct web searches using the Brave Search API.

    Expects:
        BRAVE_API_KEY in environment variables.

    Will connect to an MCP server providing Brave search actions,
    but does NOT start the server automatically.
    """

    class Config(AIConfig):
        pass
    
    def __init__(self):
        super().__init__()
        self.mcp_server = BraveSearchMCPServer()
        self.client = None
    
    async def __aenter__(self):
        """Enter the async context, starting the MCP server."""
        await self.mcp_server.__aenter__()
        self.client = MCPClient(server_name=self.mcp_server.name)
        await self.client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, stopping the MCP server."""
        if self.client:
            await self.client.disconnect()
        await self.mcp_server.__aexit__(exc_type, exc_val, exc_tb)

    async def run(self, query: str, count: int = 10, **kwargs) -> AIBotResponse:
        """Perform a Brave search with the given query.
        
        Args:
            query: The search query
            count: Number of results to return (default: 10)
            **kwargs: Additional search parameters
            
        Returns:
            AIBotResponse with search results
        """
        if not self.client:
            return AIBotResponse(message="Error: Brave Search client not initialized")
        
        try:
            results = await self.mcp_server.search(query, count)
            return AIBotResponse(
                message=f"Found {len(results)} results for query: {query}",
                data=results
            )
        except Exception as e:
            return AIBotResponse(message=f"Brave Search error: {str(e)}")


class FileSystemMCPServer(SubagentMCPServer):
    """MCP server for file system operations."""
    
    def __init__(self):
        super().__init__("filesystem")
    
    async def list_files(self, directory: str) -> List[str]:
        """List files in a directory."""
        # This would use os.listdir or similar
        return [f"{directory}/file1.txt", f"{directory}/file2.txt"]
    
    async def read_file(self, path: str) -> str:
        """Read a file's contents."""
        # This would use open() or similar
        return f"Contents of {path}"
    
    async def write_file(self, path: str, content: str) -> bool:
        """Write content to a file."""
        # This would use open() with write mode
        return True
    
    async def delete_file(self, path: str) -> bool:
        """Delete a file."""
        # This would use os.remove or similar
        return True


class FileSystemSubagent(AIModel):
    """Subagent specialized in interacting with the local file system.
    Use this subagent for reading, writing, or listing files
    in the local environment.
    """

    class Config(AIConfig):
        pass
    
    def __init__(self):
        super().__init__()
        self.mcp_server = FileSystemMCPServer()
        self.client = None
    
    async def __aenter__(self):
        """Enter the async context, starting the MCP server."""
        await self.mcp_server.__aenter__()
        self.client = MCPClient(server_name=self.mcp_server.name)
        await self.client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, stopping the MCP server."""
        if self.client:
            await self.client.disconnect()
        await self.mcp_server.__aexit__(exc_type, exc_val, exc_tb)

    async def run(self, operation: str, path: str, **kwargs) -> AIBotResponse:
        """Perform file system operations like read, write, or delete.
        
        Args:
            operation: The operation to perform (list, read, write, delete)
            path: The file or directory path
            **kwargs: Additional parameters (e.g., content for write operation)
            
        Returns:
            AIBotResponse with the operation result
        """
        if not self.client:
            return AIBotResponse(message="Error: File system client not initialized")
        
        try:
            if operation == "list":
                files = await self.mcp_server.list_files(path)
                return AIBotResponse(message=f"Listed {len(files)} files in {path}", data=files)
            
            elif operation == "read":
                content = await self.mcp_server.read_file(path)
                return AIBotResponse(message=f"Read file: {path}", data=content)
            
            elif operation == "write":
                content = kwargs.get("content", "")
                success = await self.mcp_server.write_file(path, content)
                return AIBotResponse(message=f"{'Successfully wrote' if success else 'Failed to write'} to {path}")
            
            elif operation == "delete":
                success = await self.mcp_server.delete_file(path)
                return AIBotResponse(message=f"{'Successfully deleted' if success else 'Failed to delete'} {path}")
            
            else:
                return AIBotResponse(message=f"Invalid file system operation: {operation}")
        
        except Exception as e:
            return AIBotResponse(message=f"File system operation error: {str(e)}")



class GitHubMCPServer(SubagentMCPServer):
    """MCP server for GitHub operations."""
    
    def __init__(self):
        super().__init__("github", "GITHUB_TOKEN")
    
    async def list_repos(self) -> List[Dict[str, Any]]:
        """List repositories the user has access to."""
        # This would use GitHub API
        return [{"name": "example-repo", "url": "https://github.com/user/example-repo"}]
    
    async def create_issue(self, repo: str, title: str, body: str) -> Dict[str, Any]:
        """Create an issue in a repository."""
        return {"number": 1, "title": title, "body": body, "url": f"https://github.com/user/{repo}/issues/1"}
    
    async def create_pull_request(self, repo: str, title: str, head: str, base: str, body: str) -> Dict[str, Any]:
        """Create a pull request."""
        return {
            "number": 1,
            "title": title,
            "body": body,
            "url": f"https://github.com/user/{repo}/pull/1"
        }


class GitHubSubagent(AIModel):
    """Subagent specialized in interacting with GitHub.
    Use this subagent to manage repos, create issues, pull requests, etc.

    Expects:
        GITHUB_TOKEN in environment variables.

    Will connect to an MCP server providing GitHub actions,
    but does NOT start the server automatically.
    """

    class Config(AIConfig):
        pass
    
    def __init__(self):
        super().__init__()
        self.mcp_server = GitHubMCPServer()
        self.client = None
    
    async def __aenter__(self):
        """Enter the async context, starting the MCP server."""
        await self.mcp_server.__aenter__()
        self.client = MCPClient(server_name=self.mcp_server.name)
        await self.client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, stopping the MCP server."""
        if self.client:
            await self.client.disconnect()
        await self.mcp_server.__aexit__(exc_type, exc_val, exc_tb)

    async def run(self, operation: str, **kwargs) -> AIBotResponse:
        """Perform a GitHub-related operation.
        
        Args:
            operation: The operation to perform (list_repos, create_issue, create_pr)
            **kwargs: Additional parameters specific to the operation
            
        Returns:
            AIBotResponse with the operation result
        """
        if not self.client:
            return AIBotResponse(message="Error: GitHub client not initialized")
        
        try:
            if operation == "list_repos":
                repos = await self.mcp_server.list_repos()
                return AIBotResponse(message=f"Found {len(repos)} repositories", data=repos)
            
            elif operation == "create_issue":
                repo = kwargs.get("repo")
                title = kwargs.get("title")
                body = kwargs.get("body", "")
                
                if not repo or not title:
                    return AIBotResponse(message="Error: Missing required parameters for create_issue")
                
                issue = await self.mcp_server.create_issue(repo, title, body)
                return AIBotResponse(message=f"Created issue #{issue['number']}: {title}", data=issue)
            
            elif operation == "create_pr":
                repo = kwargs.get("repo")
                title = kwargs.get("title")
                head = kwargs.get("head")
                base = kwargs.get("base", "main")
                body = kwargs.get("body", "")
                
                if not repo or not title or not head:
                    return AIBotResponse(message="Error: Missing required parameters for create_pr")
                
                pr = await self.mcp_server.create_pull_request(repo, title, head, base, body)
                return AIBotResponse(message=f"Created PR #{pr['number']}: {title}", data=pr)
            
            else:
                return AIBotResponse(message=f"Invalid GitHub operation: {operation}")
        
        except Exception as e:
            return AIBotResponse(message=f"GitHub operation error: {str(e)}")


class SlackMCPServer(SubagentMCPServer):
    """MCP server for Slack operations."""
    
    def __init__(self):
        super().__init__("slack", "SLACK_BOT_TOKEN")
    
    async def list_channels(self) -> List[Dict[str, Any]]:
        """List available Slack channels."""
        # This would use Slack API
        return [
            {"id": "C12345", "name": "general"},
            {"id": "C67890", "name": "random"}
        ]
    
    async def send_message(self, channel: str, text: str) -> Dict[str, Any]:
        """Send a message to a Slack channel."""
        return {"channel": channel, "ts": "1234567890.123456", "text": text}
    
    async def get_messages(self, channel: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from a channel."""
        return [
            {"user": "U12345", "text": "Example message 1", "ts": "1234567890.123456"},
            {"user": "U67890", "text": "Example message 2", "ts": "1234567890.123457"}
        ]


class SlackSubagent(AIModel):
    """Subagent specialized in sending messages to Slack or performing Slack actions.
    Use this subagent to send messages, read channels, or manage Slack workspace tasks.

    Expects:
        SLACK_BOT_TOKEN in environment variables.

    Will connect to an MCP server providing Slack actions,
    but does NOT start the server automatically.
    """

    class Config(AIConfig):
        pass
    
    def __init__(self):
        super().__init__()
        self.mcp_server = SlackMCPServer()
        self.client = None
    
    async def __aenter__(self):
        """Enter the async context, starting the MCP server."""
        await self.mcp_server.__aenter__()
        self.client = MCPClient(server_name=self.mcp_server.name)
        await self.client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, stopping the MCP server."""
        if self.client:
            await self.client.disconnect()
        await self.mcp_server.__aexit__(exc_type, exc_val, exc_tb)

    async def run(self, operation: str, **kwargs) -> AIBotResponse:
        """Perform an operation on Slack.
        
        Args:
            operation: The operation to perform (list_channels, send_message, get_messages)
            **kwargs: Additional parameters specific to the operation
            
        Returns:
            AIBotResponse with the operation result
        """
        if not self.client:
            return AIBotResponse(message="Error: Slack client not initialized")
        
        try:
            if operation == "list_channels":
                channels = await self.mcp_server.list_channels()
                return AIBotResponse(message=f"Found {len(channels)} Slack channels", data=channels)
            
            elif operation == "send_message":
                channel = kwargs.get("channel")
                text = kwargs.get("text")
                
                if not channel or not text:
                    return AIBotResponse(message="Error: Missing required parameters for send_message")
                
                result = await self.mcp_server.send_message(channel, text)
                return AIBotResponse(message=f"Sent message to Slack channel '{channel}'", data=result)
            
            elif operation == "get_messages":
                channel = kwargs.get("channel")
                limit = kwargs.get("limit", 10)
                
                if not channel:
                    return AIBotResponse(message="Error: Missing required channel parameter for get_messages")
                
                messages = await self.mcp_server.get_messages(channel, limit)
                return AIBotResponse(message=f"Retrieved {len(messages)} messages from channel '{channel}'", data=messages)
            
            else:
                return AIBotResponse(message=f"Invalid Slack operation: {operation}")
        
        except Exception as e:
            return AIBotResponse(message=f"Slack operation error: {str(e)}")


class FirecrawlMCPServer(SubagentMCPServer):
    """MCP server for Firecrawl security scanning operations."""
    
    def __init__(self):
        super().__init__("firecrawl", "FIRECRAWL_API_KEY")
    
    async def scan_website(self, url: str) -> Dict[str, Any]:
        """Scan a website for security vulnerabilities."""
        # This would use Firecrawl API
        return {
            "scan_id": "scan_12345",
            "url": url,
            "status": "completed",
            "vulnerabilities": [
                {"severity": "medium", "type": "XSS", "location": "/page.php?id=1"}
            ]
        }
    
    async def get_scan_results(self, scan_id: str) -> Dict[str, Any]:
        """Get results of a previous scan."""
        return {
            "scan_id": scan_id,
            "status": "completed",
            "vulnerabilities": [
                {"severity": "medium", "type": "XSS", "location": "/page.php?id=1"}
            ]
        }


class FirecrawlSubagent(AIModel):
    """Subagent specialized in Firecrawl operations (a hypothetical security scanning service).
    Use this subagent to run scans, retrieve scan results, etc.

    Expects:
        FIRECRAWL_API_KEY in environment variables.

    Will connect to an MCP server providing Firecrawl actions,
    but does NOT start the server automatically.
    """

    class Config(AIConfig):
        pass
    
    def __init__(self):
        super().__init__()
        self.mcp_server = FirecrawlMCPServer()
        self.client = None
    
    async def __aenter__(self):
        """Enter the async context, starting the MCP server."""
        await self.mcp_server.__aenter__()
        self.client = MCPClient(server_name=self.mcp_server.name)
        await self.client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, stopping the MCP server."""
        if self.client:
            await self.client.disconnect()
        await self.mcp_server.__aexit__(exc_type, exc_val, exc_tb)

    async def run(self, operation: str, **kwargs) -> AIBotResponse:
        """Perform a Firecrawl security scan or related operation.
        
        Args:
            operation: The operation to perform (scan_website, get_scan_results)
            **kwargs: Additional parameters specific to the operation
            
        Returns:
            AIBotResponse with the operation result
        """
        if not self.client:
            return AIBotResponse(message="Error: Firecrawl client not initialized")
        
        try:
            if operation == "scan_website":
                url = kwargs.get("url")
                
                if not url:
                    return AIBotResponse(message="Error: Missing required URL parameter for scan_website")
                
                scan_result = await self.mcp_server.scan_website(url)
                vulnerabilities = scan_result.get("vulnerabilities", [])
                
                return AIBotResponse(
                    message=f"Completed security scan of {url}. Found {len(vulnerabilities)} vulnerabilities.",
                    data=scan_result
                )
            
            elif operation == "get_scan_results":
                scan_id = kwargs.get("scan_id")
                
                if not scan_id:
                    return AIBotResponse(message="Error: Missing required scan_id parameter for get_scan_results")
                
                results = await self.mcp_server.get_scan_results(scan_id)
                vulnerabilities = results.get("vulnerabilities", [])
                
                return AIBotResponse(
                    message=f"Retrieved results for scan {scan_id}. Found {len(vulnerabilities)} vulnerabilities.",
                    data=results
                )
            
            else:
                return AIBotResponse(message=f"Invalid Firecrawl operation: {operation}")
        
        except Exception as e:
            return AIBotResponse(message=f"Firecrawl operation error: {str(e)}")


class PrimaryAgent(AIModel):
    """Primary agent orchestrating the subagents to handle specialized tasks.

    Use the 'start_subagents' method once when the agent is created to initialize
    the MCP servers for each subagent with an async exit stack. This avoids the overhead
    of starting them fresh for every single tool call.

    Tools in this agent delegate tasks to the appropriate subagent:
    - SlackSubagent for sending Slack messages
    - BraveSearchSubagent for web search
    - AirtableSubagent for Airtable DB ops
    - GitHubSubagent for GitHub repo operations
    - FirecrawlSubagent for security scans
    - FileSystemSubagent for local FS operations
    """

    class Config(AIConfig):
        pass

    def __init__(self):
        super().__init__()
        # These attributes will be assigned inside start_subagents.
        self.stack: Optional[AsyncExitStack] = None
        self.airtable_agent: Optional[AirtableSubagent] = None
        self.brave_agent: Optional[BraveSearchSubagent] = None
        self.fs_agent: Optional[FileSystemSubagent] = None
        self.github_agent: Optional[GitHubSubagent] = None
        self.slack_agent: Optional[SlackSubagent] = None
        self.firecrawl_agent: Optional[FirecrawlSubagent] = None

    async def start_subagents(self):
        """Start all subagents in an async context stack. This should be run only once,
        as starting servers repeatedly is slow.
        """
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()

        self.airtable_agent = await self.stack.enter_async_context(AirtableSubagent())
        self.brave_agent = await self.stack.enter_async_context(BraveSearchSubagent())
        self.fs_agent = await self.stack.enter_async_context(FileSystemSubagent())
        self.github_agent = await self.stack.enter_async_context(GitHubSubagent())
        self.slack_agent = await self.stack.enter_async_context(SlackSubagent())
        self.firecrawl_agent = await self.stack.enter_async_context(FirecrawlSubagent())

    async def stop_subagents(self):
        """Stop all subagents by exiting the async context stack."""
        if self.stack:
            await self.stack.__aexit__(None, None, None)
            self.stack = None

    @Tool(
        name="use_subagent",
        description=(
            "Call an appropriate subagent by name to perform specialized tasks. "
            "Slack subagent for sending messages, Brave subagent for web search, "
            "Airtable subagent for DB ops, GitHub subagent for repos, "
            "Firecrawl subagent for security scans, FS subagent for local files."
        )
    )
    async def use_subagent(self, subagent_name: str, **kwargs) -> str:
        """
        Call a subagent to handle a specialized task.

        Args:
            subagent_name: The name of the subagent to call. One of:
                "slack", "brave", "airtable", "github", "firecrawl", "filesystem"
            kwargs: Additional arguments passed to the subagent's run() method.

        Returns:
            A string summarizing the subagent's operation result.
        """
        if subagent_name == "slack" and self.slack_agent is not None:
            response = await self.slack_agent.run(**kwargs)
        elif subagent_name == "brave" and self.brave_agent is not None:
            response = await self.brave_agent.run(**kwargs)
        elif subagent_name == "airtable" and self.airtable_agent is not None:
            response = await self.airtable_agent.run(**kwargs)
        elif subagent_name == "github" and self.github_agent is not None:
            response = await self.github_agent.run(**kwargs)
        elif subagent_name == "firecrawl" and self.firecrawl_agent is not None:
            response = await self.firecrawl_agent.run(**kwargs)
        elif subagent_name in ("fs", "filesystem") and self.fs_agent is not None:
            response = await self.fs_agent.run(**kwargs)
        else:
            raise ValueError(f"Unknown or uninitialized subagent: {subagent_name}")
        return response.message