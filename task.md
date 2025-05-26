# Supabase MCP Server - Initial Tasks

This document lists the initial tasks to get the Supabase MCP server project started.

## Phase 1: Project Setup & Core MCP Implementation

1.  **Initialize Project Structure:**
    *   Create main project directory (e.g., `supabase_mcp_server`).
    *   Set up a Python virtual environment (`python -m venv .venv`).
    *   Create `requirements.txt` file.
    *   Create main server script (e.g., `server.py`).
    *   Create a `.gitignore` file.

2.  **Install Dependencies:**
    *   Add `supabase-py` to `requirements.txt`.
    *   Add any libraries needed for MCP communication (if not built from scratch).
    *   Install dependencies: `pip install -r requirements.txt`.

3.  **Implement Configuration:**
    *   Create a `.env.example` file with `SUPABASE_URL` and `SUPABASE_KEY`.
    *   Add code to load these variables (e.g., using `python-dotenv`) in `server.py`.

4.  **Basic MCP Server Loop (Stdio):**
    *   Research and implement the basic structure for an MCP server communicating over standard input/output (stdin/stdout).
    *   This involves:
        *   Reading JSON requests from stdin line by line.
        *   Parsing the requests.
        *   Sending JSON responses to stdout.
        *   Handling potential errors during parsing or execution.

5.  **Define MCP Manifest (`.roo/mcp.json`):**
    *   Create the initial `.roo/mcp.json` file.
    *   Define the server name and description.
    *   Add the definition for the first tool (`supabase_select_data`).

## Phase 2: First Supabase Tool

6.  **Implement `supabase_select_data` Tool:**
    *   Create a function or class to handle the `supabase_select_data` tool logic.
    *   Initialize the `supabase-py` client using the loaded configuration.
    *   Parse arguments from the MCP tool request (`table_name`, `columns`, `filters`).
    *   Use the `supabase-py` client to execute the `select` query.
    *   Format the result (or error) into the MCP response format.
    *   Integrate this logic into the main server loop/dispatcher.

7.  **Basic Testing:**
    *   Manually craft an MCP request JSON for `supabase_select_data`.
    *   Run the `server.py` script.
    *   Pipe the request JSON to the server's stdin.
    *   Verify the MCP response JSON printed to stdout is correct and contains the expected data from Supabase.

## Next Steps After Initial Tasks

*   Implement remaining CRUD tools (`insert`, `update`, `delete`).
*   Implement resource access.
*   Add error handling and logging.
*   Refine testing.