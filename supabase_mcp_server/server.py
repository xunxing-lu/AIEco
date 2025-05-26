import os
import sys
import json
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# Get Supabase credentials from environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Basic check to ensure variables are loaded
if not supabase_url or not supabase_key:
    print("Error: SUPABASE_URL and SUPABASE_KEY must be set in the .env file.", file=sys.stderr)
    sys.exit(1)

# Initialize Supabase client
try:
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Supabase client initialized successfully.")
except Exception as e:
    print(f"Error initializing Supabase client: {e}", file=sys.stderr)
    sys.exit(1)

print("MCP Server starting...")
# print(f"Supabase URL: {supabase_url}") # Can be removed now

# --- Tool Implementation ---
def handle_supabase_select(args):
    """Handles the supabase_select_data tool request."""
    try:
        table_name = args.get("table_name")
        if not table_name:
            return {"error": {"code": -32602, "message": "Invalid params: 'table_name' is required."}}

        columns = args.get("columns", "*") # Default to '*' if not provided
        filters = args.get("filters", {})

        query = supabase.table(table_name).select(columns)

        # Apply simple equality filters
        for key, value in filters.items():
            query = query.eq(key, value)

        data = query.execute()

        # Check for Supabase API errors if the library wraps them nicely,
        # otherwise rely on potential exceptions during execute()
        # For supabase-py v1/v2, data object contains the result
        return {"result": data.data} # Assuming v2 structure where result is in data.data

    except Exception as e:
        # Catch potential errors during Supabase interaction
        print(f"Error during Supabase select: {e}", file=sys.stderr)
        return {"error": {"code": -32000, "message": f"Supabase query failed: {str(e)}"}}

# --- Resource Implementation (Placeholder) ---
# def handle_resource_access(uri):
#     pass

# --- MCP Server Loop (To be implemented in Task 4) ---

# Main MCP Server Loop
while True:
    try:
        line = sys.stdin.readline()
        if not line:
            # End of input stream, exit gracefully
            break

        # Attempt to parse the incoming JSON request
        request = json.loads(line)

        # --- Request Processing Logic ---
        request_id = request.get("id")
        response = {"id": request_id} # Always include ID in response if present in request

        if request.get("method") == "use_tool" and request.get("params", {}).get("name") == "supabase_select_data":
            print(f"Processing tool request [ID: {request_id}]: supabase_select_data", file=sys.stderr)
            tool_args = request.get("params", {}).get("arguments", {})
            tool_result = handle_supabase_select(tool_args)
            if "error" in tool_result:
                response["error"] = tool_result["error"]
            else:
                response["result"] = tool_result["result"]
        # TODO: Add elif blocks for other tools and resource access
        else:
            # Handle unknown method or tool
            print(f"Received unknown request [ID: {request_id}]: {request.get('method')}", file=sys.stderr)
            response["error"] = {"code": -32601, "message": "Method not found or tool not supported."}
        # --- End Processing Logic ---

        # Send the JSON response to stdout
        print(json.dumps(response), flush=True)

    except json.JSONDecodeError:
        # Handle invalid JSON
        # Handle invalid JSON - Ensure response includes ID if possible (though unlikely here)
        error_response = {"error": {"code": -32700, "message": "Parse error: Invalid JSON received."}, "id": None}
        try:
            # Attempt to extract ID even from invalid JSON line if simple enough
            potential_id = json.loads(line).get("id")
            error_response["id"] = potential_id
        except:
            pass # Ignore if ID extraction fails
        print(json.dumps(error_response), file=sys.stderr, flush=True)

    except Exception as e:
        # Handle other unexpected errors during processing - Ensure response includes ID if possible
        error_response = {"error": {"code": -32603, "message": f"Internal server error: {str(e)}"}, "id": None}
        try:
            # Attempt to extract ID from the parsed request if available
            error_response["id"] = request.get("id")
        except NameError: # request might not be defined if error happened before parsing
             pass
        except Exception: # Catch any other issue getting the ID
            pass
        print(json.dumps(error_response), file=sys.stderr, flush=True)

print("MCP Server stopped.")