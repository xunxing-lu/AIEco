import sys
import os

# Add the AIEco directory to Python path
aieco_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, aieco_dir)

print(f"Added to path: {aieco_dir}")

# Check the content of sync_mcp_manager.py
mcp_path = os.path.join(aieco_dir, 'mcp')
sync_file = os.path.join(mcp_path, 'sync_mcp_manager.py')

print(f"sync_mcp_manager.py path: {sync_file}")
print(f"File exists: {os.path.exists(sync_file)}")

if os.path.exists(sync_file):
    with open(sync_file, 'r') as f:
        content = f.read()
        print(f"File size: {len(content)} characters")
        print("First 200 characters:")
        print(content[:200])
        print("...")
        
        # Check if SyncMCPManager class is defined
        if 'class SyncMCPManager' in content:
            print("✓ SyncMCPManager class found in file")
        else:
            print("✗ SyncMCPManager class NOT found in file")

# Try the import
try:
    from mcp.sync_mcp_manager import SyncMCPManager
    print("✓ Import successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
except Exception as e:
    print(f"✗ Other error: {e}")