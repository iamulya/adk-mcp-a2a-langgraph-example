import os
import logging
import asyncio
from contextlib import AsyncExitStack
from typing import Dict, Optional

from google.adk.tools.mcp_tool import MCPToolset, MCPTool
# Choose the correct connection params based on your MCP server type
from mcp import StdioServerParameters, SseServerParams

logger = logging.getLogger(__name__)

# Configure MCP connection based on environment variables
mcp_url_or_command = os.getenv("YOUTUBE_MCP_SERVER_URL")
if not mcp_url_or_command:
    raise ValueError("YOUTUBE_MCP_SERVER_URL environment variable not set.")

# --- CHOOSE ONE Connection Type ---
# Option 1: StdioServerParameters (if MCP server is a command-line tool)
# Adjust command/args as needed. Assumes the env var is the command itself.
connection_params = StdioServerParameters(command=mcp_url_or_command)

# Option 2: SseServerParams (if MCP server is running at a URL endpoint)
# connection_params = SseServerParams(url=mcp_url_or_command)
# ---------------------------------

# Global ExitStack and Toolset instance (lazy initialized)
_exit_stack: Optional[AsyncExitStack] = None
_mcp_toolset: Optional[MCPToolset] = None
_tools: Optional[Dict[str, MCPTool]] = None

async def get_mcp_tools() -> Dict[str, MCPTool]:
    """Initializes and returns the MCP tools, managing the session."""
    global _exit_stack, _mcp_toolset, _tools
    if _tools is None:
        logger.info(f"Initializing MCPToolset for YouTube Agent at {mcp_url_or_command}")
        _exit_stack = AsyncExitStack()
        _mcp_toolset = MCPToolset(connection_params=connection_params, exit_stack=_exit_stack)
        # The context manager ensures the session is initialized
        await _exit_stack.enter_async_context(_mcp_toolset)
        _tools = {tool.name: tool for tool in await _mcp_toolset.load_tools()}
        logger.info(f"Loaded MCP tools: {list(_tools.keys())}")

         # --- Register cleanup ---
        import atexit
        # Ensure cleanup runs in an event loop if called from a sync context like atexit
        if asyncio.get_event_loop().is_running():
             atexit.register(lambda: asyncio.ensure_future(_cleanup_mcp_session()))
        else:
             atexit.register(asyncio.run, _cleanup_mcp_session())


    return _tools

async def _cleanup_mcp_session():
    """Cleans up the MCP session on exit."""
    global _exit_stack, _tools
    if _exit_stack:
        logger.info("Cleaning up YouTube Agent MCP session...")
        await _exit_stack.aclose()
        _exit_stack = None
        _tools = None # Reset tools as well
        logger.info("YouTube Agent MCP session closed.")

async def get_channel_videos(channel_id: str, date: str) -> list[str]:
    """Gets video IDs from a YouTube channel for a specific date via MCP."""
    tools = await get_mcp_tools()
    # *** Replace with the EXACT name provided by your MCP server ***
    tool_name = "get_channel_videos_by_date"
    if tool_name not in tools:
        raise ValueError(f"MCP Tool '{tool_name}' not found. Available: {list(tools.keys())}")
    logger.info(f"Calling MCP tool '{tool_name}' with channel_id={channel_id}, date={date}")
    try:
        # Pass arguments matching the MCP tool's expected input schema
        result = await tools[tool_name].run_async(args={"channel_id": channel_id, "date": date}, tool_context=None)
        # Ensure result is a list of strings
        if isinstance(result, list):
            return [str(item) for item in result]
        elif result is None:
            return []
        else:
            logger.warning(f"MCP tool '{tool_name}' returned non-list: {result}. Wrapping in list.")
            return [str(result)]
    except Exception as e:
        logger.error(f"Error calling MCP tool '{tool_name}': {e}", exc_info=True)
        raise # Re-raise to be caught by the sync wrapper

async def get_playlist_videos(playlist_id: str) -> list[str]:
    """Gets video IDs from a YouTube playlist via MCP."""
    tools = await get_mcp_tools()
     # *** Replace with the EXACT name provided by your MCP server ***
    tool_name = "get_playlist_videos"
    if tool_name not in tools:
        raise ValueError(f"MCP Tool '{tool_name}' not found. Available: {list(tools.keys())}")
    logger.info(f"Calling MCP tool '{tool_name}' with playlist_id={playlist_id}")
    try:
        # Pass arguments matching the MCP tool's expected input schema
        result = await tools[tool_name].run_async(args={"playlist_id": playlist_id}, tool_context=None)
        # Ensure result is a list of strings
        if isinstance(result, list):
            return [str(item) for item in result]
        elif result is None:
             return []
        else:
            logger.warning(f"MCP tool '{tool_name}' returned non-list: {result}. Wrapping in list.")
            return [str(result)]
    except Exception as e:
        logger.error(f"Error calling MCP tool '{tool_name}': {e}", exc_info=True)
        raise # Re-raise to be caught by the sync wrapper

# Note: We wrap these async functions in LangChain Tools within agent.py
