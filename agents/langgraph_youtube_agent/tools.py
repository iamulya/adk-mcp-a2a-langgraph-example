import os
import json
import logging
import asyncio
from contextlib import AsyncExitStack
from typing import Dict, Optional, Any, List

from google.adk.tools.mcp_tool import MCPTool
from langgraph_sdk.auth.types import MinimalUserDict # Correct import if needed, might not be needed for tool calls directly
# Correct import for MCP connection parameters might depend on specific MCP client library used by ADK/MCPTool,
# but for demonstration using ADK's perspective:
from mcp import StdioServerParameters, SseServerParams, types as mcp_types

from ..utils.secrets import get_secret # Assuming utils is importable from root

logger = logging.getLogger(__name__)

# --- MCP Session Management ---
# We need potentially multiple sessions if tools are on different servers
_mcp_sessions: Dict[str, asyncio.Queue] = {} # URL -> Queue[Session]
_mcp_stacks: Dict[str, AsyncExitStack] = {} # URL -> ExitStack
_mcp_session_locks: Dict[str, asyncio.Lock] = {} # URL -> Lock

async def get_mcp_session(mcp_url_or_command: str) -> mcp_types.ClientSession:
    """Gets or creates a managed MCP session for a given server URL/command."""
    if mcp_url_or_command not in _mcp_session_locks:
        _mcp_session_locks[mcp_url_or_command] = asyncio.Lock()

    async with _mcp_session_locks[mcp_url_or_command]:
        if mcp_url_or_command not in _mcp_sessions:
            logger.info(f"Creating new MCP session queue for {mcp_url_or_command}")
            _mcp_sessions[mcp_url_or_command] = asyncio.Queue(maxsize=1) # Pool size 1 for simplicity
            _mcp_stacks[mcp_url_or_command] = AsyncExitStack()

            # --- CHOOSE Connection Type Based on URL/Command ---
            # This is a heuristic, adjust logic if needed
            if mcp_url_or_command.startswith("http"):
                connection_params = SseServerParams(url=mcp_url_or_command) # Assumes SSE if URL
                logger.info("Using SseServerParams")
            else:
                 # Assumes Stdio if not a URL, treats the env var as the command
                connection_params = StdioServerParameters(command=mcp_url_or_command)
                logger.info("Using StdioServerParameters")
            # ---------------------------------------------------

            try:
                from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
                session_manager = MCPSessionManager(
                    connection_params=connection_params,
                    exit_stack=_mcp_stacks[mcp_url_or_command]
                )
                session = await session_manager.create_session()
                await _mcp_sessions[mcp_url_or_command].put(session)
                logger.info(f"MCP session initialized for {mcp_url_or_command}")

                # Register cleanup for this specific session stack
                import atexit
                atexit.register(asyncio.run, _cleanup_mcp_session(mcp_url_or_command))

            except Exception as e:
                logger.error(f"Failed to initialize MCP session for {mcp_url_or_command}: {e}", exc_info=True)
                # Clean up partially created resources if init fails
                if mcp_url_or_command in _mcp_stacks:
                    await _mcp_stacks[mcp_url_or_command].aclose()
                    del _mcp_stacks[mcp_url_or_command]
                if mcp_url_or_command in _mcp_sessions:
                    del _mcp_sessions[mcp_url_or_command]
                raise # Re-raise the exception

        # Get a session from the queue (pool)
        session = await _mcp_sessions[mcp_url_or_command].get()
        return session

async def release_mcp_session(mcp_url_or_command: str, session: mcp_types.ClientSession):
    """Releases a session back to the pool."""
    if mcp_url_or_command in _mcp_sessions:
        await _mcp_sessions[mcp_url_or_command].put(session)

async def _cleanup_mcp_session(mcp_url_or_command: str):
    """Cleans up a specific MCP session stack."""
    if mcp_url_or_command in _mcp_stacks:
        logger.info(f"Cleaning up MCP session for {mcp_url_or_command}...")
        stack = _mcp_stacks.pop(mcp_url_or_command)
        await stack.aclose()
        if mcp_url_or_command in _mcp_sessions:
            del _mcp_sessions[mcp_url_or_command] # Remove queue
        if mcp_url_or_command in _mcp_session_locks:
            del _mcp_session_locks[mcp_url_or_command] # Remove lock
        logger.info(f"MCP session closed for {mcp_url_or_command}.")

# --- MCP Tool Functions ---

async def get_channel_videos(channel_id: str, date: str) -> list[str]:
    """Gets video URLs from a YouTube channel for a specific date via MCP."""
    mcp_endpoint = os.getenv("MCP_URL_GET_CHANNEL")
    if not mcp_endpoint:
        raise ValueError("MCP_URL_GET_CHANNEL environment variable not set.")

    session = await get_mcp_session(mcp_endpoint)
    try:
        # *** Tool name must match the name provided by the MCP server ***
        tool_name = "get_youtube_videos_for_channel_date"
        logger.info(f"Calling MCP tool '{tool_name}' on {mcp_endpoint}")
        # Arguments must match the MCP tool's input schema
        result = await session.call_tool(tool_name, arguments={"channel_id": channel_id, "date": date})
        logger.debug(f"MCP Raw result for {tool_name}: {result}")

        # --- Parse the specific response structure ---
        if isinstance(result, dict) and "video_urls" in result and isinstance(result["video_urls"], list):
            return [str(url) for url in result["video_urls"]]
        elif isinstance(result, dict) and "error" in result:
             logger.error(f"MCP tool '{tool_name}' returned error: {result['error']}")
             return [f"Error from get_channel_videos: {result['error']}"]
        elif result is None:
             return []
        else:
            logger.warning(f"Unexpected result format from MCP tool '{tool_name}': {result}")
            # Attempt to return as string list if possible, otherwise error
            try:
                return [str(result)]
            except:
                return [f"Error: Unexpected result format from {tool_name}"]
        # ------------------------------------------
    except Exception as e:
        logger.error(f"Error calling MCP tool '{tool_name}': {e}", exc_info=True)
        return [f"Error contacting video finding service: {type(e).__name__}"]
    finally:
        await release_mcp_session(mcp_endpoint, session)


async def get_playlist_videos(playlist_id: str) -> list[str]:
    """Gets video URLs from a YouTube playlist via MCP."""
    mcp_endpoint = os.getenv("MCP_URL_GET_PLAYLIST")
    if not mcp_endpoint:
        raise ValueError("MCP_URL_GET_PLAYLIST environment variable not set.")

    session = await get_mcp_session(mcp_endpoint)
    try:
        # *** Tool name must match the name provided by the MCP server ***
        tool_name = "get_playlist_videos"
        logger.info(f"Calling MCP tool '{tool_name}' on {mcp_endpoint}")
        # Arguments must match the MCP tool's input schema
        result = await session.call_tool(tool_name, arguments={"playlist_id": playlist_id})
        logger.debug(f"MCP Raw result for {tool_name}: {result}")

        # --- Parse the specific response structure ---
        # Assuming the response is a dict containing a list under 'video_urls' or similar key.
        # Adjust the key based on your actual MCP tool response schema.
        video_urls_key = "video_urls" # Example key, replace if needed
        if isinstance(result, dict) and video_urls_key in result and isinstance(result[video_urls_key], list):
            return [str(url) for url in result[video_urls_key]]
        elif isinstance(result, dict) and "error" in result:
             logger.error(f"MCP tool '{tool_name}' returned error: {result['error']}")
             return [f"Error from get_playlist_videos: {result['error']}"]
        elif result is None:
             return []
        elif isinstance(result, list): # Direct list fallback
            logger.warning(f"MCP tool '{tool_name}' returned a direct list, expected dict.")
            return [str(item) for item in result]
        else:
            logger.warning(f"Unexpected result format from MCP tool '{tool_name}': {result}")
            try:
                return [str(result)]
            except:
                 return [f"Error: Unexpected result format from {tool_name}"]
        # ------------------------------------------
    except Exception as e:
        logger.error(f"Error calling MCP tool '{tool_name}': {e}", exc_info=True)
        return [f"Error contacting playlist service: {type(e).__name__}"]
    finally:
        await release_mcp_session(mcp_endpoint, session)

# Note: These async functions are wrapped in sync LangChain Tools in agent.py
