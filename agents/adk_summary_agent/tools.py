import os
import json
import asyncio
import logging
from typing import Any, Dict, Optional, List
from contextlib import AsyncExitStack

# Assuming 'common' library is available via install or copied path
try:
    from common.client import A2AClient
    from common.types import TaskSendParams, TextPart, Message, TaskState, DataPart
except ImportError:
    raise ImportError("Could not import A2A common library. Ensure 'a2a-samples' is installed or 'common' directory is in PYTHONPATH.")

from google.adk.tools.mcp_tool import MCPTool
# Choose the correct connection params based on your MCP server type
from mcp import StdioServerParameters, SseServerParams, types as mcp_types
from ..utils.secrets import get_secret # Assuming utils is importable from root

logger = logging.getLogger(__name__)

# --- MCP Session Management ---
_summary_mcp_sessions: Dict[str, asyncio.Queue] = {} # URL -> Queue[Session]
_summary_mcp_stacks: Dict[str, AsyncExitStack] = {} # URL -> ExitStack
_summary_mcp_session_locks: Dict[str, asyncio.Lock] = {} # URL -> Lock

async def get_summary_mcp_session(mcp_url_or_command: str) -> mcp_types.ClientSession:
    """Gets or creates a managed MCP session for a given summarization server URL/command."""
    if mcp_url_or_command not in _summary_mcp_session_locks:
        _summary_mcp_session_locks[mcp_url_or_command] = asyncio.Lock()

    async with _summary_mcp_session_locks[mcp_url_or_command]:
        if mcp_url_or_command not in _summary_mcp_sessions:
            logger.info(f"Creating new MCP session queue for {mcp_url_or_command}")
            _summary_mcp_sessions[mcp_url_or_command] = asyncio.Queue(maxsize=1)
            _summary_mcp_stacks[mcp_url_or_command] = AsyncExitStack()

            # --- CHOOSE Connection Type Based on URL/Command ---
            if mcp_url_or_command.startswith("http"):
                connection_params = SseServerParams(url=mcp_url_or_command)
                logger.info("Using SseServerParams for summary tool")
            else:
                connection_params = StdioServerParameters(command=mcp_url_or_command)
                logger.info("Using StdioServerParameters for summary tool")
            # ---------------------------------------------------

            try:
                from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
                session_manager = MCPSessionManager(
                    connection_params=connection_params,
                    exit_stack=_summary_mcp_stacks[mcp_url_or_command]
                )
                session = await session_manager.create_session()
                await _summary_mcp_sessions[mcp_url_or_command].put(session)
                logger.info(f"MCP session initialized for {mcp_url_or_command}")

                import atexit
                if asyncio.get_event_loop().is_running():
                    atexit.register(lambda: asyncio.ensure_future(_cleanup_summary_mcp_session(mcp_url_or_command)))
                else:
                    atexit.register(asyncio.run, _cleanup_summary_mcp_session(mcp_url_or_command))

            except Exception as e:
                logger.error(f"Failed to initialize MCP session for {mcp_url_or_command}: {e}", exc_info=True)
                if mcp_url_or_command in _summary_mcp_stacks:
                    await _summary_mcp_stacks[mcp_url_or_command].aclose()
                    del _summary_mcp_stacks[mcp_url_or_command]
                if mcp_url_or_command in _summary_mcp_sessions:
                    del _summary_mcp_sessions[mcp_url_or_command]
                raise

        session = await _summary_mcp_sessions[mcp_url_or_command].get()
        return session

async def release_summary_mcp_session(mcp_url_or_command: str, session: mcp_types.ClientSession):
    """Releases a summary session back to the pool."""
    if mcp_url_or_command in _summary_mcp_sessions:
        await _summary_mcp_sessions[mcp_url_or_command].put(session)

async def _cleanup_summary_mcp_session(mcp_url_or_command: str):
    """Cleans up a specific Summary MCP session stack."""
    if mcp_url_or_command in _summary_mcp_stacks:
        logger.info(f"Cleaning up Summary MCP session for {mcp_url_or_command}...")
        stack = _summary_mcp_stacks.pop(mcp_url_or_command)
        await stack.aclose()
        if mcp_url_or_command in _summary_mcp_sessions:
            del _summary_mcp_sessions[mcp_url_or_command]
        if mcp_url_or_command in _summary_mcp_session_locks:
             del _summary_mcp_session_locks[mcp_url_or_command]
        logger.info(f"Summary MCP session closed for {mcp_url_or_command}.")

# --- MCP Tool Functions ---

async def summarize_video(video_url: str) -> str:
    """Creates a summary for a single YouTube video via MCP."""
    mcp_endpoint = os.getenv("MCP_URL_SUMMARIZE")
    if not mcp_endpoint:
        raise ValueError("MCP_URL_SUMMARIZE environment variable not set.")

    session = await get_summary_mcp_session(mcp_endpoint)
    try:
        # *** Tool name must match the name provided by the MCP server ***
        tool_name = "get_youtube_video_summary"
        logger.info(f"Calling MCP tool '{tool_name}' on {mcp_endpoint} for URL {video_url}")
        # Arguments must match the MCP tool's input schema
        result = await session.call_tool(tool_name, arguments={"video_url": video_url})
        logger.debug(f"MCP Raw result for {tool_name}: {result}")

        # --- Parse the specific response structure ---
        if isinstance(result, dict) and "summary" in result:
            return str(result["summary"])
        elif isinstance(result, dict) and "error" in result:
            logger.error(f"MCP tool '{tool_name}' returned error: {result['error']}")
            return f"Error from summarize_video: {result['error']}"
        else:
            logger.warning(f"Unexpected result format from MCP tool '{tool_name}': {result}")
            return f"Error: Unexpected result format from {tool_name}"
        # ------------------------------------------
    except Exception as e:
        logger.error(f"Error calling MCP tool '{tool_name}': {e}", exc_info=True)
        return f"Error contacting summarization service: {type(e).__name__}"
    finally:
        await release_summary_mcp_session(mcp_endpoint, session)


async def combine_summaries(summaries: list[str]) -> str:
    """Combines multiple video summaries into a final summary via MCP."""
    mcp_endpoint = os.getenv("MCP_URL_COMBINE")
    if not mcp_endpoint:
        raise ValueError("MCP_URL_COMBINE environment variable not set.")

    session = await get_summary_mcp_session(mcp_endpoint)
    try:
        # *** Tool name must match the name provided by the MCP server ***
        tool_name = "generate_final_summary"
        logger.info(f"Calling MCP tool '{tool_name}' on {mcp_endpoint} with {len(summaries)} summaries")
        # Arguments must match the MCP tool's input schema (e.g., expecting {"summaries": [...]})
        result = await session.call_tool(tool_name, arguments={"summaries": summaries})
        logger.debug(f"MCP Raw result for {tool_name}: {result}")

        # --- Parse the specific response structure ---
        # Assuming the response is a dict containing the summary under 'final_summary'
        # Adjust the key based on your actual MCP tool response schema.
        summary_key = "final_summary" # Example key, replace if needed
        if isinstance(result, dict) and summary_key in result:
            return str(result[summary_key])
        elif isinstance(result, dict) and "error" in result:
             logger.error(f"MCP tool '{tool_name}' returned error: {result['error']}")
             return f"Error from combine_summaries: {result['error']}"
        elif isinstance(result, str): # If tool returns just the string directly
             return result
        elif result == {}: # Handle empty dict response if that indicates success but no text
             logger.warning(f"MCP tool '{tool_name}' returned empty dict.")
             return "Successfully combined summaries (no text returned)."
        else:
            logger.warning(f"Unexpected result format from MCP tool '{tool_name}': {result}")
            return f"Error: Unexpected result format from {tool_name}"
        # ------------------------------------------
    except Exception as e:
        logger.error(f"Error calling MCP tool '{tool_name}': {e}", exc_info=True)
        return f"Error contacting final summary service: {type(e).__name__}"
    finally:
        await release_summary_mcp_session(mcp_endpoint, session)

# --- A2A Delegation Tool ---
async def find_videos_via_a2a(
    youtube_agent_url: str,
    query: str,
    session_id: str,
    tool_context: Any = None # ADK Tool context
) -> list[str]:
    """
    Delegates finding YouTube videos to a remote YouTube agent via A2A.
    Returns a list of video IDs or a list containing a single error string.
    """
    logger.info(f"Initiating A2A call to {youtube_agent_url} with query: '{query}'")
    try:
        a2a_client = A2AClient(url=youtube_agent_url)
        task_params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text=query)]),
            sessionId=session_id,
            acceptedOutputModes=["application/json"] # Prefer JSON structured data
        )
        response = await a2a_client.send_task(task_params.model_dump(exclude_defaults=True))
        logger.info(f"A2A response received: Status={response.result.status.state}")

        if response.result.status.state == TaskState.COMPLETED and response.result.artifacts:
            for artifact in response.result.artifacts:
                for part in artifact.parts:
                    # Prioritize DataPart
                    if part.type == "data" and isinstance(part.data, list):
                         # Ensure all items are strings
                        video_ids = [str(item) for item in part.data]
                        logger.info(f"Extracted video IDs via DataPart: {video_ids}")
                        # Check if it's an error message list
                        if video_ids and video_ids[0].startswith("Error:"):
                             logger.warning(f"YouTube agent returned error via DataPart: {video_ids[0]}")
                             return video_ids # Propagate the error list
                        return video_ids
                    # Fallback to TextPart (expecting JSON string list)
                    elif part.type == "text":
                        try:
                            data = json.loads(part.text)
                            if isinstance(data, list):
                                video_ids = [str(item) for item in data]
                                logger.info(f"Extracted video IDs via TextPart: {video_ids}")
                                # Check if it's an error message list
                                if video_ids and video_ids[0].startswith("Error:"):
                                    logger.warning(f"YouTube agent returned error via TextPart: {video_ids[0]}")
                                    return video_ids # Propagate the error list
                                return video_ids
                        except json.JSONDecodeError:
                            logger.warning(f"A2A Text artifact not JSON: {part.text}")
                            if part.text.startswith("Error:"):
                                return [part.text] # Return error if text starts with Error:
                            continue # Ignore non-JSON text if not an error

            logger.warning("A2A task completed but no parsable video list artifact found.")
            return ["Error: YouTube agent returned no valid video list."]

        elif response.result.status.state == TaskState.FAILED:
             error_msg = f"Error: YouTube agent task failed: {response.result.status.message.parts[0].text if response.result.status.message else 'Unknown reason'}"
             logger.error(error_msg)
             return [error_msg]
        else:
            logger.warning(f"A2A task completed with unexpected state: {response.result.status.state}")
            return [f"Error: Unexpected response state from YouTube agent: {response.result.status.state}"]
    except Exception as e:
        logger.error(f"Error during A2A call to YouTube agent: {e}", exc_info=True)
        return [f"Error: Failed to communicate with YouTube agent - {type(e).__name__}."]

# ADK will wrap these async functions using FunctionTool
