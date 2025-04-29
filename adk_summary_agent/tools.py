import os
import json
import asyncio
import logging
from typing import Any, Dict, Optional
from contextlib import AsyncExitStack

# Assuming 'common' library is available via install or copied path
try:
    from common.client import A2AClient, A2ACardResolver
    from common.types import TaskSendParams, TextPart, Message, TaskState, DataPart
except ImportError:
    raise ImportError("Could not import A2A common library. Ensure 'a2a-samples' is installed or 'common' directory is in PYTHONPATH.")

from google.adk.tools.mcp_tool import MCPToolset, MCPTool
# Choose the correct connection params based on your MCP server type
from mcp import StdioServerParameters, SseServerParams

logger = logging.getLogger(__name__)

# --- MCP Tool Setup for Summarization ---
summary_mcp_url_or_command = os.getenv("SUMMARY_MCP_SERVER_URL")
if not summary_mcp_url_or_command:
    raise ValueError("SUMMARY_MCP_SERVER_URL environment variable not set.")

# --- CHOOSE ONE Connection Type ---
# Option 1: StdioServerParameters (if MCP server is a command-line tool)
summary_connection_params = StdioServerParameters(command=summary_mcp_url_or_command)

# Option 2: SseServerParams (if MCP server is running at a URL endpoint)
# summary_connection_params = SseServerParams(url=summary_mcp_url_or_command)
# ---------------------------------

# Global ExitStack and Toolset instance (lazy initialized)
_summary_exit_stack: Optional[AsyncExitStack] = None
_summary_mcp_toolset: Optional[MCPToolset] = None
_summary_tools: Optional[Dict[str, MCPTool]] = None

async def get_summary_mcp_tools() -> Dict[str, MCPTool]:
    """Initializes and returns the MCP tools for summarization."""
    global _summary_exit_stack, _summary_mcp_toolset, _summary_tools
    if _summary_tools is None:
        logger.info(f"Initializing MCPToolset for Summary Agent at {summary_mcp_url_or_command}")
        _summary_exit_stack = AsyncExitStack()
        _summary_mcp_toolset = MCPToolset(connection_params=summary_connection_params, exit_stack=_summary_exit_stack)
        await _summary_exit_stack.enter_async_context(_summary_mcp_toolset)
        _summary_tools = {tool.name: tool for tool in await _summary_mcp_toolset.load_tools()}
        logger.info(f"Loaded Summary MCP tools: {list(_summary_tools.keys())}")

        # --- Register cleanup ---
        import atexit
        if asyncio.get_event_loop().is_running():
             atexit.register(lambda: asyncio.ensure_future(_cleanup_summary_mcp_session()))
        else:
             atexit.register(asyncio.run, _cleanup_summary_mcp_session())

    return _summary_tools

async def _cleanup_summary_mcp_session():
    """Cleans up the Summary MCP session on exit."""
    global _summary_exit_stack, _summary_tools
    if _summary_exit_stack:
        logger.info("Cleaning up Summary Agent MCP session...")
        await _summary_exit_stack.aclose()
        _summary_exit_stack = None
        _summary_tools = None
        logger.info("Summary Agent MCP session closed.")

async def summarize_video(video_id: str) -> str:
    """Creates a summary for a single YouTube video via MCP."""
    tools = await get_summary_mcp_tools()
    # *** Replace with the EXACT name provided by your MCP server ***
    tool_name = "summarize_video"
    if tool_name not in tools:
        raise ValueError(f"MCP Tool '{tool_name}' not found. Available: {list(tools.keys())}")
    logger.info(f"Calling MCP tool '{tool_name}' for video {video_id}")
    try:
        # Pass arguments matching the MCP tool's expected input schema
        result = await tools[tool_name].run_async(args={"video_id": video_id}, tool_context=None)
        # Assuming the tool returns a dict with a 'summary' key, or just the summary string
        if isinstance(result, dict) and "summary" in result:
             return str(result["summary"])
        else:
             return str(result) # Ensure result is string
    except Exception as e:
        logger.error(f"Error calling MCP tool '{tool_name}': {e}", exc_info=True)
        return f"Error: Failed to summarize video {video_id} - {type(e).__name__}"


async def combine_summaries(summaries: list[str]) -> str:
    """Combines multiple video summaries into a final summary via MCP."""
    tools = await get_summary_mcp_tools()
    # *** Replace with the EXACT name provided by your MCP server ***
    tool_name = "combine_summaries"
    if tool_name not in tools:
        raise ValueError(f"MCP Tool '{tool_name}' not found. Available: {list(tools.keys())}")
    logger.info(f"Calling MCP tool '{tool_name}' with {len(summaries)} summaries")
    try:
        # Pass arguments matching the MCP tool's expected input schema
        result = await tools[tool_name].run_async(args={"summaries": summaries}, tool_context=None)
        # Assuming the tool returns a dict with a 'combined_summary' key, or just the summary string
        if isinstance(result, dict) and "combined_summary" in result:
             return str(result["combined_summary"])
        else:
             return str(result)
    except Exception as e:
        logger.error(f"Error calling MCP tool '{tool_name}': {e}", exc_info=True)
        return f"Error: Failed to combine summaries - {type(e).__name__}"

# --- A2A Delegation Tool ---
async def find_videos_via_a2a(
    youtube_agent_url: str,
    query: str,
    session_id: str,
    tool_context: Any = None # ADK Tool context
) -> list[str]:
    """
    Delegates the task of finding YouTube videos to a remote YouTube agent via A2A.
    Returns a list of video IDs or an error string inside the list.
    """
    logger.info(f"Initiating A2A call to {youtube_agent_url} with query: '{query}'")
    try:
        a2a_client = A2AClient(url=youtube_agent_url)
        task_params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text=query)]),
            sessionId=session_id,
            acceptedOutputModes=["application/json", "text/plain"] # Expect JSON list
        )
        # Use non-streaming for simplicity in getting the final list
        response = await a2a_client.send_task(task_params.model_dump(exclude_defaults=True))
        logger.info(f"A2A response received: Status={response.result.status.state}")

        if response.result.status.state == TaskState.COMPLETED and response.result.artifacts:
            video_ids = []
            for artifact in response.result.artifacts:
                for part in artifact.parts:
                    # Prefer DataPart if available
                    if part.type == "data" and isinstance(part.data, list):
                        video_ids.extend(str(item) for item in part.data)
                        logger.info(f"Extracted video IDs via DataPart: {video_ids}")
                        return video_ids # Return immediately if found DataPart
                    # Fallback to TextPart (expecting JSON string list)
                    elif part.type == "text":
                        try:
                            data = json.loads(part.text)
                            if isinstance(data, list):
                                 # Check if it's an error list returned by the agent
                                if data and isinstance(data[0], str) and data[0].startswith("Error:"):
                                     logger.warning(f"YouTube agent returned error via JSON list: {data[0]}")
                                     return data # Propagate the error list
                                else:
                                     video_ids.extend(str(item) for item in data)
                                     logger.info(f"Extracted video IDs via TextPart: {video_ids}")
                                     return video_ids # Return if valid list found in TextPart
                        except json.JSONDecodeError:
                            logger.warning(f"A2A Text artifact not JSON: {part.text}")
                            # If text isn't JSON, treat it as a potential error message from the agent
                            if part.text.startswith("Error:"):
                                return [part.text]
                            else:
                                # Or just log it and continue searching artifacts
                                logger.warning(f"Received unexpected text artifact content: {part.text}")
                                continue

            # If loop finishes without returning a list from DataPart or TextPart
            logger.warning("A2A task completed but no suitable artifact found or parsed.")
            return ["Error: YouTube agent returned no parsable video list."]

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
