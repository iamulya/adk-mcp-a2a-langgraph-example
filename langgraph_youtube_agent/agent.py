import os
import logging
import json
import asyncio
from typing import Any, AsyncIterable, Dict
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# Import async tool functions
from .tools import get_channel_videos, get_playlist_videos

logger = logging.getLogger(__name__)

# --- LangChain Tools wrapping MCP Tool functions ---
# Using sync wrappers for create_react_agent compatibility
@tool
def get_channel_videos_tool_sync(channel_id: str, date: str) -> str:
    """Looks up videos for a given YouTube channel ID and specific date (YYYY-MM-DD). Returns a JSON list of video IDs or a JSON list containing an error string."""
    try:
        # Get or create event loop for sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(get_channel_videos(channel_id, date))
        return json.dumps(result) # Return JSON string
    except Exception as e:
        logger.error(f"Error in get_channel_videos_tool_sync: {e}")
        return json.dumps([f"Error calling MCP tool get_channel_videos: {type(e).__name__}"])

@tool
def get_playlist_videos_tool_sync(playlist_id: str) -> str:
    """Looks up videos for a given YouTube playlist ID. Returns a JSON list of video IDs or a JSON list containing an error string."""
    try:
        # Get or create event loop for sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(get_playlist_videos(playlist_id))
        return json.dumps(result) # Return JSON string
    except Exception as e:
        logger.error(f"Error in get_playlist_videos_tool_sync: {e}")
        return json.dumps([f"Error calling MCP tool get_playlist_videos: {type(e).__name__}"])

# --- LangGraph Agent Definition ---
class YouTubeVideoAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
    SUPPORTED_OUTPUT_TYPES = ["application/json", "text/plain"] # Can output JSON list or text error

    def __init__(self):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)
        # *** Use the correct tool names as defined above ***
        self.tools = [get_channel_videos_tool_sync, get_playlist_videos_tool_sync]

        system_message = (
            "You are an assistant that finds YouTube video IDs using specialized tools. "
            "You have access to two tools: "
            f"1. `{get_channel_videos_tool_sync.name}`: Use this when asked for videos from a specific channel on a specific date. Requires `channel_id` and `date` (YYYY-MM-DD format)."
            f"2. `{get_playlist_videos_tool_sync.name}`: Use this when asked for videos from a specific playlist. Requires `playlist_id`."
            "Analyze the user's request and select the appropriate tool with the correct arguments. "
            "The tools will return a JSON string representing a list of video IDs or an error message inside a list (e.g., '[\"Error: ...\"]'). "
            "Your final answer should ONLY be the JSON string list returned by the tool (e.g., '[\"vid1\", \"vid2\"]') or the JSON string error list (e.g. '[\"Error: Tool failed\"]') if the tool failed or found nothing."
            "Do not add any other text before or after the JSON list or error message."
        )
        self.graph = create_react_agent(llm, self.tools, messages_modifier=system_message)
        logger.info("YouTubeVideoAgent initialized with ReAct graph.")

    def invoke(self, query: str, session_id: str) -> Any:
        """Invokes the LangGraph agent."""
        logger.info(f"Invoking YouTubeVideoAgent for query: '{query}' (Session: {session_id})")
        input_data = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": session_id}}
        try:
            final_state = self.graph.invoke(input_data, config=config)
            last_message = final_state.get("messages", [])[-1]
            response_content = last_message.content if last_message else "Agent failed to produce a response."
            logger.info(f"YouTubeVideoAgent invocation result: {response_content}")
            # Ensure the output is a JSON string list, even for errors
            try:
                json.loads(response_content) # Validate if it's already JSON
                return response_content
            except json.JSONDecodeError:
                logger.warning(f"Agent final output wasn't valid JSON, wrapping: {response_content}")
                return json.dumps([response_content]) # Wrap non-JSON as error list
        except Exception as e:
             logger.error(f"Error invoking LangGraph agent: {e}", exc_info=True)
             return json.dumps([f"Error: Failed to process request - {type(e).__name__}"])

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        """Streams the LangGraph agent's response (simplified)."""
        logger.info(f"Streaming YouTubeVideoAgent for query: '{query}' (Session: {session_id})")
        input_data = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": session_id}}
        final_result_content = json.dumps([f"Error: Failed to process request - Unknown Streaming Error"])
        try:
            async for chunk in self.graph.astream_log(input_data, config=config, include_types=["llm", "tools"]):
                # Yield intermediate thought process if desired
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing...", # Simplified update
                }

            # Get final result after stream completes
            final_state = await self.graph.ainvoke(input_data, config=config)
            last_message = final_state.get("messages", [])[-1]
            final_result_content = last_message.content if last_message else json.dumps(["Agent failed to produce a response."])
             # Ensure the output is a JSON string list, even for errors
            try:
                json.loads(final_result_content) # Validate if it's already JSON
            except json.JSONDecodeError:
                logger.warning(f"Agent final output wasn't valid JSON, wrapping: {final_result_content}")
                final_result_content = json.dumps([final_result_content]) # Wrap non-JSON as error list

        except Exception as e:
            logger.error(f"Error streaming LangGraph agent: {e}", exc_info=True)
            final_result_content = json.dumps([f"Error: Failed to process request - {type(e).__name__}"])
        finally:
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": final_result_content,
            }
