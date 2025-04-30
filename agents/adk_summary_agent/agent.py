import os
import logging
from typing import Any, AsyncIterable, Dict

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# Import async tool functions
from .tools import summarize_video, combine_summaries, find_videos_via_a2a
from ..utils.secrets import get_google_api_key_from_secret_manager # Import secret manager utility

logger = logging.getLogger(__name__)

# Get A2A URL from environment, with a default for local testing
YOUTUBE_AGENT_A2A_URL = os.getenv("YOUTUBE_AGENT_A2A_URL", "http://localhost:10003")

class SummaryAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
    SUPPORTED_OUTPUT_TYPES = ["text/plain"]

    def __init__(self):
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
        google_api_key = None
        if not use_vertex:
            try:
                google_api_key = get_google_api_key_from_secret_manager()
            except ValueError as e:
                logger.error(f"Failed to get Google API Key from Secret Manager: {e}")
                raise ValueError("GOOGLE_API_KEY_SECRET_ID and SECRET_PROJECT_ID must be set in .env when not using Vertex AI.") from e

        model_name = "gemini-1.5-flash-latest"
        logger.info(f"Initializing SummaryAgent with model: {model_name}, Vertex: {use_vertex}")

        # Wrap tools for ADK, explicitly passing configured A2A URL to the A2A tool
        # *** Tool names for FunctionTool will be the Python function names ***
        find_videos_tool = FunctionTool(
            func=find_videos_via_a2a,
            kwargs={"youtube_agent_url": YOUTUBE_AGENT_A2A_URL}
        )
        summarize_video_tool = FunctionTool(func=summarize_video)
        combine_summaries_tool = FunctionTool(func=combine_summaries)

        instruction = f"""
You are a YouTube video summarization assistant. Your goal is to provide a concise summary of YouTube videos based on user requests by coordinating with other tools and agents.

You have access to the following tools:
1.  `{find_videos_tool.name}`: Use this tool FIRST to get a list of video IDs based on the user's request (channel & date, or playlist). You MUST provide the user's query details (like channel ID, playlist ID, date) to this tool's `query` argument. You MUST also pass the current `session_id` to this tool. The tool will return a list of video IDs (e.g., ["vid1", "vid2"]) or a list containing a single error string starting with 'Error:' (e.g., ["Error: Couldn't find videos"]).
2.  `{summarize_video_tool.name}`: Takes a single `video_id` string and returns its summary string via an MCP call. It might return an error message string starting with 'Error:'.
3.  `{combine_summaries_tool.name}`: Takes a list of individual summary strings (`summaries`) and returns a final combined summary string via an MCP call. It might return an error message string starting with 'Error:'.

Workflow:
1.  Receive the user's request (e.g., "Summarize videos from channel X on date Y", "Summarize playlist Z"). Extract the `session_id` from the state.
2.  Call `{find_videos_tool.name}` with the `query` containing the specific channel/playlist details and the extracted `session_id`.
3.  Carefully analyze the list returned by `{find_videos_tool.name}`.
4.  If the list contains an error string (the first element starts with 'Error:'): Inform the user about the specific error reported by the tool. STOP processing.
5.  If the list is empty: Inform the user that no videos were found for their criteria. STOP processing.
6.  If the list contains valid video IDs:
    a. Initialize an empty list to store individual summaries.
    b. Keep track of any errors during summarization.
    c. For EACH `video_id` in the list:
        i. Call `{summarize_video_tool.name}` with the current `video_id`.
        ii. Check the result: If it's an error string (starts with 'Error:'), note the error (e.g., by printing or logging it internally) and continue to the next video. If it's a valid summary, append it to your list of summaries.
    d. After attempting to summarize all videos:
        i. If NO valid summaries were collected (all resulted in errors or the initial list was empty), inform the user about the summarization failures. STOP processing.
        ii. If there are valid summaries, call `{combine_summaries_tool.name}` with the complete list of *successfully* generated summary strings.
        iii. Check the result of `{combine_summaries_tool.name}`. If it's an error string, report that error.
        iv. If it's the combined summary, return ONLY this final combined summary string to the user. You can optionally add a brief note if some videos failed to summarize (e.g., "Here is the combined summary for the videos I could process: ...").

IMPORTANT:
- Always call `{find_videos_tool.name}` first. Check its output carefully for errors or an empty list before proceeding.
- You MUST extract the `session_id` from the state and pass it to `{find_videos_tool.name}`. The state key is 'session_id'.
- Handle potential errors from each tool call gracefully and inform the user.
- Only return the final combined summary string (or an informative error message) as your final response.
        """

        # Initialize LlmAgent. If using Vertex, API key is handled by ADK's default credentials.
        # If not using Vertex, google_api_key needs to be passed (fetched via Secret Manager).
        agent_kwargs = {
            "model": model_name,
            "name": "adk_summary_agent",
            "instruction": instruction,
            "description": "Summarizes YouTube videos using MCP tools and A2A delegation.",
            "tools": [find_videos_tool, summarize_video_tool, combine_summaries_tool],
        }
        if not use_vertex:
            agent_kwargs["api_key"] = google_api_key

        self._agent = LlmAgent(**agent_kwargs)
        logger.info("SummaryAgent initialized with ADK LlmAgent.")

    # Expose invoke/stream methods expected by the TaskManager
    def invoke(self, query: str, session_id: str) -> str:
        """Invokes the ADK agent."""
        logger.info(f"Invoking SummaryAgent for query: '{query}' (Session: {session_id})")
        from google.adk.runners import InMemoryRunner
        from google.genai import types as genai_types

        runner = InMemoryRunner(self._agent)
        # Use the provided session_id for the ADK run
        initial_state = {"session_id": session_id} # Inject session_id into state
        session = runner.session_service.create_session(
             app_name=self._agent.name, user_id="a2a_caller", session_id=session_id, state=initial_state
        )
        input_content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])

        try:
            events = list(runner.run(user_id=session.user_id, session_id=session.id, new_message=input_content))
            final_response = ""
            if events:
                last_event = events[-1]
                if last_event.error_code or last_event.error_message:
                     final_response = f"Error: Agent execution failed - {last_event.error_message or last_event.error_code}"
                elif last_event.content and last_event.content.parts:
                    final_response = "".join(p.text for p in last_event.content.parts if p.text)
            response_content = final_response if final_response else "Agent failed to produce a response."
            logger.info(f"SummaryAgent invocation result: {response_content}")
            return response_content
        except Exception as e:
            logger.error(f"Error invoking ADK agent: {e}", exc_info=True)
            return f"Error: Failed to process request - {type(e).__name__}"

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        """Streams the ADK agent's response."""
        logger.info(f"Streaming SummaryAgent for query: '{query}' (Session: {session_id})")
        from google.adk.runners import InMemoryRunner
        from google.genai import types as genai_types

        runner = InMemoryRunner(self._agent)
        initial_state = {"session_id": session_id} # Inject session_id into state
        session = runner.session_service.create_session(
             app_name=self._agent.name, user_id="a2a_caller", session_id=session_id, state=initial_state
        )
        input_content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])

        try:
            async for event in runner.run_async(user_id=session.user_id, session_id=session.id, new_message=input_content):
                content_text = ""
                is_final = event.is_final_response()
                artifacts = None # Placeholder

                if event.error_code or event.error_message:
                    content_text = f"Error during execution: {event.error_message or event.error_code}"
                    is_final = True
                elif event.content and event.content.parts:
                     content_text = "".join(p.text for p in event.content.parts if p.text)
                     if "Error:" in content_text and ("YouTube agent" in content_text or "communicate" in content_text or "summarize" in content_text or "combine" in content_text):
                         is_final = True

                yield {
                    "is_task_complete": is_final,
                    "require_user_input": False,
                    "content": content_text if content_text else "Processing step...",
                    "artifacts": artifacts
                }
                if is_final:
                    break
        except Exception as e:
            logger.error(f"Error streaming ADK agent: {e}", exc_info=True)
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Error: Failed to process request - {type(e).__name__}"
            }
