import os
import logging
import click
from dotenv import load_dotenv
import asyncio

# Assuming 'common' library is available via install or copied path
try:
    from common.server import A2AServer
    from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
except ImportError:
     raise ImportError("Could not import A2A common library. Ensure 'a2a-samples' is installed or 'common' directory is in PYTHONPATH.")

from .agent import SummaryAgent
from .task_manager import AgentTaskManager
from .tools import _cleanup_summary_mcp_session # Import cleanup function

# Load .env file from the current agent's directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10004) # Default port for this agent
def main(host, port):
    """Starts the ADK Summary Agent A2A server."""
    try:
        # Check for Secret Manager config OR Vertex config
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
        if not use_vertex and (not os.getenv("SECRET_PROJECT_ID") or not os.getenv("GOOGLE_API_KEY_SECRET_ID")):
             raise MissingAPIKeyError("SECRET_PROJECT_ID and GOOGLE_API_KEY_SECRET_ID must be set in .env if not using Vertex AI.")
        if use_vertex and (not os.getenv("GOOGLE_CLOUD_PROJECT") or not os.getenv("GOOGLE_CLOUD_LOCATION")):
             raise MissingAPIKeyError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in .env if using Vertex AI.")

        if not os.getenv("SUMMARY_MCP_SERVER_URL"):
             raise MissingAPIKeyError("SUMMARY_MCP_SERVER_URL environment variable not set.")
        if not os.getenv("YOUTUBE_AGENT_A2A_URL"):
            logger.warning("YOUTUBE_AGENT_A2A_URL not set, defaulting to http://localhost:10003")

        # Initialize the agent (this might trigger MCP connection via tools.py)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        agent = SummaryAgent()

        capabilities = AgentCapabilities(streaming=True, pushNotifications=False)
        skills = [
            AgentSkill(
                id="summarize_youtube", name="Summarize YouTube Content",
                description="Summarizes YouTube videos based on channel/date or playlist ID.",
                tags=["youtube", "summary", "video", "playlist", "channel", "mcp", "a2a"],
                examples=[
                    "Summarize videos from channel 'test_channel' on 2024-10-26",
                    "Give me a summary for playlist 'test_playlist'"
                ],
            )
        ]
        agent_card = AgentCard(
            name="ADK Summary Agent (MCP+A2A)",
            description="Summarizes YouTube videos using MCP tools and A2A delegation.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            capabilities=capabilities,
            skills=skills,
            defaultInputModes=SummaryAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=SummaryAgent.SUPPORTED_OUTPUT_TYPES,
        )

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=agent),
            host=host,
            port=port,
        )

        logger.info(f"Starting ADK Summary Agent server on {host}:{port}")
        server.start() # This blocks until server stops
    except MissingAPIKeyError as e:
        logger.error(f"Configuration Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}", exc_info=True)
        exit(1)
    finally:
        # Ensure cleanup happens even if server start fails or stops
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule cleanup if running
                asyncio.ensure_future(_cleanup_summary_mcp_session(os.getenv("MCP_URL_SUMMARIZE")))
                asyncio.ensure_future(_cleanup_summary_mcp_session(os.getenv("MCP_URL_COMBINE")))
            else:
                # Run cleanup synchronously if loop isn't running
                asyncio.run(_cleanup_summary_mcp_session(os.getenv("MCP_URL_SUMMARIZE")))
                asyncio.run(_cleanup_summary_mcp_session(os.getenv("MCP_URL_COMBINE")))
        except Exception as cleanup_err:
             logger.error(f"Error during MCP cleanup: {cleanup_err}")

if __name__ == "__main__":
    main()
