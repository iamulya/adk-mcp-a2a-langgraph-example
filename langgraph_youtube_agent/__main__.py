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

from .agent import YouTubeVideoAgent
from .task_manager import AgentTaskManager
from .tools import _cleanup_mcp_session # Import cleanup function

# Load .env file from the current agent's directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10003) # Default port for this agent
def main(host, port):
    """Starts the LangGraph YouTube Agent A2A server."""
    try:
        # Check for necessary environment variables
        if not os.getenv("SECRET_PROJECT_ID") or not os.getenv("GOOGLE_API_KEY_SECRET_ID"):
             raise MissingAPIKeyError("SECRET_PROJECT_ID and GOOGLE_API_KEY_SECRET_ID must be set in .env")
        if not os.getenv("MCP_URL_GET_PLAYLIST"):
             raise MissingAPIKeyError("MCP_URL_GET_PLAYLIST environment variable not set.")
        if not os.getenv("MCP_URL_GET_CHANNEL"):
            raise MissingAPIKeyError("MCP_URL_GET_CHANNEL environment variable not set.")

        # Initialize the agent (this might trigger MCP connection via tools.py)
        # Need an event loop running
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        agent = YouTubeVideoAgent()

        capabilities = AgentCapabilities(streaming=True, pushNotifications=False)
        skills = [
            AgentSkill(
                id="get_channel_videos", name="Get Channel Videos by Date",
                description="Finds videos uploaded to a specific YouTube channel on a given date.",
                tags=["youtube", "video", "channel", "date"],
                examples=["Find videos from channel 'channel_xyz' on 2024-10-26"],
            ),
             AgentSkill(
                id="get_playlist_videos", name="Get Playlist Videos",
                description="Finds all videos within a specific YouTube playlist.",
                tags=["youtube", "video", "playlist"],
                examples=["Get videos from playlist 'playlist_abc'"],
            )
        ]
        agent_card = AgentCard(
            name="LangGraph YouTube Agent (MCP)",
            description="Retrieves YouTube video IDs via MCP tools.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            capabilities=capabilities,
            skills=skills,
            defaultInputModes=YouTubeVideoAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=YouTubeVideoAgent.SUPPORTED_OUTPUT_TYPES,
        )

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=agent),
            host=host,
            port=port,
        )

        logger.info(f"Starting LangGraph YouTube Agent server on {host}:{port}")
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
            # Attempt to run cleanup in the loop if it exists
            loop = asyncio.get_event_loop()
            # Need to check loop state as it might have been closed by server exit
            if loop.is_running():
                 # Schedule cleanup if running, don't wait here as server.start() blocks
                 asyncio.ensure_future(_cleanup_mcp_session(os.getenv("MCP_URL_GET_PLAYLIST")))
                 asyncio.ensure_future(_cleanup_mcp_session(os.getenv("MCP_URL_GET_CHANNEL")))
            else:
                 # Run cleanup synchronously if loop isn't running (e.g., startup error)
                 asyncio.run(_cleanup_mcp_session(os.getenv("MCP_URL_GET_PLAYLIST")))
                 asyncio.run(_cleanup_mcp_session(os.getenv("MCP_URL_GET_CHANNEL")))
        except Exception as cleanup_err:
             logger.error(f"Error during MCP cleanup: {cleanup_err}")

if __name__ == "__main__":
    main()
