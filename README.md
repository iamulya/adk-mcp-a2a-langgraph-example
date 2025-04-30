# ADK/LangGraph A2A YouTube Summarizer 

This project demonstrates inter-agent communication using the A2A protocol between an agent built with Google's Agent Development Kit (ADK) and another built with LangGraph. It uses real MCP tool endpoints and Google Cloud Secret Manager for API key handling.

- **LangGraph YouTube Agent**: Finds YouTube video IDs based on channel/date or playlist ID using specific MCP tool URLs. Runs an A2A server.
- **ADK Summary Agent**: Coordinates the summarization process. It receives a user request, delegates video finding to the YouTube Agent via A2A, receives the video IDs, calls specific MCP tool URLs to summarize each video, combines the summaries using another MCP tool URL, and returns the final result (streaming updates). Runs an A2A server.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended Python package manager)
- Google Cloud Project with Secret Manager API enabled.
- A secret stored in Google Cloud Secret Manager containing your Google AI (Gemini) API Key.
- URLs for **four** running MCP servers (or fewer if tools are hosted together):
    - `get_playlist_videos` tool endpoint.
    - `get_youtube_videos_for_channel_date` tool endpoint.
    - `get_youtube_video_summary` tool endpoint.
    - `generate_final_summary` tool endpoint.
- (Optional) Service account JSON files if your MCP servers require authentication.
- `a2a-samples` library installed or `common` directory copied from the `google-a2a` repository.

## Setup

1.  **Clone the repository (or run the creation script):**
    ```bash
    cd adk-langgraph-a2a-youtube-summarizer
    ```

2.  **Install Dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate
    # Ensure a2a-samples is installed or common dir setup in pyproject.toml
    uv pip install -e . # Install root package and workspace members
    ```

3.  **Configure Environment Variables:**
    *   Fill in the placeholders in `langgraph_youtube_agent/.env`. **Remove `GOOGLE_API_KEY`**. Add `SECRET_PROJECT_ID` and `GOOGLE_API_KEY_SECRET_ID`. Add `MCP_URL_GET_PLAYLIST` and `MCP_URL_GET_CHANNEL`.
    *   Fill in the placeholders in `adk_summary_agent/.env`. **Remove `GOOGLE_API_KEY`**. Add `SECRET_PROJECT_ID` and `GOOGLE_API_KEY_SECRET_ID`. Add `MCP_URL_SUMMARIZE` and `MCP_URL_COMBINE`. Ensure `YOUTUBE_AGENT_A2A_URL` is correct.
    *Replace placeholders.*

4.  **Authentication:** Ensure your environment is authenticated to Google Cloud with permissions to access the specified secret in Secret Manager (e.g., run `gcloud auth application-default login`).

## Running

1.  **Start the MCP Servers:** Ensure your MCP servers are running and accessible at the URLs specified in the `.env` files.

2.  **Start the LangGraph YouTube Agent A2A Server:** (Terminal 1)
    ```bash
    # From the root directory, activate venv
    uv run langgraph_youtube_agent --port 10003
    ```

3.  **Start the ADK Summary Agent A2A Server:** (Terminal 2)
    ```bash
    # From the root directory, activate venv
    uv run adk_summary_agent --port 10004
    ```

4.  **Interact using an A2A Client:** (Terminal 3)
    Use an A2A client pointed at the ADK Summary Agent's URL (`http://localhost:10004`).
    ```bash
    # Example using the sample CLI
    # cd <path_to_google_a2a>/samples/python/hosts/cli
    # Activate venv if needed
    # uv run . --agent http://localhost:10004 --stream
    ```
    Enter prompts like:
    *   `Summarize videos from channel <channel_id> on <YYYY-MM-DD>`
    *   `Give me a summary for playlist <playlist_id>`

    Observe logs and streaming output.
