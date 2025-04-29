# ADK/LangGraph A2A YouTube Summarizer

This project demonstrates inter-agent communication using the A2A protocol between an agent built with Google's Agent Development Kit (ADK) and another built with LangGraph.

- **LangGraph YouTube Agent**: Finds YouTube video IDs based on channel/date or playlist ID using MCP tools. Runs an A2A server.
- **ADK Summary Agent**: Coordinates the summarization process. It receives a user request, delegates video finding to the YouTube Agent via A2A, receives the video IDs, calls MCP tools to summarize each video, combines the summaries using another MCP tool, and returns the final result (potentially streaming updates). Runs an A2A server.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended Python package manager)
- Access to Google AI (Gemini API Key) or Google Cloud Vertex AI
- URLs for two running MCP servers:
    - One serving YouTube tools (`get_channel_videos`, `get_playlist_videos`).
    - One serving Summarization tools (`summarize_video`, `combine_summaries`).
- (Optional) Service account JSON files if your MCP servers require authentication.
- `a2a-samples` library installed or `common` directory copied from the `google-a2a` repository.

## Setup

1.  **Clone the repository (or run the creation script):**
    ```bash
    # If you haven't already...
    # git clone <your-repo-url>
    cd adk-langgraph-a2a-youtube-summarizer
    ```

2.  **Install Dependencies:**
    ```bash
    uv venv # Create virtual environment
    source .venv/bin/activate # Or .\venv\Scripts\activate on Windows
    # Ensure a2a-samples is installed or common dir setup in pyproject.toml
    uv pip install -e . # Install root package and workspace members
    ```

3.  **Configure Environment Variables:**
    *   Fill in the placeholders in `langgraph_youtube_agent/.env`.
    *   Fill in the placeholders in `adk_summary_agent/.env`.
    *Replace placeholders with your actual keys, URLs, and paths.*

## Running

1.  **Start the MCP Servers:** Ensure your MCP servers for YouTube tools and Summarization tools are running and accessible at the URLs specified in the `.env` files.

2.  **Start the LangGraph YouTube Agent A2A Server:**
    Open a terminal, navigate to the project root (`adk-langgraph-a2a-youtube-summarizer`), activate the virtual environment, and run:
    ```bash
    # From the root directory
    uv run langgraph_youtube_agent --port 10003
    ```
    *(Wait for it to start)*

3.  **Start the ADK Summary Agent A2A Server:**
    Open *another* terminal, navigate to the project root, activate the virtual environment, and run:
    ```bash
    # From the root directory
    uv run adk_summary_agent --port 10004
    ```
    *(Wait for it to start)*

4.  **Interact using an A2A Client:**
    Use an A2A client (like the one in `google-a2a/samples/python/hosts/cli`) pointed at the ADK Summary Agent's URL (`http://localhost:10004`). Make sure the client environment also has `a2a-samples` installed or access to the `common` library.
    ```bash
    # Example using the sample CLI
    # From a third terminal, navigate to the google-a2a repo
    cd <path_to_google_a2a>/samples/python/hosts/cli
    # Activate the appropriate venv if needed
    uv run . --agent http://localhost:10004 --stream
    ```
    Then, enter prompts like:
    *   `Summarize videos from channel <channel_id> on <YYYY-MM-DD>`
    *   `Give me a summary for playlist <playlist_id>`

    Observe the logs in the agent terminals and the streaming output (or final result) in the client terminal.
