# adk-langgraph-a2a-youtube-summarizer/utils/secrets.py
import os
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1) # Cache the secret to avoid repeated calls
def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    """Retrieves a secret from Google Cloud Secret Manager."""
    try:
        from google.cloud import secretmanager
    except ImportError:
        raise ImportError("Please install google-cloud-secret-manager (`uv pip install google-cloud-secret-manager`)")

    if not project_id or not secret_id:
        raise ValueError("Both SECRET_PROJECT_ID and GOOGLE_API_KEY_SECRET_ID must be set in .env")

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    logger.info(f"Attempting to fetch secret: {name}")
    try:
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        logger.info(f"Successfully fetched secret '{secret_id}'")
        return payload
    except Exception as e:
        logger.error(f"Failed to access secret version {name}: {e}", exc_info=True)
        raise ValueError(f"Could not retrieve secret {secret_id} from project {project_id}. Check permissions and configuration.") from e

def get_google_api_key_from_secret_manager() -> str:
     """Convenience function to get the Google API key using env vars."""
     secret_project_id = os.getenv("SECRET_PROJECT_ID")
     api_key_secret_id = os.getenv("GOOGLE_API_KEY_SECRET_ID")
     return get_secret(secret_project_id, api_key_secret_id)
