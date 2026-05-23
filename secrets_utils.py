import os
import logging
import google.auth
from google.cloud import secretmanager
from google.api_core.exceptions import GoogleAPIError

logger = logging.getLogger(__name__)

# Global cache to avoid repeated network calls
_CACHED_API_KEY = None

def get_api_key(secret_id="GEMINI_API_KEY"):
    """
    Fetches the API key from local environment variable or Google Cloud Secret Manager.
    Results are cached in memory for the lifetime of the process.
    """
    global _CACHED_API_KEY
    if _CACHED_API_KEY:
        return _CACHED_API_KEY

    # 1. Check local environment first (for local dev or direct Cloud Run mounts)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv(secret_id)
    if api_key:
        logger.debug("Found API key in environment variables")
        _CACHED_API_KEY = api_key
        return api_key

    # 2. Fetch from Google Cloud Secret Manager
    try:
        logger.info("Attempting to fetch API key from Google Cloud Secret Manager")
        credentials, project = google.auth.default()
        
        if not project:
            # Fallback if google.auth doesn't find the project ID automatically
            project = os.getenv("GOOGLE_CLOUD_PROJECT")

        if not project:
            logger.error("Could not determine Google Cloud project ID.")
            return None

        client = secretmanager.SecretManagerServiceClient(credentials=credentials)
        
        secret_names_to_try = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        if secret_id not in secret_names_to_try:
            secret_names_to_try.insert(0, secret_id)

        for name_to_try in secret_names_to_try:
            try:
                name = f"projects/{project}/secrets/{name_to_try}/versions/latest"
                response = client.access_secret_version(request={"name": name})
                api_key = response.payload.data.decode("UTF-8")
                
                _CACHED_API_KEY = api_key
                logger.info("Successfully fetched %s from Secret Manager", name_to_try)
                return api_key
            except Exception as e:
                logger.debug("Could not fetch %s from Secret Manager: %s", name_to_try, str(e))
                continue
        
        logger.error("Failed to access API key from Secret Manager (tried: %s)", ", ".join(secret_names_to_try))
        
    except GoogleAPIError as e:
        logger.error("Failed to access Secret Manager: %s", str(e))
    except Exception as e:
        logger.error("Unexpected error fetching secret: %s", str(e))
        
    return None
