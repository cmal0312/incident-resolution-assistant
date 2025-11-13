import json
import os
import requests
from utils.logger import get_logger
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()

logger = get_logger("http_client")

DEFAULT_ENDPOINT = os.getenv("SAGE_ENDPOINT")
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "insomnium/1.3.0",
    "Authorization": os.getenv("AUTH_TOKEN")
}
DEFAULT_AI_AGENT_ID = os.getenv("AI_AGENT_ID")
DEFAULT_SUMMARIZATION_AGENT_ID = os.getenv("SUMMARIZATION_AGENT_ID")
DEFAULT_TIMEOUT = 60  

def post_incident_json(
    json_path: str,
    ai_agent_id: str = DEFAULT_AI_AGENT_ID,
    configuration_environment: str = "DEV",
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: int = DEFAULT_TIMEOUT
) -> Dict:

    logger.info("Posting JSON to endpoint: %s", endpoint)
    with open(json_path, "r", encoding="utf-8") as f:
        payload_records = json.load(f)

    #body: convert the array to a JSON string 
    user_query_str = json.dumps(payload_records, ensure_ascii=False)
    body = {
        "ai_agent_id": ai_agent_id,
        "user_query": user_query_str,
        "configuration_environment": configuration_environment
    }

    try:
        resp = requests.post(endpoint, headers=DEFAULT_HEADERS, json=body, timeout=timeout)
        resp.raise_for_status()
        logger.info("POST successful, status code: %s", resp.status_code)
        try:
            response_json = resp.json()
            logger.info("Response JSON parsed")
            return response_json
        except ValueError:
            logger.error("Response content is not valid JSON")
            raise
    except requests.RequestException as e:
        logger.exception("HTTP request failed: %s", e)
        raise

def get_summarized_output(
        json_list: List[dict],
        ai_agent_id: str = DEFAULT_SUMMARIZATION_AGENT_ID,
        configuration_environment: str = "DEV",
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: int = DEFAULT_TIMEOUT
) -> Dict:
    
    logger.info("Posting data for summarization to endpoint: %s", endpoint)
    user_query_str = json.dumps(json_list, ensure_ascii=False)
    body = {
        "ai_agent_id": ai_agent_id,
        "user_query": user_query_str,
        "configuration_environment": configuration_environment
    }

    try:
        resp = requests.post(endpoint, headers=DEFAULT_HEADERS, json=body, timeout=timeout)
        resp.raise_for_status()
        logger.info("POST successful, status code: %s", resp.status_code)
        try:
            response_json = resp.json()
            logger.info("Response JSON parsed")
            return response_json
        except ValueError:
            logger.error("Response content is not valid JSON")
            raise
    except requests.RequestException as e:
        logger.exception("HTTP request failed: %s", e)
        raise

def extract_incidents_from_response(response_json: dict) -> Optional[List[dict]]:
    """
    Extracts the list of incidents ('insidents') from the API response.
    Handles both stringified and already-parsed 'agent_response'.
    """
    logger.info("Extracting incidents from API response")

    try:
        data = response_json.get("data", {})
        responses = data.get("responses", {})
        agent_response = (
            responses.get("agent_response")
            or data.get("agent_response")
        )

        if agent_response is None:
            logger.warning("agent_response not found in response JSON")
            return None
        
        if isinstance(agent_response, str):
            try:
                agent_response = agent_response.strip()
                agent_response = json.loads(agent_response)
                logger.info("Parsed stringified agent_response JSON successfully")
            except json.JSONDecodeError as e:
                logger.warning("JSON decoding failed (%s). Attempting relaxed parsing.", e)
                fixed = agent_response.replace('\n', '').replace('\r', '')
                try:
                    agent_response = json.loads(fixed)
                    logger.info("Fallback JSON parsing succeeded after cleanup")
                except Exception as e2:
                    logger.error("Fallback parsing failed: %s", e2)
                    return None


        if "agent_response" in agent_response:
            agent_response = agent_response["agent_response"]

        items = agent_response.get("insidents")
        if isinstance(items, list):
            logger.info("Found %d incidents under agent_response.insidents", len(items))
            return items

        logger.warning("No 'insidents' list found in agent_response")
        return None

    except Exception as e:
        logger.exception("Unexpected error while extracting incidents: %s", e)
        return None
