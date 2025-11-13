import streamlit as st
import os
import math
import time
import traceback
import pandas as pd
from json_creator import create_json_from_file
from http_client import (
    post_incident_json,
    extract_incidents_from_response,
    get_summarized_output,
)
from faiss_updater import update_faiss_with_new_data
from retriever import IncidentRetriever
from utils.logger import get_logger
import json
from dotenv import load_dotenv

load_dotenv()

logger = get_logger("app")

DEFAULT_AI_AGENT_ID = os.getenv("AI_AGENT_ID")
DEFAULT_SUMMARIZATION_AGENT_ID = os.getenv("SUMMARIZATION_AGENT_ID")
DEFAULT_ENDPOINT = os.getenv("SAGE_ENDPOINT")

st.set_page_config(page_title="Incident Search", layout="centered")
st.title("Incident Resolution Assistant")

# Search Section
st.header("Enter new Incident")
query = st.text_input("Enter your issue:")

if st.button("Search"):

    if not query.strip():
        st.warning("Please enter a query.")

    else:
        try:
            retriever = IncidentRetriever()
            results, distances = retriever.search(query, 5)

            if results.empty:
                st.info("No matching incidents found.")

            else:
                # Calculate confidence scores and filter <40%
                confidence_scores = [(1 / (1 + dist)) * 100 for dist in distances]
                results["confidence"] = confidence_scores
                results = results[results["confidence"] >= 40.0].reset_index(drop=True)

                if results.empty:
                    st.info("No incidents found with sufficient confidence (>= 40%).")

                else:
                    summary = get_summarized_output(
                        results.to_dict(orient="records"),
                        ai_agent_id= DEFAULT_SUMMARIZATION_AGENT_ID,
                        configuration_environment="DEV",
                        endpoint= DEFAULT_ENDPOINT,
                    )
                    # Safely navigate nested JSON
                    agent_response = (
                        summary.get("data", {})
                        .get("responses", {})
                        .get("agent_response", {})
                    )

                    if not agent_response:
                        st.warning("No summarized response found.")

                    else:
                        st.subheader("Related Incidents")
                        st.write(", ".join(agent_response.get("incident_numbers", [])))
                        st.subheader("Overview")
                        st.write(agent_response.get("overview", "N/A"))
                        st.subheader("Common Reasons")
                        for reason in agent_response.get("common_reasons", []):
                            st.write(f"- {reason}")
                        st.subheader("Suggested Resolutions")
                        for fix in agent_response.get("suggested_resolutions", []):
                            st.write(f"- {fix}")
                        st.subheader("Key Takeaways")
                        st.write(agent_response.get("key_takeaways", "N/A"))

        except Exception:
            st.error("Error during search:")
            logger.exception("Search error")

# Upload Section
MAX_BATCH_SIZE = 10

st.header("Upload Monthly Incidents")
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file", type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    try:
        os.makedirs("data", exist_ok=True)
        ext = uploaded_file.name.split(".")[-1].lower()
        temp_path = os.path.join("data", f"temp_upload.{ext}")

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Creating temporary JSON from uploaded file...")
        temp_json_path = create_json_from_file(temp_path)
        st.success(f"Temporary JSON created: {temp_json_path}")
        logger.info("Temporary JSON created at %s", temp_json_path)

        with open(temp_json_path, "r", encoding="utf-8") as f:
            all_incidents = json.load(f)

        total = len(all_incidents)
        num_batches = math.ceil(total / MAX_BATCH_SIZE)
        st.info(
            f"Total incidents: {total}. Processing in {num_batches} batch(es) of up to {MAX_BATCH_SIZE} each."
        )
        all_processed_incidents = []

        for i in range(num_batches):
            batch = all_incidents[i * MAX_BATCH_SIZE : (i + 1) * MAX_BATCH_SIZE]
            st.write(f"Processing batch {i + 1}/{num_batches} ({len(batch)} items)...")
            batch_path = os.path.join("data", f"batch_{i + 1}.json")

            with open(batch_path, "w", encoding="utf-8") as bf:
                json.dump(batch, bf, ensure_ascii=False, indent=2)

            try:
                response_json = post_incident_json(
                    json_path=batch_path,
                    ai_agent_id= DEFAULT_AI_AGENT_ID,
                    configuration_environment="DEV",
                    endpoint= DEFAULT_ENDPOINT,
                )
                logger.info("Batch %d response: %s", i + 1, response_json)
                success_flag = response_json.get("success") or (
                    response_json.get("status") == "success"
                )

                if not success_flag:
                    logger.warning(
                        "Batch %d: Remote endpoint returned success=false: %s",
                        i + 1,
                        response_json,
                    )
                    st.warning(
                        f"Batch {i + 1}: Remote endpoint returned success=false. Skipping this batch."
                    )
                    continue
                incidents = extract_incidents_from_response(response_json)

                if incidents:
                    all_processed_incidents.extend(incidents)
                    st.success(
                        f"Batch {i + 1}: Received {len(incidents)} incidents from remote."
                    )
                    logger.info(
                        "Batch %d: Extracted %d incidents", i + 1, len(incidents)
                    )

                else:
                    st.warning(f"Batch {i + 1}: No incidents found in remote response.")
                    logger.warning(
                        "Batch %d: No incidents found in response: %s",
                        i + 1,
                        response_json,
                    )

            except Exception as e:
                logger.exception("Batch %d failed: %s", i + 1, e)
                st.error(f"Batch {i + 1} failed — continuing...")
                continue
            time.sleep(3)

        if not all_processed_incidents:
            st.error("No incidents successfully processed.")
            logger.error("No batches processed successfully.")

        else:
            processed_path = os.path.join("data", "incidents_from_api.json")
            with open(processed_path, "w", encoding="utf-8") as f:
                json.dump(all_processed_incidents, f, ensure_ascii=False, indent=2)
            st.success(
                f"All batches completed. Total incidents received: {len(all_processed_incidents)}"
            )
            logger.info("Saved all incidents to %s", processed_path)
            st.info("Updating FAISS index (this may take a while)...")
            new_count, total_count = update_faiss_with_new_data(processed_path)
            st.success(
                f"FAISS update complete. Added {new_count} new incidents. Total: {total_count}"
            )
            logger.info(
                "FAISS update completed. new_count=%d total_count=%d",
                new_count,
                total_count,
            )
            
    except Exception:
        st.error("Error during upload and processing:")
        logger.exception("Upload processing error")
