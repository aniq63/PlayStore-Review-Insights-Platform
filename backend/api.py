"""
backend/api.py

All Play Store review analysis routes live here as an APIRouter.
The FastAPI app instance and lifespan are managed in main.py.
"""

import time
import pandas as pd
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.serving.api_client import PlayStoreSentimentAPI
from src.clustering.cluster_reviews import ReviewClusterer
from src.insights.generate_insights import InsightGenerator
from src.logger import logging
from src.utils.scraper import SerpApiScraper


# --------------------------------------------------
# ROUTER
# --------------------------------------------------

router = APIRouter()


# --------------------------------------------------
# GLOBAL SERVICE INSTANCES  (initialised at startup)
# --------------------------------------------------

sentiment_client: PlayStoreSentimentAPI = None
clusterer: ReviewClusterer = None
insight_engine: InsightGenerator = None
serpapi_scraper: SerpApiScraper = None


def init_services():
    """
    Called once from main.py lifespan startup.
    Initialises all heavy ML/API dependencies so they are reused across requests.
    """
    global sentiment_client, clusterer, insight_engine, serpapi_scraper

    logging.info("Initializing backend service dependencies...")
    sentiment_client = PlayStoreSentimentAPI()
    clusterer = ReviewClusterer()
    insight_engine = InsightGenerator(output_dir="backend/static/insights")
    serpapi_scraper = SerpApiScraper()
    logging.info("Backend service dependencies initialized successfully.")


# --------------------------------------------------
# SIMPLE IN-MEMORY CACHE
# --------------------------------------------------

CACHE_TTL = 3600   # 1 hour

_cache: dict = {}


def _get_cache(key: str):
    if key in _cache:
        item = _cache[key]
        if time.time() - item["time"] < CACHE_TTL:
            return item["data"]
        del _cache[key]
    return None


def _set_cache(key: str, data):
    _cache[key] = {
        "time": time.time(),
        "data": data,
    }


# --------------------------------------------------
# REQUEST / RESPONSE MODELS
# --------------------------------------------------

class ReviewRequest(BaseModel):
    app_id: str
    reviews: List[str] = []   # optional – if empty, SerpApi is used to fetch


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@router.get("/status", summary="Service status check")
def service_status():
    """Returns whether all backend ML services are ready."""
    ready = all([
        sentiment_client is not None,
        clusterer is not None,
        insight_engine is not None,
        serpapi_scraper is not None,
    ])
    return {
        "services_ready": ready,
        "sentiment_client": sentiment_client is not None,
        "clusterer": clusterer is not None,
        "insight_engine": insight_engine is not None,
        "serpapi_scraper": serpapi_scraper is not None,
    }


@router.post("/analyze", summary="Analyze Play Store reviews")
def analyze(request: ReviewRequest):
    """
    Full pipeline:
      1. Optionally fetch reviews via SerpApi (if none provided)
      2. Batch sentiment prediction via Databricks Serving Endpoint
      3. Cluster reviews & extract topics
      4. Generate insight visualisations
      5. Return a structured JSON response (result is cached per app_id)
    """
    try:
        app_id = request.app_id
        reviews = request.reviews
        logging.info(
            f"Received analysis request for app: {app_id} with {len(reviews)} reviews"
        )

        # ── STEP 0: Validate service initialisation ──────────────────────────
        if sentiment_client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Backend services are not initialised. "
                    "Please check that DATABRICKS_TOKEN and other secrets are set."
                ),
            )

        # ── STEP 1: Cache check ──────────────────────────────────────────────
        cached = _get_cache(app_id)
        if cached:
            logging.info(f"Returning cached result for app: {app_id}")
            return cached

        # ── STEP 2: Fetch reviews via SerpApi if none were provided ──────────
        if not reviews and app_id:
            logging.info(
                f"No reviews provided – fetching from SerpApi for app: {app_id}"
            )
            reviews = serpapi_scraper.fetch_reviews(app_id, max_reviews=100)

        if not reviews:
            raise HTTPException(
                status_code=400,
                detail="Reviews list cannot be empty and could not be fetched automatically.",
            )

        # ── STEP 3: Batch sentiment prediction ──────────────────────────────
        logging.info("Starting batch prediction via Databricks Serving Endpoint...")
        batch_size = 16
        predictions = []

        for i in range(0, len(reviews), batch_size):
            batch = reviews[i : i + batch_size]
            batch_pred = sentiment_client.predict(batch)
            predictions.extend(batch_pred)

        pred_df = pd.DataFrame(predictions)
        pred_df.rename(columns={"review": "content"}, inplace=True)

        if pred_df.empty:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate predictions – empty response from model."
            )

        # ── STEP 4: Clustering & topic extraction ────────────────────────────
        logging.info("Starting Review Clusterer...")
        cluster_df = clusterer.run(pred_df)

        # ── STEP 5: Generate insight visualisations ──────────────────────────
        logging.info("Generating insight visualisations...")
        insights = insight_engine.generate_all(cluster_df)

        topics_df = (
            cluster_df["clean_topic"].value_counts().head(5).reset_index()
        )
        topics_df.columns = ["topic", "count"]
        topics = topics_df.to_dict(orient="records")

        # ── STEP 6: Build response payload ───────────────────────────────────
        sentiment_counts = pred_df["sentiment"].value_counts().to_dict()
        sentiment_distribution = {
            "Positive": sentiment_counts.get("Positive", 0),
            "Neutral":  sentiment_counts.get("Neutral",  0),
            "Negative": sentiment_counts.get("Negative", 0),
        }

        response = {
            "app_id": app_id,
            "total_reviews_analyzed": len(reviews),
            "sentiment_distribution": sentiment_distribution,
            "topics": topics,
            "insights": {
                "top_negative_topics": insights["top_negative_topics"].to_dict(
                    orient="records"
                ),
                "visualizations": {
                    "sentiment_plot": insights["sentiment_plot"],
                    "topics_plot":    insights["topics_plot"],
                    "heatmap":        insights["heatmap"],
                },
            },
        }

        # ── STEP 7: Cache & return ───────────────────────────────────────────
        _set_cache(app_id, response)
        logging.info("Analysis complete and cached successfully.")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in /analyze endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))