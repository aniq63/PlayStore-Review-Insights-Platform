import os
import sys
import time
import pandas as pd
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from src.serving.api_client import PlayStoreSentimentAPI
from src.clustering.cluster_reviews import ReviewClusterer
from src.insights.generate_insights import InsightGenerator

from src.logger import logging
from src.exception import MyException
from src.config import CATALOG, ML_SCHEMA, REGISTERED_MODEL
from src.utils.scraper import SerpApiScraper


# Initialize global dependencies for the application
sentiment_client = None
clusterer = None
insight_engine = None
serpapi_scraper = None

try:
    logging.info("Initializing Backend API dependencies...")
    sentiment_client = PlayStoreSentimentAPI()
    clusterer = ReviewClusterer()
    insight_engine = InsightGenerator(output_dir="backend/static/insights")
    serpapi_scraper = SerpApiScraper()
    logging.info("Backend API dependencies initialized successfully.")
except Exception as e:
    logging.critical(f"CRITICAL: Failed to initialize API dependencies: {e}")
    # We don't crash the server here so the health check endpoint still works,
    # and we can report the error through the /analyze endpoint.

CACHE_TTL = 3600   # 1 hour cache

app = FastAPI(title="PlayStore Review Analyzer API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# SIMPLE MEMORY CACHE
# --------------------------------------------------

cache = {}


def get_cache(key):

    if key in cache:

        item = cache[key]

        if time.time() - item["time"] < CACHE_TTL:
            return item["data"]

        del cache[key]

    return None


def set_cache(key, data):

    cache[key] = {
        "time": time.time(),
        "data": data
    }


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ReviewRequest(BaseModel):

    app_id: str
    reviews: List[str]


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/")
def health():

    return {"status": "running"}


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

@app.post("/analyze")
def analyze(request: ReviewRequest):

    try:
        app_id = request.app_id
        reviews = request.reviews
        logging.info(f"Received analysis request for app: {app_id} with {len(reviews)} reviews")

        # --------------------------------------------------
        # STEP 0: CHECK INITIALIZATION
        # --------------------------------------------------
        if sentiment_client is None:
            raise HTTPException(
                status_code=500, 
                detail="Backend failed to initialize. Please check if DATABRICKS_TOKEN and other secrets are set in your .env file."
            )

        # --------------------------------------------------
        # STEP 1: CHECK CACHE
        # --------------------------------------------------

        cached = get_cache(app_id)

        if cached:
            logging.info(f"Returning cached result for app: {app_id}")
            return cached

        if not reviews and app_id:
            logging.info(f"No reviews provided in request, fetching from SerpApi for app: {app_id}")
            reviews = serpapi_scraper.fetch_reviews(app_id, max_reviews=100)
            
        if not reviews:
            raise HTTPException(status_code=400, detail="Reviews list cannot be empty and could not be fetched.")


        # --------------------------------------------------
        # STEP 2: BATCH PREDICTION
        # --------------------------------------------------

        logging.info("Starting batch prediction via Databricks Serving Endpoint...")
        batch_size = 16
        predictions = []

        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            batch_pred = sentiment_client.predict(batch)
            predictions.extend(batch_pred)

        pred_df = pd.DataFrame(predictions)
        pred_df.rename(columns={"review": "content"}, inplace=True)

        if pred_df.empty:
            raise HTTPException(status_code=500, detail="Failed to generate predictions.")


        # --------------------------------------------------
        # STEP 3: CLUSTERING & TOPIC EXTRACTION
        # --------------------------------------------------
        
        logging.info("Starting Review Clusterer...")
        cluster_df = clusterer.run(pred_df)

        # --------------------------------------------------
        # STEP 4: GENERATE INSIGHTS (PLOTS)
        # --------------------------------------------------

        logging.info("Generating Insight Visualizations...")
        insights = insight_engine.generate_all(cluster_df)

        topics_df = cluster_df["clean_topic"].value_counts().head(5).reset_index()
        topics_df.columns = ["topic", "count"]
        topics = topics_df.to_dict(orient="records")
        
        # --------------------------------------------------
        # STEP 5: PREPARE FRONTEND JSON
        # --------------------------------------------------
        
        sentiment_counts = pred_df["sentiment"].value_counts().to_dict()
        sentiment_distribution = {
            "Positive": sentiment_counts.get("Positive", 0),
            "Neutral": sentiment_counts.get("Neutral", 0),
            "Negative": sentiment_counts.get("Negative", 0)
        }

        response = {
            "sentiment_distribution": sentiment_distribution,
            "topics": topics,
            "insights": {
                "top_negative_topics":
                    insights["top_negative_topics"].to_dict(
                        orient="records"
                    ),
                "visualizations": {
                    "sentiment_plot": insights["sentiment_plot"],
                    "topics_plot": insights["topics_plot"],
                    "heatmap": insights["heatmap"]
                }
            }
        }

        # --------------------------------------------------
        # STEP 5: CACHE RESULT
        # --------------------------------------------------

        set_cache(app_id, response)
        logging.info("Analysis complete and cached successfully.")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in /analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))