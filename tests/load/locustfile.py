"""Load-test the recommendations API with valid MovieLens requests."""

import json
import math
import random
from pathlib import Path

from locust import HttpUser, between, task

PAYLOAD_DIR = Path(__file__).parent / "payloads"
SCENARIOS = tuple(
    (path.stem, json.loads(path.read_text(encoding="utf-8")))
    for path in sorted(PAYLOAD_DIR.glob("*.json"))
)


def validate_recommendations(payload: dict, body: dict) -> None:
    """Reject incorrect responses even when the API returns HTTP 200."""
    if body["user_id"] != payload["user_id"]:
        raise ValueError("Unexpected user_id")

    candidates = {item["movie_id"]: item for item in payload["candidates"]}
    items = body["items"]
    expected_count = min(payload.get("top_k", len(candidates)), len(candidates))
    if not isinstance(items, list) or len(items) != expected_count:
        raise ValueError("Unexpected number of recommendations")

    movie_ids = [item["movie_id"] for item in items]
    if len(set(movie_ids)) != len(movie_ids) or not set(movie_ids) <= candidates.keys():
        raise ValueError("Recommendations contain duplicate or unknown movies")

    scores = [item["score"] for item in items]
    if not all(
        isinstance(score, (int, float)) and math.isfinite(score) for score in scores
    ):
        raise ValueError("Recommendation scores must be finite numbers")
    if scores != sorted(scores, reverse=True):
        raise ValueError("Recommendations are not ranked by descending score")

    for item in items:
        candidate = candidates[item["movie_id"]]
        if item["title"] != candidate["title"] or item["genres"] != candidate["genres"]:
            raise ValueError("Recommendation movie metadata differs from the request")


class RecommendationUser(HttpUser):
    host = "http://localhost:8003"
    wait_time = between(0.1, 0.5)

    @task
    def recommend(self) -> None:
        scenario, payload = random.choice(SCENARIOS)
        with self.client.post(
            "/v1/recommendations",
            json=payload,
            name=f"/v1/recommendations [{scenario}]",
            timeout=10,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Expected HTTP 200, got {response.status_code}")
                return
            try:
                validate_recommendations(payload, response.json())
            except (KeyError, TypeError, ValueError) as error:
                response.failure(f"Invalid recommendation response: {error}")
