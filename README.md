# Triton models

Run Triton Server and the local `triton-inference:0.3.0` API image with Docker
Compose. Both services currently use `linux/arm64`.

The API image reads `TRITON_URL`, `TRITON_MODEL_NAME`, and `VOCABULARY_PATH`.
Compose defaults to `triton:8001` for gRPC and `recommender_onnx` for the model.
It mounts a host vocabulary file read-only at `/app/vocabulary.json` and sets
`VOCABULARY_PATH` to that container path.

The model uses dynamic batching with `max_batch_size: 32` and
`max_queue_delay_microseconds: 1000`. The Triton client must send `user_id` and
`movie_id` with shape `[B, 1]`, and `genres` with shape `[B, G]`; scores are
returned with shape `[B, 1]`. Pad genres to the vocabulary's `max_genres` so
requests have matching `G`. Each row is one user/movie pair; split more than
32 candidates into separate inference calls and combine scores before top-k
selection. The API client implementation is maintained outside this repository
and must support these tensor shapes and chunking.

Before starting, run training to export `models/recommender_onnx/1/model.onnx`
and `./artifacts/vocabulary.json` from the same training run. The JSON contains
`user_vocab`, `item_vocab`, `genre_vocab`, and `max_genres`, with the exact
mappings used to train that model. Override `training.vocabulary_path` to save
it elsewhere, and set `VOCABULARY_HOST_PATH` to match. Compose will fail if the
host file is missing instead of creating a directory at its path.

To customize the host file path or the API's Triton settings, copy `.env.example`
to `.env` and edit it. `VOCABULARY_HOST_PATH` is a host path;
`VOCABULARY_PATH` is the fixed path inside the API container. `.env` is optional
when using the defaults.

```sh
docker compose config --quiet
docker compose up -d
```

The API starts after Triton's readiness healthcheck passes. API documentation is
available at <http://localhost:8003/docs>, and recommendations use
`POST http://localhost:8003/v1/recommendations`. Triton's HTTP, gRPC, and metrics
ports are exposed at `8000`, `8001`, and `8002`, respectively. The API image must
already exist locally; Compose does not pull it from a registry.

Run the Locust load tests separately with `make load-test` after starting the
services. See [tests/load/README.md](tests/load/README.md) for scenarios, load
settings, and reports. Load tests are excluded from pytest, `make check`, and
pre-commit test execution.
