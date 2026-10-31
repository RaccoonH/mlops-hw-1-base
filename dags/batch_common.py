import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()


def validate_rows(rows, day):
    if len(rows) == 0:
        raise ValueError("Empty rows")

    customer_set = set()
    for r in rows:
        if len(r['customer_id']) == 0:
            raise ValueError("Wrong customer_id")
        if r['customer_id'] in customer_set:
            raise ValueError("Not unique customer_id")
        customer_set.add(r['customer_id'])

        if 'business_date' not in r:
            raise KeyError("No business_date")
        if r['business_date'] != day:
            raise ValueError("Wrong business_date")

        for idx in range(len(FEATURES)):
            feature = FEATURES[idx]
            limits = LIMITS[idx]
            if feature not in r:
                raise KeyError(f"No {feature}")
            if isinstance(r[feature], float) == False or math.isnan(r[feature]) or r[feature] < limits[0] or r[feature] > limits[1]:
                raise ValueError(f"Wrong value {feature}")

    return rows


def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


def infer_batches(rows, model, batch_size):
    if isinstance(batch_size, int) == False or isinstance(batch_size, bool) or batch_size < 1 or batch_size > 1000:
        raise ValueError("Wrong batch_size")

    result = []
    for i in range(0, len(rows), batch_size):
        size = min(batch_size, len(rows) - i)
        print(f"Infer {i / batch_size} size {size}")
        result.extend(predict(rows[i:i+size], model))
    return result

def validate_predictions(predictions, expected_ids):
    customer_ids = set()
    for p in predictions:
        customer_ids.add(p['customer_id'])
        if isinstance(p['score'], float) == False or p['score'] < 0 or p['score'] > 1:
            raise ValueError("Wrong score")
        if isinstance(p['label'], int) == False or p['label'] < 0 or p['label'] > 1:
            raise ValueError("Wrong label")
        if int(p['score'] >= 0.5) != p['label']:
            raise ValueError("Wrong label")
    if customer_ids != set(expected_ids):
        raise ValueError("Wrong customer_ids")


def replace_day(conn, day, model_version, predictions):
    if len(predictions) == 0:
        raise ValueError("Empty predictions")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{day}:{model_version}",))
            cur.execute("DELETE FROM predictions WHERE business_date = %s AND model_version = %s", (day, model_version))
            cur.executemany("INSERT INTO predictions (business_date, model_version, customer_id, score, label) VALUES (%s, %s, %s, %s, %s)",
                [(day, model_version, p['customer_id'], p['score'], p['label']) for p in predictions]
            )
