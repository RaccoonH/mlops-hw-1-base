from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from batch_common import business_date, validate_rows, save, read, digest, infer_batches, validate_predictions, replace_day

import pendulum
import requests
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import dag, task, Param, BaseHook


@dag(
    dag_id="hw_basic_batch_inference",
    start_date=pendulum.datetime(2026, 9, 1, tz="Europe/Moscow"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    params={
        "business_date": Param("2026-09-01", type="string", format="date"),
        "batch_size": Param(32, type="integer", minimum=1, maximum=1000)
    },
    default_args={
        "retries": 2,
        "retry_delay": pendulum.duration(seconds=5),
        "execution_timeout": pendulum.duration(minutes=2),
    },
    tags=["homework", "basic"],
)
def basic_batch_inference():
    @task
    def extract(**context: Any) -> dict:
        date = business_date(context['params']['business_date'], context.get('data_interval_start'))
        conn = BaseHook.get_connection("source_api")
        response = requests.get(f"{conn.host}/basic/{date}.json", timeout=10)
        response.raise_for_status()
        payload = response.json()
        if payload['business_date'] != date:
            raise ValueError()

        response = requests.get(f"{conn.host}/models/v1.json", timeout=10)
        response.raise_for_status()
        return save({
            "rows": payload['rows'],
            "business_date": payload['business_date'],
            "model": response.json(),
            "model_sha256": digest(response.content)
        })

    @task(retries=0)
    def infer(ref: dict, **context: Any) -> dict:
        artefact = read(ref)
        validated_rows = validate_rows(artefact['rows'], artefact['business_date'])
        infer_result = infer_batches(validated_rows, artefact['model'], context['params']['batch_size'])
        expected_ids = [r['customer_id'] for r in validated_rows]
        validate_predictions(infer_result, expected_ids)
        return save({
            "business_date": artefact['business_date'],
            "model_version": "v1",
            "predictions": infer_result,
            "model_sha256": artefact['model_sha256']
        })

    @task
    def publish(ref: dict) -> dict:
        result = read(ref)
        conn = PostgresHook(postgres_conn_id="warehouse").get_conn()
        try:
            replace_day(conn, result['business_date'], result['model_version'], result['predictions'])
        finally:
            conn.close()
        return {
            "business_date": result['business_date'],
            "model_version": result['model_version'],
            "rows": len(result['predictions']),
        }


    publish(infer(extract()))


basic_batch_inference()
