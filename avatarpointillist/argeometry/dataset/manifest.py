"""Dataset manifests; paths in records are relative to the repository working directory."""
import json


def load_manifest(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        if len(data) != 1 or not isinstance(data[0], dict):
            raise ValueError("Expected a sample mapping or a one-item list containing it")
        data = data[0]
    if not isinstance(data, dict) or not data:
        raise ValueError("Dataset manifest must contain at least one sample")
    for sample_id, record in data.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record for {sample_id}")
    return data
