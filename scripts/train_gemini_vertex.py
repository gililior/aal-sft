#!/usr/bin/env python3
"""Launch a Vertex AI (Gemini Enterprise Agent Platform) supervised tuning job.

    pip install google-genai
    gcloud auth application-default login
    python scripts/train_gemini_vertex.py \
        --project MY_PROJECT --bucket MY_BUCKET \
        --data data/v1/stateless --base-model gemini-3.5-flash --epochs 3

Notes
* Google's tuning docs (checked Oct 2026) list Gemini 3.5 Flash and 3.1 Flash-Lite
  as tunable; Gemini 3.8 Flash is not listed. Pass --base-model gemini-3.8-flash
  to try it; the API will reject it if unsupported.
* Gemini SFT trains without the thinking trace. Evaluate the tuned model with
  thinking_level=MINIMAL (eval default for tuned models), and keep that in mind
  when comparing against the paper's thinking_level=high baselines.
* Regions with tuning support: us-central1, europe-west4.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def gcs_upload(local: str, uri: str):
    subprocess.run(["gcloud", "storage", "cp", local, uri], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--bucket", required=True, help="GCS bucket name (no gs://)")
    ap.add_argument("--data", required=True, help="dir with train.gemini.jsonl / val.gemini.jsonl")
    ap.add_argument("--base-model", default="gemini-3.5-flash")
    ap.add_argument("--name", default=None, help="tuned model display name")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-multiplier", type=float, default=1.0)
    ap.add_argument("--adapter-size", default=None,
                    help="e.g. ADAPTER_SIZE_EIGHT / ADAPTER_SIZE_SIXTEEN (default: service default)")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    from google import genai
    from google.genai import types

    name = args.name or f"aal-sft-{os.path.basename(args.data.rstrip('/'))}-{time.strftime('%m%d-%H%M')}"
    prefix = f"gs://{args.bucket}/aal_sft/{name}"
    train_uri, val_uri = f"{prefix}/train.jsonl", f"{prefix}/val.jsonl"
    gcs_upload(os.path.join(args.data, "train.gemini.jsonl"), train_uri)
    val_local = os.path.join(args.data, "val.gemini.jsonl")
    has_val = os.path.exists(val_local) and os.path.getsize(val_local) > 0
    if has_val:
        gcs_upload(val_local, val_uri)

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    cfg = dict(tuned_model_display_name=name, epoch_count=args.epochs,
               learning_rate_multiplier=args.lr_multiplier)
    if args.adapter_size:
        cfg["adapter_size"] = args.adapter_size
    if has_val:
        cfg["validation_dataset"] = types.TuningValidationDataset(gcs_uri=val_uri)

    job = client.tunings.tune(
        base_model=args.base_model,
        training_dataset=types.TuningDataset(gcs_uri=train_uri),
        config=types.CreateTuningJobConfig(**cfg),
    )
    print("tuning job:", job.name)
    if args.no_wait:
        return

    while True:
        job = client.tunings.get(name=job.name)
        state = str(job.state)
        print(time.strftime("%H:%M:%S"), state, flush=True)
        if any(s in state for s in ("SUCCEEDED", "FAILED", "CANCELLED")):
            break
        time.sleep(60)
    if "SUCCEEDED" not in str(job.state):
        sys.exit(f"tuning ended in {job.state}: {getattr(job, 'error', '')}")
    info = {"job": job.name, "base_model": args.base_model, "data": args.data,
            "tuned_model": job.tuned_model.model, "endpoint": job.tuned_model.endpoint}
    print(json.dumps(info, indent=2))
    with open(os.path.join(args.data, f"{name}.tuned.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("\nEvaluate with:\n  python scripts/evaluate.py --backend vertex "
          f"--model {info['endpoint']} --project {args.project} --location {args.location}")


if __name__ == "__main__":
    main()
