#!/usr/bin/env python3
"""Create a reproducible, age-balanced SFHQ-T2I source list before teacher inference."""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_RATIOS = "adult=0.80,senior=0.10,teen=0.05,child=0.05"
AGE_PATTERN = re.compile(r"\b(\d{1,3})\s*[- ]?\s*year[s]?\s*[- ]?\s*old\b", re.I)
MONTH_PATTERN = re.compile(r"\b\d{1,3}\s*[- ]?\s*month[s]?\s*[- ]?\s*old\b", re.I)
HIGH_TEXTURE_TERMS = (
    "age-spotted",
    "detailed pores",
    "detailed skin",
    "detailed texture",
    "leathery",
    "lined skin",
    "papery",
    "porous skin",
    "rough skin",
    "saggy",
    "sun-damaged",
    "weathered",
    "wrinkled",
)
NONPHOTO_TERMS = (
    "3d render",
    "artistic portrayal",
    "cartoon",
    "digital art",
    "illustration",
    "oil painting",
    "painting",
)


def parse_ratios(value):
    ratios = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError(f"ratio must use BUCKET=VALUE: {item}")
        name, amount = item.split("=", 1)
        ratios[name.strip()] = float(amount)
    expected = {"adult", "senior", "teen", "child"}
    if set(ratios) != expected:
        raise ValueError(f"ratios must define exactly: {', '.join(sorted(expected))}")
    total = sum(ratios.values())
    if any(value < 0 for value in ratios.values()) or abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must be non-negative and sum to 1.0, got {total}")
    return ratios


def extract_age(prompt):
    match = AGE_PATTERN.search(prompt)
    if match:
        return int(match.group(1))
    if MONTH_PATTERN.search(prompt):
        return 0
    return None


def age_bucket(age):
    if age is None:
        return None
    if age < 13:
        return "child"
    if age < 20:
        return "teen"
    if age < 60:
        return "adult"
    return "senior"


def contains_term(prompt, terms):
    lowered = prompt.lower()
    return any(term in lowered for term in terms)


def allocate(total, ratios):
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def sample_bucket(rows, count, max_texture_ratio, rng):
    safe = [row for row in rows if not row["high_texture"]]
    textured = [row for row in rows if row["high_texture"]]
    rng.shuffle(safe)
    rng.shuffle(textured)

    textured_count = min(len(textured), round(count * max_texture_ratio))
    safe_count = min(len(safe), count - textured_count)
    textured_count = min(len(textured), count - safe_count)
    selected = safe[:safe_count] + textured[:textured_count]
    if len(selected) != count:
        raise ValueError(f"bucket has only {len(selected)} usable rows; requested {count}")
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="input/sfhq_t2i/SFHQ_T2I_dataset.csv")
    parser.add_argument("--images", default="input/sfhq_t2i/images/images")
    parser.add_argument("--out", default="out/sfhq_sources_10k.txt")
    parser.add_argument("--report", default="out/sfhq_sources_10k.json")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", default=DEFAULT_RATIOS)
    parser.add_argument(
        "--max-texture-ratio",
        type=float,
        default=0.15,
        help="Maximum high-texture prompt fraction inside each age bucket.",
    )
    parser.add_argument(
        "--include-dalle3",
        action="store_true",
        help="Include DALLE3 rows despite the dataset author's quality warning.",
    )
    parser.add_argument(
        "--include-nonphoto",
        action="store_true",
        help="Include prompts explicitly requesting paintings, illustration, or 3D renders.",
    )
    args = parser.parse_args()

    if args.n <= 0:
        parser.error("--n must be positive")
    if not 0 <= args.max_texture_ratio <= 1:
        parser.error("--max-texture-ratio must be between 0 and 1")
    try:
        ratios = parse_ratios(args.ratios)
    except ValueError as exc:
        parser.error(str(exc))

    image_dir = Path(args.images)
    rows_by_bucket = {name: [] for name in ratios}
    rejected = Counter()
    model_input = Counter()

    with Path(args.csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filename = row["image_filename"]
            model = row["model_used"]
            model_input[model] += 1
            try:
                configs = json.loads(row["configs"])
            except (json.JSONDecodeError, TypeError):
                configs = {}
            prompt = configs.get("orig_prompt") or row["text_prompt"]
            age = extract_age(prompt)
            bucket = age_bucket(age)

            if bucket is None:
                rejected["unknown_age"] += 1
                continue
            if model == "DALLE3" and not args.include_dalle3:
                rejected["dalle3"] += 1
                continue
            if contains_term(prompt, NONPHOTO_TERMS) and not args.include_nonphoto:
                rejected["nonphoto_prompt"] += 1
                continue
            if not (image_dir / filename).is_file():
                rejected["missing_image"] += 1
                continue

            rows_by_bucket[bucket].append(
                {
                    "filename": filename,
                    "model": model,
                    "age": age,
                    "high_texture": contains_term(prompt, HIGH_TEXTURE_TERMS),
                }
            )

    targets = allocate(args.n, ratios)
    rng = random.Random(args.seed)
    selected = []
    for bucket, count in targets.items():
        selected.extend(
            sample_bucket(rows_by_bucket[bucket], count, args.max_texture_ratio, rng)
        )
    rng.shuffle(selected)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{row['filename']}\n" for row in selected), encoding="utf-8"
    )

    report = {
        "csv": args.csv,
        "images": args.images,
        "output": args.out,
        "n": len(selected),
        "seed": args.seed,
        "ratios": ratios,
        "max_texture_ratio": args.max_texture_ratio,
        "selected_by_age": dict(Counter(row["age"] for row in selected)),
        "selected_by_bucket": dict(Counter(age_bucket(row["age"]) for row in selected)),
        "selected_by_model": dict(Counter(row["model"] for row in selected)),
        "selected_high_texture": sum(row["high_texture"] for row in selected),
        "available_by_bucket": {name: len(rows) for name, rows in rows_by_bucket.items()},
        "input_by_model": dict(model_input),
        "rejected": dict(rejected),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[done] selected={len(selected)} list={output_path} report={report_path}")
    print(f"[buckets] {report['selected_by_bucket']}")
    print(f"[models] {report['selected_by_model']}")
    print(
        f"[texture] {report['selected_high_texture']}/{len(selected)} "
        f"({report['selected_high_texture'] / len(selected):.1%})"
    )
    print(f"[rejected] {report['rejected']}")


if __name__ == "__main__":
    main()
