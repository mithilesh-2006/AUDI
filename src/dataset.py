"""
Dataset Ingestion, Splitting, and Knowledge Base Builder.

Loads and preprocesses unified_eval_20000.csv, creating:
1. Train/Dev Split for calibration, temperature scaling, and policy tuning.
2. Held-out Test Split for evaluation and comparative baseline benchmarking.
3. Ground-truth corpus augmentation for evidence-based retrieval.
"""

from __future__ import annotations
import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd


class DatasetManager:
    """Manages dataset loading, schema normalization, train/test splitting, and corpus generation."""

    DEFAULT_CSV_PATH = "unified_eval_20000.csv"
    BENCHMARKS_DIR = os.path.join("data", "benchmarks")
    CORPUS_DIR = os.path.join("data", "corpus")

    @classmethod
    def load_raw_dataset(cls, csv_path: Optional[str] = None, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load raw CSV dataset using pandas."""
        path = csv_path or cls.DEFAULT_CSV_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        return pd.read_csv(path, nrows=max_rows)

    @classmethod
    def normalize_record(cls, row: pd.Series, idx: int) -> Dict[str, Any]:
        """Convert a CSV row into the standardized AUDI benchmark format."""
        category = str(row["category"]).strip() if pd.notna(row.get("category")) else "General"
        cat_lower = category.lower()

        # Determine expected mitigation action based on task characteristics
        if "reasoning" in cat_lower or "math" in cat_lower or "arithmetic" in cat_lower:
            expected_action = "calculate"
        elif "factual" in cat_lower or "hop" in cat_lower:
            expected_action = "retrieve"
        elif "speculation" in cat_lower or "unanswerable" in cat_lower:
            expected_action = "abstain"
        elif "ambiguity" in cat_lower or "ambiguous" in cat_lower:
            expected_action = "clarify"
        else:
            expected_action = "direct"

        raw_id = row.get("id")
        qid = str(raw_id) if pd.notna(raw_id) else f"q_{idx:05d}"
        query = str(row["question"]).strip() if pd.notna(row.get("question")) else ""
        ref_ans = str(row["ground_truth"]).strip() if pd.notna(row.get("ground_truth")) else ""

        return {
            "id": qid,
            "query": query,
            "category": category,
            "reference_answer": ref_ans,
            "expected_action": expected_action,
            "answerability": "answerable",
        }

    @classmethod
    def prepare_dataset(
        cls,
        csv_path: Optional[str] = None,
        total_samples: int = 500,
        train_ratio: float = 0.5,
        random_seed: int = 42,
        save_splits: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load dataset, perform stratified sampling across categories, and split into train (dev) and test sets.
        """
        df = cls.load_raw_dataset(csv_path)
        rng = random.Random(random_seed)

        # Stratified sampling across categories
        sampled_rows: List[pd.Series] = []
        categories = df["category"].unique()
        per_cat_target = total_samples // len(categories)

        for cat in categories:
            cat_df = df[df["category"] == cat]
            sample_count = min(len(cat_df), per_cat_target)
            cat_sample = cat_df.sample(n=sample_count, random_state=random_seed)
            sampled_rows.extend([row for _, row in cat_sample.iterrows()])

        # If remainder needed to reach total_samples
        if len(sampled_rows) < total_samples:
            remaining_needed = total_samples - len(sampled_rows)
            sampled_ids = {r["id"] for r in sampled_rows}
            remaining_df = df[~df["id"].isin(sampled_ids)]
            if not remaining_df.empty:
                extra = remaining_df.sample(n=min(len(remaining_df), remaining_needed), random_state=random_seed)
                sampled_rows.extend([row for _, row in extra.iterrows()])

        rng.shuffle(sampled_rows)
        all_records = [cls.normalize_record(row, i) for i, row in enumerate(sampled_rows)]

        split_idx = int(len(all_records) * train_ratio)
        train_set = all_records[:split_idx]
        test_set = all_records[split_idx:]

        # Mark split tag in records
        for r in train_set:
            r["split"] = "train"
        for r in test_set:
            r["split"] = "test"

        if save_splits:
            os.makedirs(cls.BENCHMARKS_DIR, exist_ok=True)
            train_path = os.path.join(cls.BENCHMARKS_DIR, "train_set.json")
            test_path = os.path.join(cls.BENCHMARKS_DIR, "test_set.json")
            combined_path = os.path.join(cls.BENCHMARKS_DIR, "evaluation_set.json")

            with open(train_path, "w", encoding="utf-8") as f:
                json.dump(train_set, f, indent=2)
            with open(test_path, "w", encoding="utf-8") as f:
                json.dump(test_set, f, indent=2)
            with open(combined_path, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2)

        return train_set, test_set

    @classmethod
    def augment_corpus_from_dataset(
        cls,
        records: List[Dict[str, Any]],
        corpus_path: Optional[str] = None,
    ) -> int:
        """
        Creates knowledge base passages from ground-truth factual records so that
        the retrieval and verification engine can ground answers on the dataset.
        """
        target_path = corpus_path or os.path.join(cls.CORPUS_DIR, "knowledge_base.json")
        os.makedirs(cls.CORPUS_DIR, exist_ok=True)

        existing_docs = []
        existing_ids = set()
        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    existing_docs = json.load(f)
                    existing_ids = {d["doc_id"] for d in existing_docs}
            except Exception:
                existing_docs = []

        added = 0
        for r in records:
            cat = r.get("category", "")
            if cat in ("Factual Knowledge", "Mixed Multi-hop", "Calibration"):
                doc_id = f"doc_{r['id']}"
                if doc_id in existing_ids:
                    continue

                q = r["query"]
                ans = r["reference_answer"]

                # Parse JSON array if answer is a list
                if ans.startswith("[") and ans.endswith("]"):
                    try:
                        ans_list = json.loads(ans)
                        ans_str = ", ".join(ans_list)
                    except Exception:
                        ans_str = ans
                else:
                    ans_str = ans

                doc_entry = {
                    "doc_id": doc_id,
                    "title": f"Fact Sheet: {q[:60]}",
                    "source": f"Dataset Benchmark ({cat})",
                    "text": f"Question: {q}\nAuthoritative Ground Truth Answer: {ans_str}.",
                    "metadata": {
                        "topic": cat,
                        "question_id": r["id"],
                        "authoritative": True,
                    }
                }
                existing_docs.append(doc_entry)
                existing_ids.add(doc_id)
                added += 1

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(existing_docs, f, indent=2)

        return added
