#!/usr/bin/env python
"""
Generate submission.csv for the HCM AI Contest VideoQA challenge.

Reads test.csv (columns: id, video, ..., question), answers each question
through the local API's /api/vqa/answer (LLM-powered when Gemini is enabled),
and writes submission.csv in the contest format (id,answer).

Usage:
    python scripts/generate_vqa_submission.py
    python scripts/generate_vqa_submission.py --test test.csv --output submission.csv --api http://localhost:8000
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import httpx


def load_questions(test_csv: str) -> list:
    questions = []
    with open(test_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_id = row.get("id", "").strip()
            try:
                qid = int(raw_id)
            except ValueError:
                qid = raw_id
            questions.append({
                "id": qid,
                "video": row.get("video", "").strip(),
                "question": row.get("question", "").strip(),
            })
    return questions


async def answer_one(client: httpx.AsyncClient, api: str, item: dict, timeout: float) -> dict:
    last_err = ""
    for attempt in range(6):
        try:
            resp = await client.post(
                f"{api}/api/vqa/answer",
                json={"id": item["id"], "video": item["video"], "question": item["question"]},
                timeout=timeout,
            )
            if resp.status_code == 429:
                backoff = min(60 * 2 ** attempt, 300)
                print(f"    429 rate-limit, retrying in {backoff}s", flush=True)
                await asyncio.sleep(backoff)
                last_err = "429"
                continue
            if resp.status_code != 200:
                detail = ""
                try:
                    detail = resp.json().get("detail", "")
                except Exception:
                    pass
                return {"answer": "", "source": "error", "error": f"HTTP {resp.status_code}: {detail}"}
            data = resp.json()
            return {
                "answer": data.get("answer", "").strip(),
                "source": data.get("source", ""),
            }
        except Exception as e:
            last_err = str(e)
            backoff = min(30 * 2 ** attempt, 300)
            print(f"    error ({e}), retrying in {backoff}s", flush=True)
            await asyncio.sleep(backoff)
    return {"answer": "", "source": "error", "error": f"gave up: {last_err}"}


async def run(args, questions, existing):
    out_path = Path(args.output)
    start = time.time()
    stats = {"ok": 0, "empty": 0, "error": 0, "llm": 0, "metadata": 0}

    sem = asyncio.Semaphore(max(1, args.concurrency))
    done = 0
    total = len(questions)

    def _record(qid: int, res: dict):
        nonlocal done
        if res["source"] == "llm":
            stats["llm"] += 1
        elif res["source"] == "metadata":
            stats["metadata"] += 1
        if res["answer"]:
            stats["ok"] += 1
        else:
            stats["empty" if res["source"] != "error" else "error"] += 1
        done += 1
        tag = "LLM" if res["source"] == "llm" else (res["source"] or "?")
        err = f" [{res.get('error')}]" if res["source"] == "error" else ""
        print(f"[{done:3d}/{total}] id={qid} src={tag:8s} {res['answer'][:70]}{err}", flush=True)

    async with httpx.AsyncClient(base_url=args.api) as client:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "answer"])

            async def process(q: dict):
                qid = q["id"]
                async with sem:
                    if qid in existing:
                        res = {"answer": existing[qid], "source": "metadata"}
                        _record(qid, res)
                        return qid, res["answer"]
                    if not q["video"] or not q["question"]:
                        res = {"answer": "", "source": "empty"}
                        _record(qid, res)
                        return qid, res["answer"]
                    res = await answer_one(client, args.api, q, args.timeout)
                    _record(qid, res)
                    return qid, res["answer"]

            results = await asyncio.gather(*(process(q) for q in questions))

            # Write in original order
            by_id = {qid: ans for qid, ans in results}
            for q in questions:
                writer.writerow([q["id"], by_id[q["id"]]])
            f.flush()

    elapsed = time.time() - start
    print()
    print(f"DONE in {elapsed:.1f}s ({elapsed / max(total, 1):.1f}s/question)")
    print(f"  ok={stats['ok']}  llm={stats['llm']}  metadata={stats['metadata']}  empty={stats['empty']}  error={stats['error']}")
    print(f"Output: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Generate VQA submission.csv")
    parser.add_argument("--test", default="data/HCMAIContestVideoQA-public/test.csv")
    parser.add_argument("--output", default="data/HCMAIContestVideoQA-public/submission.csv")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--delay", type=float, default=0.6, help="Delay between requests (rate limit)")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel requests (default 2, safe for ~10 RPM free tier)")
    parser.add_argument("--resume", action="store_true", help="Keep existing answers in output file")
    args = parser.parse_args()

    questions = load_questions(args.test)
    if not questions:
        print(f"[ERROR] No questions in {args.test}")
        sys.exit(1)
    print(f"Loaded {len(questions)} questions from {args.test}")

    existing = {}
    if args.resume and Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("answer"):
                    try:
                        existing[int(row["id"])] = row.get("answer", "")
                    except ValueError:
                        existing[row["id"]] = row.get("answer", "")
        print(f"Resuming with {len(existing)} existing answers")

    asyncio.run(run(args, questions, existing))


if __name__ == "__main__":
    main()
