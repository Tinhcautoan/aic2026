#!/usr/bin/env python
"""
Batch process 100 questions and generate submission.csv automatically.
Supports both TKIS (text-based) and VKIS (visual-based) queries.

Usage:
    python scripts/batch_process.py --questions /path/to/questions.csv --output submission.csv
    python scripts/batch_process.py --api http://localhost:8000 --questions questions.csv
"""

import argparse
import csv
import time
import requests
from pathlib import Path
from typing import List, Dict
import sys

def load_questions(csv_path: str) -> List[Dict]:
    """Load questions from CSV file."""
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'query_id': row.get('query_id', ''),
                'task': row.get('task', 'TKIS'),
                'text': row.get('text', ''),
                'query_image': row.get('query_image', ''),
            })
    return questions

def search_question(api_url: str, question: Dict, top_k: int = 100, timeout: int = 30) -> List[str]:
    """
    Run search for a single question.
    Returns list of answers in order: ["video.mp4,123", "video2.mp4,456", ...]
    """
    query_id = question.get('query_id', '')
    task_type = question.get('task', 'TKIS').upper()
    
    try:
        if task_type == 'TKIS':
            # Text-based query
            text_query = question.get('text', '').strip()
            if not text_query:
                print(f"    ⚠️  No text provided")
                return [''] * 100
            
            resp = requests.post(
                f"{api_url}/api/search?expand_neighbors=false&dedup=true",
                json={"query": text_query, "top_k": top_k},
                timeout=timeout
            )
        else:
            # Visual-based query (VKIS)
            image_path = question.get('query_image', '').strip()
            if not image_path or not Path(image_path).exists():
                print(f"    ⚠️  Image not found: {image_path}")
                return [''] * 100
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {'top_k': str(top_k)}
                resp = requests.post(
                    f"{api_url}/api/search-image",
                    files=files,
                    data=data,
                    timeout=timeout
                )
        
        if resp.status_code != 200:
            error_msg = resp.json().get('detail', f'HTTP {resp.status_code}')
            print(f"    ❌ API error: {error_msg}")
            return [''] * 100
        
        result = resp.json()
        answers = []
        for r in result.get('results', [])[:100]:
            video_id = r.get('video_id', '').strip()
            frame_idx = r.get('frame_idx', 0)
            if video_id:
                answers.append(f"{video_id},{frame_idx}")
        
        # Pad with empty strings if less than 100
        while len(answers) < 100:
            answers.append('')
        
        print(f"    ✓ Found {len([a for a in answers if a])} results")
        return answers[:100]
    
    except requests.exceptions.Timeout:
        print(f"    ❌ Timeout (took more than {timeout}s)")
        return [''] * 100
    except requests.exceptions.ConnectionError:
        print(f"    ❌ Connection error (API at {api_url} not reachable?)")
        return [''] * 100
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return [''] * 100

def main():
    parser = argparse.ArgumentParser(
        description='Batch process contest questions and generate submission.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/batch_process.py --questions questions.csv
  python scripts/batch_process.py --api http://localhost:8000 --questions questions.csv --output submission.csv
        """
    )
    parser.add_argument('--api', default='http://localhost:8000', help='API URL (default: http://localhost:8000)')
    parser.add_argument('--questions', required=True, help='Path to questions.csv')
    parser.add_argument('--output', default='submission.csv', help='Output CSV file (default: submission.csv)')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds (default: 30)')
    parser.add_argument('--top-k', type=int, default=100, help='Number of results per query (default: 100)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between requests in seconds (default: 0.5)')
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.questions).exists():
        print(f"❌ Questions file not found: {args.questions}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  📝 Batch Answer Submission")
    print(f"{'='*60}")
    print(f"  Input:  {args.questions}")
    print(f"  Output: {args.output}")
    print(f"  API:    {args.api}")
    print(f"  Top K:  {args.top_k}")
    print(f"{'='*60}\n")
    
    # Load questions
    print("📂 Loading questions...", end='', flush=True)
    questions = load_questions(args.questions)
    print(f" ✓ Loaded {len(questions)} questions\n")
    
    if len(questions) == 0:
        print("❌ No questions found in CSV file")
        sys.exit(1)
    
    # Process each question
    submission = {}
    start_time = time.time()
    
    for i, q in enumerate(questions, 1):
        query_id = q.get('query_id', f'Q_{i:04d}')
        task_type = q.get('task', 'TKIS').upper()
        
        print(f"[{i:3d}/{len(questions)}] {query_id} ({task_type})...", end='', flush=True)
        
        answers = search_question(args.api, q, top_k=args.top_k, timeout=args.timeout)
        submission[query_id] = answers
        
        if i < len(questions):
            time.sleep(args.delay)  # Rate limit
    
    elapsed = time.time() - start_time
    print(f"\n")
    
    # Export to CSV
    print("💾 Writing submission...", end='', flush=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header: query_id + answer_001 to answer_100
        header = ['query_id'] + [f'answer_{i:03d}' for i in range(1, 101)]
        writer.writerow(header)
        
        # Write each row
        for query_id in [q['query_id'] for q in questions]:
            ansList = submission.get(query_id, [''] * 100)
            row = [query_id]
            for ans in ansList:
                row.append(ans if ans else '')
            writer.writerow(row)
    
    print(f" ✓ Written\n")
    
    # Summary
    print(f"{'='*60}")
    print(f"  ✅ DONE")
    print(f"{'='*60}")
    total_queries = len(submission)
    total_answers = sum(len([a for a in ans if a]) for ans in submission.values())
    total_expected = total_queries * 100
    
    print(f"  Queries:      {total_queries}/100")
    print(f"  Answers:      {total_answers}/{total_expected}")
    print(f"  Output file:  {args.output}")
    print(f"  Total time:   {elapsed:.1f}s ({elapsed/len(questions):.1f}s per query)")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
