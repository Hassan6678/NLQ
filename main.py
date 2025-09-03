#!/usr/bin/env python3
"""
Batch Question Processor for NLQ System

This script reads all questions from questions.txt, processes them through the NLQ system
using the SAME approach as Flask API, and saves the responses in JSON format.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from config import config
from helper.nlq_system import NLQSystem


# Global NLQ system instance (same as Flask API)
nlq_system: Optional[NLQSystem] = None


def get_nlq_system() -> NLQSystem:
    """Get or initialize the NLQ system (same as Flask API)."""
    global nlq_system
    if nlq_system is None:
        nlq_system = NLQSystem()
    return nlq_system


def ensure_data_loaded(dataset_path: str = 'data/llm_dataset_v11.gz', table_name: str = 'sales_data') -> None:
    """Startup data loader: DB first, gz fallback (same as Flask API)."""
    try:
        system = get_nlq_system()

        db_exists = os.path.exists(config.database.db_path)

        if db_exists and system.fast_attach_existing(table_name):
            print(f"✅ Using existing DB table '{table_name}'")
            return

        print("🔄 Loading dataset from gz into DB (first run or missing table)")
        system.load_data(str(Path(dataset_path).expanduser().resolve()), table_name)
        print("✅ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        raise


def read_questions_from_file(file_path: str) -> List[str]:
    """Read questions from questions.txt file, filtering out comments and empty lines."""
    questions = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Questions file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines, comments, and section headers
            if not line or line.startswith('#') or line.startswith('##'):
                continue

            # Extract question from numbered format (e.g., "1. What are the total sales...")
            if '. ' in line:
                # Remove the number prefix
                parts = line.split('. ', 1)
                if len(parts) == 2:
                    question = parts[1].strip()
                    questions.append(question)
            else:
                # Line without numbering
                questions.append(line)

    print(f"Loaded {len(questions)} questions from {file_path}")
    return questions


def process_question(question: str, question_id: int) -> Dict[str, Any]:
    """Process a single question using SAME approach as Flask API."""
    start_time = time.time()

    try:
        print(f"Processing question {question_id}: {question[:60]}{'...' if len(question) > 60 else ''}")

        system = get_nlq_system()
        result = system.query(question)

        # Ensure JSON-safe data (EXACT same as Flask API)
        try:
            data_records = result.data.replace([float('inf'), float('-inf')], pd.NA).where(lambda d: ~d.isna(), None).to_dict('records') if result.row_count > 0 else []
        except Exception:
            try:
                data_records = json.loads(result.data.to_json(orient='records')) if result.row_count > 0 else []
            except Exception:
                data_records = result.data.to_dict('records') if result.row_count > 0 else []

        # Structure the response (EXACT same as Flask API)
        response = {
            'success': True,
            'question': question,
            'answer': result.summary_text if result.row_count and result.summary_text else "No data found for the specified query.",
            'sql': result.sql_query,
            'row_count': result.row_count,
            'execution_time': getattr(result, 'execution_time', None),
            'data': data_records if hasattr(result, 'data') and result.row_count > 0 else [],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'question_id': question_id,
            'processing_time': time.time() - start_time
        }

        print(f"Question {question_id} completed successfully in {response['processing_time']:.2f}s")
        return response

    except Exception as e:
        print(f"Question {question_id} failed: {e}")

        return {
            'success': False,
            'question': question,
            'message': str(e),
            'question_id': question_id,
            'processing_time': time.time() - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }


def save_results_to_json(results: List[Dict[str, Any]], output_dir: str = "output") -> str:
    """Save all results to JSON files."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Save individual results
    for result in results:
        question_id = result['question_id']
        filename = f"question_{question_id:02d}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Save combined results
    combined_filepath = os.path.join(output_dir, "all_results.json")
    with open(combined_filepath, 'w', encoding='utf-8') as f:
        from datetime import datetime
        json.dump({
            "batch_info": {
                "total_questions": len(results),
                "successful": sum(1 for r in results if r['success']),
                "failed": sum(1 for r in results if not r['success']),
                "timestamp": datetime.now().isoformat(),
                "total_processing_time": sum(r.get('processing_time', 0) for r in results)
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} individual result files and combined results to {output_dir}")
    return combined_filepath


def main():
    """Main batch processing function using SAME approach as Flask API."""
    print("🚀 Starting NLQ Batch Question Processor")

    # Initialize system and load data (SAME as Flask API)
    try:
        if not config.validate():
            raise RuntimeError("Configuration validation failed")

        print("Initializing NLQ System...")
        get_nlq_system()
        print("✅ NLQ System initialized successfully")

        # Load data using SAME approach as Flask API
        ensure_data_loaded()

    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return 1

    # Read questions
    questions_file = "questions.txt"
    try:
        questions = read_questions_from_file(questions_file)
        if not questions:
            print("❌ No questions found in file")
            return 3
    except Exception as e:
        print(f"❌ Failed to read questions file: {e}")
        return 3

    # Process all questions
    print(f"🔄 Processing {len(questions)} questions...")

    results = []
    successful = 0
    failed = 0

    for i, question in enumerate(questions, 1):
        result = process_question(question, i)
        results.append(result)

        if result['success']:
            successful += 1
        else:
            failed += 1

        # Progress update every 5 questions
        if i % 5 == 0:
            print(f"Progress: {i}/{len(questions)} questions processed")

        # Small delay to prevent overwhelming the system
        time.sleep(0.1)

    # Save results
    try:
        output_file = save_results_to_json(results)
        print(f"✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return 4

    # Summary
    total_time = sum(r.get('processing_time', 0) for r in results)
    print("=" * 60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total Questions: {len(questions)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(".2f")
    print(".2f")
    print(f"Results saved to: output/")

    if failed > 0:
        print(f"⚠️  {failed} questions failed - check individual result files for details")

    print("🎉 Batch processing completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
