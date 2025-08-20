#!/usr/bin/env python3
"""
Database Response Analysis CLI

Analyze stored responses in the existing prompt_logs.db to understand 
cost vs. output patterns and identify truncated responses.
"""

import argparse
import sqlite3
from pathlib import Path


def analyze_response_costs():
    """Analyze the cost vs success patterns in existing database."""
    
    db_path = Path("logs/prompts/prompt_logs.db")
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("💰 COST vs SUCCESS ANALYSIS")
    print("=" * 60)
    
    # Get overview stats
    cursor.execute("""
        SELECT 
            COUNT(*) as total_calls,
            SUM(CASE WHEN parsed_success = 1 THEN 1 ELSE 0 END) as successful,
            SUM(CASE WHEN parsed_success = 0 THEN 1 ELSE 0 END) as failed,
            SUM(CASE WHEN LENGTH(raw_response) > 1000 AND parsed_success = 0 AND raw_response NOT LIKE 'ERROR:%' THEN 1 ELSE 0 END) as large_failed
        FROM prompt_logs
    """)
    
    total, successful, failed, large_failed = cursor.fetchone()
    
    print(f"Total API Calls: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Failed with Large Response (>1K chars): {large_failed}")
    print(f"🚨 LIKELY CHARGED FOR FAILURES: {large_failed} calls")
    print()
    
    # Analyze by model
    print("📊 BY MODEL (showing response lengths for failures):")
    print("-" * 60)
    
    cursor.execute("""
        SELECT 
            model_name,
            COUNT(*) as attempts,
            SUM(CASE WHEN parsed_success = 1 THEN 1 ELSE 0 END) as successes,
            AVG(CASE WHEN parsed_success = 0 THEN LENGTH(raw_response) ELSE NULL END) as avg_failed_length,
            MAX(CASE WHEN parsed_success = 0 THEN LENGTH(raw_response) ELSE 0 END) as max_failed_length
        FROM prompt_logs 
        GROUP BY model_name 
        ORDER BY attempts DESC
    """)
    
    for row in cursor.fetchall():
        model, attempts, successes, avg_failed_len, max_failed_len = row
        success_rate = (successes / attempts * 100) if attempts > 0 else 0
        avg_failed_len = int(avg_failed_len) if avg_failed_len else 0
        
        status = "✅" if success_rate > 50 else "⚠️" if success_rate > 0 else "❌"
        print(f"{status} {model}")
        print(f"   {successes}/{attempts} success ({success_rate:.0f}%)")
        if avg_failed_len > 0:
            print(f"   Avg failed response: {avg_failed_len:,} chars (max: {max_failed_len:,})")
        print()
    
    # Show specific large failures (likely charged but failed)
    print("� LARGE FAILED RESPONSES (you were likely charged for these):")
    print("-" * 60)
    
    cursor.execute("""
        SELECT 
            id, model_name, prompt_version,
            LENGTH(raw_response) as response_length,
            parsing_errors,
            CASE 
                WHEN raw_response LIKE 'ERROR:%' THEN 'API_ERROR'
                WHEN LENGTH(raw_response) = 0 THEN 'EMPTY'
                WHEN raw_response LIKE '[%' OR raw_response LIKE '{%' THEN 'TRUNCATED_JSON'
                ELSE 'OTHER'
            END as failure_type
        FROM prompt_logs 
        WHERE parsed_success = 0 AND LENGTH(raw_response) > 1000
        ORDER BY LENGTH(raw_response) DESC
    """)
    
    large_failures = cursor.fetchall()
    for row in large_failures:
        id, model, version, length, error, failure_type = row
        print(f"💸 ID {id}: {model} v{version}")
        print(f"   Response: {length:,} characters ({failure_type})")
        print(f"   Error: {error[:80]}...")
        print()
    
    if not large_failures:
        print("✅ No large failed responses found")
    
    conn.close()


def show_response_content(response_id: int):
    """Show the full content of a specific response."""
    
    db_path = Path("logs/prompts/prompt_logs.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT model_name, prompt_version, parsed_success, 
               LENGTH(raw_response), raw_response, parsing_errors
        FROM prompt_logs WHERE id = ?
    """, (response_id,))
    
    result = cursor.fetchone()
    if not result:
        print(f"❌ No response found with ID {response_id}")
        return
        
    model, version, success, length, response, error = result
    
    print(f"📄 RESPONSE ID {response_id}")
    print("=" * 50)
    print(f"Model: {model} v{version}")
    print(f"Success: {'✅' if success else '❌'}")
    print(f"Length: {length:,} characters")
    
    if error:
        print(f"Error: {error}")
    
    print()
    print("CONTENT:")
    print("-" * 30)
    
    if length > 2000:
        print("FIRST 1000 CHARACTERS:")
        print(response[:1000])
        print("\n... (truncated) ...\n")
        print("LAST 1000 CHARACTERS:")
        print(response[-1000:])
    else:
        print(response)
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM response costs and failures")
    parser.add_argument(
        "--show-response",
        type=int,
        metavar="ID",
        help="Show full content of a specific response by ID",
    )

    args = parser.parse_args()

    if args.show_response:
        show_response_content(args.show_response)
    else:
        analyze_response_costs()


if __name__ == "__main__":
    main()
