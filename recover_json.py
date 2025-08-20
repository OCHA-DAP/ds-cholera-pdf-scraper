#!/usr/bin/env python3
"""
JSON Recovery CLI

Recover truncated or malformed JSON responses from failed API calls
without needing to make new API requests.
"""

import argparse
import json
from pathlib import Path

from src.json_recovery import JSONRecovery


def main():
    parser = argparse.ArgumentParser(
        description="Recover JSON from failed LLM responses"
    )
    parser.add_argument(
        "--recover-id",
        type=int,
        metavar="ID",
        help="Recover a specific response by database ID",
    )
    parser.add_argument(
        "--recover-all",
        action="store_true",
        help="Attempt to recover all failed large responses",
    )
    parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Save recovered data back to database as new entries",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        metavar="PATH",
        help="Export recovered data to CSV file",
    )

    args = parser.parse_args()

    if not args.recover_id and not args.recover_all:
        parser.print_help()
        return

    recovery = JSONRecovery()

    if args.recover_id:
        print(f"🔧 ATTEMPTING RECOVERY FOR RESPONSE ID {args.recover_id}")
        print("=" * 60)

        result = recovery.recover_failed_response(args.recover_id)
        if not result:
            print(f"❌ No failed response found with ID {args.recover_id}")
            return

        print(f"Model: {result['model_name']}")
        print(f"Prompt Version: {result['prompt_version']}")
        print(f"Original Error: {result['original_error']}")
        print(f"Raw Response Length: {result['raw_response_length']:,} chars")
        print()

        if result["recovery_success"]:
            print(f"✅ RECOVERY SUCCESSFUL!")
            print(f"Method: {result['recovery_method']}")
            print(f"Records Recovered: {result['recovered_records']}")

            if args.save_to_db:
                success = recovery.update_database_with_recovery(
                    args.recover_id, result["recovered_data"]
                )
                if success:
                    print("💾 Saved recovered data to database")
                else:
                    print("❌ Failed to save to database")

            if args.export_csv:
                import pandas as pd

                df = pd.DataFrame(result["recovered_data"])
                df.to_csv(args.export_csv, index=False)
                print(f"📊 Exported to {args.export_csv}")

        else:
            print(f"❌ Recovery failed: {result['recovery_method']}")

    elif args.recover_all:
        print("🔧 ATTEMPTING RECOVERY FOR ALL FAILED LARGE RESPONSES")
        print("=" * 70)

        results = recovery.recover_all_failed_large_responses()

        if not results:
            print("✅ No large failed responses found to recover")
            return

        successful_recoveries = []

        for result in results:
            print(f"ID {result['response_id']}: {result['model_name']}")

            if result["recovery_success"]:
                print(
                    f"   ✅ Recovered {result['recovered_records']} records "
                    f"({result['recovery_method']})"
                )
                successful_recoveries.append(result)

                if args.save_to_db:
                    recovery.update_database_with_recovery(
                        result["response_id"], result["recovered_data"]
                    )
            else:
                print(f"   ❌ Failed ({result['recovery_method']})")

        print()
        print(f"📊 SUMMARY:")
        print(f"Total attempts: {len(results)}")
        print(f"Successful recoveries: {len(successful_recoveries)}")
        print(f"Success rate: {len(successful_recoveries)/len(results)*100:.1f}%")

        if successful_recoveries and args.export_csv:
            # Combine all recovered data
            all_data = []
            for result in successful_recoveries:
                for record in result["recovered_data"]:
                    record["_recovery_source_id"] = result["response_id"]
                    record["_recovery_method"] = result["recovery_method"]
                    all_data.append(record)

            import pandas as pd

            df = pd.DataFrame(all_data)
            df.to_csv(args.export_csv, index=False)
            print(f"📊 Exported {len(all_data)} recovered records to {args.export_csv}")


if __name__ == "__main__":
    main()
