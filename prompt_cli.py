#!/usr/bin/env python3
"""
Command-line interface for managing prompts and viewing logs.
"""

import argparse
import json
import sys
from pathlib import Path

from src.config import Config
from src.prompt_logger import PromptLogger
from src.prompt_manager import PromptManager


def list_prompt_types(args):
    """List all available prompt types."""
    pm = PromptManager()
    prompt_types = pm.list_prompt_types()

    if not prompt_types:
        print("📭 No prompt types found")
        return

    print(f"📋 Available prompt types ({len(prompt_types)}):")
    for pt in prompt_types:
        versions = pm.list_versions(pt)
        current_version = None
        for v, meta in versions.items():
            if meta.get("is_current"):
                current_version = v
                break

        print(f"  🎯 {pt}")
        print(f"     Current version: {current_version or 'None set'}")
        print(f"     Total versions: {len(versions)}")


def list_versions(args):
    """List versions for a specific prompt type."""
    pm = PromptManager()

    if not args.prompt_type:
        print("❌ Please specify --prompt-type")
        return

    try:
        versions = pm.list_versions(args.prompt_type)
        if not versions:
            print(f"📭 No versions found for prompt type '{args.prompt_type}'")
            return

        print(f"📋 Versions for '{args.prompt_type}' ({len(versions)}):")

        # Sort by creation date
        sorted_versions = sorted(
            versions.items(), key=lambda x: x[1]["created_at"], reverse=True
        )

        for version, meta in sorted_versions:
            current_marker = "🎯 " if meta.get("is_current") else "   "
            print(f"{current_marker}{version}")
            print(f"     Created: {meta['created_at']}")
            print(f"     Description: {meta['description']}")

    except ValueError as e:
        print(f"❌ {e}")


def set_current(args):
    """Set the current version for a prompt type."""
    pm = PromptManager()

    if not args.prompt_type or not args.version:
        print("❌ Please specify both --prompt-type and --version")
        return

    try:
        pm.set_current_version(args.prompt_type, args.version)
        print(f"✅ Set {args.prompt_type} v{args.version} as current")
    except ValueError as e:
        print(f"❌ {e}")


def view_prompt(args):
    """View the content of a prompt version."""
    pm = PromptManager()

    if not args.prompt_type:
        print("❌ Please specify --prompt-type")
        return

    try:
        if args.version:
            prompt_data = pm.get_prompt_version(args.prompt_type, args.version)
        else:
            prompt_data = pm.get_current_prompt(args.prompt_type)

        print(f"📄 Prompt: {args.prompt_type} v{prompt_data['version']}")
        print(f"📅 Created: {prompt_data['created_at']}")
        print(f"📝 Description: {prompt_data['description']}")
        print("\n" + "=" * 60)
        print("SYSTEM PROMPT:")
        print("=" * 60)
        print(prompt_data["system_prompt"])
        print("\n" + "=" * 60)
        print("USER PROMPT TEMPLATE:")
        print("=" * 60)
        print(
            prompt_data["user_prompt_template"][:1000] + "..."
            if len(prompt_data["user_prompt_template"]) > 1000
            else prompt_data["user_prompt_template"]
        )

    except ValueError as e:
        print(f"❌ {e}")


def view_logs(args):
    """View logged LLM calls with accuracy metrics."""
    logger = PromptLogger()

    # Query logs with filters
    logs = logger.query_logs(
        prompt_type=args.prompt_type,
        prompt_version=args.version,
        parsed_success=args.success_only,
        limit=args.limit or 10,
    )

    if not logs:
        print("📭 No logs found matching the criteria")
        return

    print(f"📋 Found {len(logs)} logged calls:")
    print()

    for i, log in enumerate(logs, 1):
        status = "✅" if log.get("parsed_success") else "❌"
        records = log.get("records_extracted", 0)
        exec_time = log.get("execution_time_seconds")

        print(
            f"{status} Log {i}: {log.get('prompt_type')} v{log.get('prompt_version')}"
        )
        print(f"   📅 {log.get('timestamp')}")
        print(f"   🤖 Model: {log.get('model_name')}")
        print(f"   📊 Records: {records}")
        if exec_time:
            print(f"   ⏱️  Time: {exec_time:.2f}s")
        if log.get("parsing_errors"):
            print(f"   ⚠️  Error: {log.get('parsing_errors')}")

        # Check for accuracy metrics file
        call_id = log.get("id", "unknown")
        accuracy_file = (
            Config.LOGS_DIR / "accuracy" / f"evaluation_{call_id}_metrics.json"
        )

        if accuracy_file.exists():
            try:
                import json

                with open(accuracy_file, "r") as f:
                    accuracy_data = json.load(f)
                print(
                    f"   🎯 Accuracy: Coverage={accuracy_data.get('coverage_rate', 'N/A')}%, "
                    f"Precision={accuracy_data.get('precision_rate', 'N/A')}%, "
                    f"Overall={accuracy_data.get('overall_accuracy', 'N/A')}%"
                )
                print(
                    f"   🏆 Composite Score: {accuracy_data.get('composite_score', 'N/A')}%"
                )
            except Exception:
                print("   📊 Accuracy data available but failed to load")

        print()


def performance_summary(args):
    """Show performance summary for prompt versions."""
    logger = PromptLogger()

    summary = logger.get_performance_summary(
        prompt_type=args.prompt_type, prompt_version=args.version
    )

    if "message" in summary:
        print(f"📭 {summary['message']}")
        return

    print("📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Prompt Type: {summary['prompt_type'] or 'All'}")
    print(f"Version: {summary['prompt_version'] or 'All'}")
    print(f"Total Calls: {summary['total_calls']}")
    print(f"Success Rate: {summary['parsing_success_rate']}%")
    print(f"Total Records Extracted: {summary['total_records_extracted']}")
    print(f"Avg Records/Call: {summary['avg_records_per_call']}")
    if summary["avg_execution_time_seconds"]:
        print(f"Avg Execution Time: {summary['avg_execution_time_seconds']}s")

    date_range = summary["date_range"]
    print(f"Date Range: {date_range['earliest']} to {date_range['latest']}")


def export_to_markdown(args):
    """Export a prompt to markdown format for editing."""
    pm = PromptManager()

    try:
        output_path = args.output or f"{args.prompt_type}_{args.version}.md"
        result_path = pm.export_to_markdown(args.prompt_type, args.version, output_path)
        print(f"✅ Exported to: {result_path}")
        print(
            f"💡 Edit the markdown file and use 'import-from-markdown' to create a new version"
        )
    except ValueError as e:
        print(f"❌ {e}")


def import_from_markdown(args):
    """Import a prompt from markdown file to create new version."""
    pm = PromptManager()

    try:
        result_path = pm.create_prompt_from_markdown(
            args.prompt_type, args.markdown_file
        )
        print(f"✅ Created prompt from markdown: {result_path}")

        # Parse the file to get version info
        with open(args.markdown_file, "r") as f:
            content = f.read()

        # Extract version from frontmatter
        lines = content.split("\n")
        version = None
        for line in lines[1:10]:  # Check first few lines for version
            if line.strip().startswith("version:"):
                version = line.split(":", 1)[1].strip()
                break

        if version:
            # Ask if user wants to set as current
            response = input(
                f"Set {args.prompt_type} v{version} as current version? (y/N): "
            )
            if response.lower() in ["y", "yes"]:
                pm.set_current_version(args.prompt_type, version)
                print(f"✅ Set v{version} as current version")

    except Exception as e:
        print(f"❌ Error importing markdown: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage prompts and view logs")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List prompt types
    list_types_parser = subparsers.add_parser(
        "list-types", help="List all prompt types"
    )
    list_types_parser.set_defaults(func=list_prompt_types)

    # List versions
    list_versions_parser = subparsers.add_parser(
        "list-versions", help="List versions for a prompt type"
    )
    list_versions_parser.add_argument(
        "--prompt-type", required=True, help="Prompt type name"
    )
    list_versions_parser.set_defaults(func=list_versions)

    # Set current version
    set_current_parser = subparsers.add_parser(
        "set-current", help="Set current version"
    )
    set_current_parser.add_argument(
        "--prompt-type", required=True, help="Prompt type name"
    )
    set_current_parser.add_argument(
        "--version", required=True, help="Version to set as current"
    )
    set_current_parser.set_defaults(func=set_current)

    # View prompt
    view_prompt_parser = subparsers.add_parser(
        "view-prompt", help="View prompt content"
    )
    view_prompt_parser.add_argument(
        "--prompt-type", required=True, help="Prompt type name"
    )
    view_prompt_parser.add_argument(
        "--version", help="Specific version (default: current)"
    )
    view_prompt_parser.set_defaults(func=view_prompt)

    # Export to markdown
    export_md_parser = subparsers.add_parser(
        "export-to-markdown", help="Export prompt to markdown for editing"
    )
    export_md_parser.add_argument("--prompt-type", required=True, help="Prompt type")
    export_md_parser.add_argument("--version", required=True, help="Version to export")
    export_md_parser.add_argument("--output", help="Output markdown file path")
    export_md_parser.set_defaults(func=export_to_markdown)

    # Import from markdown
    import_md_parser = subparsers.add_parser(
        "import-from-markdown", help="Import prompt from markdown file"
    )
    import_md_parser.add_argument("--prompt-type", required=True, help="Prompt type")
    import_md_parser.add_argument(
        "--markdown-file", required=True, help="Markdown file to import"
    )
    import_md_parser.set_defaults(func=import_from_markdown)

    # View logs
    view_logs_parser = subparsers.add_parser("logs", help="View logged LLM calls")
    view_logs_parser.add_argument("--prompt-type", help="Filter by prompt type")
    view_logs_parser.add_argument("--version", help="Filter by version")
    view_logs_parser.add_argument(
        "--success-only", action="store_true", help="Show only successful calls"
    )
    view_logs_parser.add_argument("--limit", type=int, help="Limit number of results")
    view_logs_parser.set_defaults(func=view_logs)

    # Performance summary
    perf_parser = subparsers.add_parser("performance", help="Show performance summary")
    perf_parser.add_argument("--prompt-type", help="Filter by prompt type")
    perf_parser.add_argument("--version", help="Filter by version")
    perf_parser.set_defaults(func=performance_summary)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
