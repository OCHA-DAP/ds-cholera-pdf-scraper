"""
Enhanced Prompt Management System with Markdown support.
Handles versioning, loading, and management of prompt templates.
Supports both JSON and Markdown formats for easy editing.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class PromptManager:
    """
    Manages prompt templates with versioning and metadata.
    Supports both JSON and Markdown formats.
    """

    def __init__(self, prompts_dir: str = None):
        """Initialize PromptManager with prompts directory."""
        if prompts_dir is None:
            # Default to prompts directory in project root
            project_root = Path(__file__).parent.parent
            prompts_dir = project_root / "prompts"

        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(exist_ok=True)

        # Metadata file for tracking versions
        self.metadata_file = self.prompts_dir / "prompt_metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load prompt metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save prompt metadata to file."""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _parse_markdown_prompt(self, md_content: str) -> Dict[str, Any]:
        """
        Parse a markdown prompt file with YAML frontmatter.

        Expected format:
        ---
        version: v2.2.0
        description: Enhanced with auto-detection
        ---

        # System Prompt

        Your system prompt content here...

        # User Prompt Template

        Your user prompt template here...

        ## Examples (optional)

        Optional examples section...
        """
        lines = md_content.strip().split("\n")

        # Parse YAML frontmatter
        if not (lines[0].strip() == "---"):
            raise ValueError("Markdown prompt must start with YAML frontmatter (---)")

        frontmatter_end = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                frontmatter_end = i
                break

        if frontmatter_end == -1:
            raise ValueError("Markdown prompt must end YAML frontmatter with ---")

        # Parse YAML frontmatter manually (simple key: value format)
        metadata = {}
        for line in lines[1:frontmatter_end]:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

        # Parse content sections
        content_lines = lines[frontmatter_end + 1 :]
        content = "\n".join(content_lines)

        # Extract sections using headers - preserve ALL content
        system_prompt = ""
        user_prompt_template = ""
        examples = None

        # Find System Prompt section
        system_match = re.search(r'\n# System Prompt\s*\n(.*?)(?=\n# [^#]|\Z)', content, re.DOTALL)
        if system_match:
            system_prompt = system_match.group(1).strip()

        # Find User Prompt Template section - preserve EVERYTHING after the header
        user_match = re.search(r'\n# User Prompt Template\s*\n(.*?)(?=\n# [^#]|\Z)', content, re.DOTALL)
        if user_match:
            user_prompt_template = user_match.group(1).strip()
            
            # Also try to extract examples for the separate examples field (optional)
            examples_matches = re.findall(r'\n## Examples[^\n]*\n(.*?)(?=\n## [^#]|\n# |\Z)', user_prompt_template, re.DOTALL)
            if examples_matches:
                examples = '\n'.join(examples_matches).strip()

        # Build result with core fields
        result = {
            "version": metadata.get("version", "unknown"),
            "created_at": metadata.get("created_at", datetime.now().isoformat()),
            "description": metadata.get("description", ""),
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
            "examples": examples,
        }

        # Add any additional metadata fields from YAML frontmatter
        for key, value in metadata.items():
            if key not in result:  # Don't override core fields
                result[key] = value

        return result

    def create_prompt_from_markdown(self, prompt_type: str, md_file_path: str) -> str:
        """
        Create a prompt version from a markdown file.

        Args:
            prompt_type: Type identifier (e.g., 'health_data_extraction')
            md_file_path: Path to the markdown file

        Returns:
            str: Path to the created JSON prompt file
        """
        # Read markdown file
        with open(md_file_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Parse markdown
        prompt_data = self._parse_markdown_prompt(md_content)

        # Create prompt directory for this type
        prompt_type_dir = self.prompts_dir / prompt_type
        prompt_type_dir.mkdir(exist_ok=True)

        # Create JSON prompt file
        version = prompt_data["version"]
        prompt_filename = f"{prompt_type}_{version}.json"
        prompt_path = prompt_type_dir / prompt_filename

        # Save JSON file
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)

        # Update metadata
        if prompt_type not in self.metadata:
            self.metadata[prompt_type] = {}

        # Convert to relative paths from project root
        project_root = Path(__file__).parent.parent
        try:
            relative_prompt_path = prompt_path.relative_to(project_root)
        except ValueError:
            # If path is not relative to project root, use as-is
            relative_prompt_path = prompt_path

        try:
            md_path = Path(md_file_path)
            relative_md_path = md_path.relative_to(project_root)
        except ValueError:
            # If path is not relative to project root, use as-is
            relative_md_path = Path(md_file_path)

        self.metadata[prompt_type][version] = {
            "created_at": prompt_data["created_at"],
            "description": prompt_data["description"],
            "file_path": str(relative_prompt_path),
            "source_markdown": str(relative_md_path),
            "is_current": False,
        }

        self._save_metadata()

        return str(prompt_path)

    def export_to_markdown(
        self, prompt_type: str, version: str, output_path: str
    ) -> str:
        """
        Export a JSON prompt to markdown format for editing.

        Args:
            prompt_type: Type identifier
            version: Version identifier
            output_path: Where to save the markdown file

        Returns:
            str: Path to the created markdown file
        """
        prompt_data = self.get_prompt_version(prompt_type, version)

        # Build markdown content
        md_content = f"""---
version: {prompt_data['version']}
description: {prompt_data['description']}
created_at: {prompt_data['created_at']}
---

# System Prompt

{prompt_data['system_prompt']}

# User Prompt Template

{prompt_data['user_prompt_template']}"""

        if prompt_data.get("examples"):
            md_content += f"""

## Examples

{prompt_data['examples']}"""

        # Save markdown file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return output_path

    def create_prompt_version(
        self,
        prompt_type: str,
        version: str,
        system_prompt: str,
        user_prompt_template: str,
        description: str,
        examples: Optional[str] = None,
    ) -> str:
        """
        Create a new version of a prompt template.

        Args:
            prompt_type: Type identifier (e.g., 'health_data_extraction')
            version: Version identifier (e.g., 'v1.0.0')
            system_prompt: System message content
            user_prompt_template: User prompt template with placeholders
            description: Brief description of this version
            examples: Optional examples/few-shot content

        Returns:
            str: Path to the created prompt file
        """
        # Create prompt directory for this type
        prompt_type_dir = self.prompts_dir / prompt_type
        prompt_type_dir.mkdir(exist_ok=True)

        # Create prompt file
        prompt_filename = f"{prompt_type}_{version}.json"
        prompt_path = prompt_type_dir / prompt_filename

        # Prompt content structure
        prompt_content = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
            "examples": examples,
        }

        # Save prompt file
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(prompt_content, f, indent=2, ensure_ascii=False)

        # Update metadata
        if prompt_type not in self.metadata:
            self.metadata[prompt_type] = {}

        # Convert to relative path from project root
        project_root = Path(__file__).parent.parent
        try:
            relative_prompt_path = prompt_path.relative_to(project_root)
        except ValueError:
            # If path is not relative to project root, use as-is
            relative_prompt_path = prompt_path

        self.metadata[prompt_type][version] = {
            "created_at": prompt_content["created_at"],
            "description": description,
            "file_path": str(relative_prompt_path),
            "is_current": False,
        }

        self._save_metadata()

        return str(prompt_path)

    def set_current_version(self, prompt_type: str, version: str):
        """Set the current version for a prompt type."""
        if prompt_type not in self.metadata:
            raise ValueError(f"Prompt type '{prompt_type}' not found")

        if version not in self.metadata[prompt_type]:
            raise ValueError(
                f"Version '{version}' not found for prompt type '{prompt_type}'"
            )

        # Clear current flag from all versions
        for v in self.metadata[prompt_type]:
            self.metadata[prompt_type][v]["is_current"] = False

        # Set current version
        self.metadata[prompt_type][version]["is_current"] = True
        self._save_metadata()

    def get_current_prompt(self, prompt_type: str) -> Dict[str, Any]:
        """Get the current version of a prompt type."""
        if prompt_type not in self.metadata:
            raise ValueError(f"Prompt type '{prompt_type}' not found")

        # Find current version
        current_version = None
        for version, metadata in self.metadata[prompt_type].items():
            if metadata.get("is_current"):
                current_version = version
                break

        # If no current version set, use the latest
        if current_version is None:
            versions = list(self.metadata[prompt_type].keys())
            if not versions:
                raise ValueError(f"No versions found for prompt type '{prompt_type}'")

            current_version = sorted(versions)[-1]
            self.set_current_version(prompt_type, current_version)

        return self.get_prompt_version(prompt_type, current_version)

    def get_prompt_version(self, prompt_type: str, version: str) -> Dict[str, Any]:
        """Get a specific version of a prompt."""
        if prompt_type not in self.metadata:
            raise ValueError(f"Prompt type '{prompt_type}' not found")

        if version not in self.metadata[prompt_type]:
            raise ValueError(
                f"Version '{version}' not found for prompt type '{prompt_type}'"
            )

        # Load prompt file
        file_path = self.metadata[prompt_type][version]["file_path"]
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_content = json.load(f)

        return prompt_content

    def list_versions(self, prompt_type: str) -> Dict[str, Dict[str, Any]]:
        """List all versions for a prompt type."""
        if prompt_type not in self.metadata:
            return {}

        return self.metadata[prompt_type]

    def list_prompt_types(self) -> list:
        """List all available prompt types."""
        return list(self.metadata.keys())

    def build_prompt(self, prompt_type: str, version: str = None, **kwargs) -> tuple:
        """
        Build a complete prompt from template with variable substitution.

        Args:
            prompt_type: Type of prompt to build
            version: Specific version to use (if None, uses current version)
            **kwargs: Variables to substitute in the template

        Returns:
            tuple: (system_prompt, user_prompt, metadata, template_for_logging)
        """
        if version:
            prompt_data = self.get_prompt_version(prompt_type, version)
        else:
            prompt_data = self.get_current_prompt(prompt_type)

        # Substitute variables in user prompt template
        user_prompt = prompt_data["user_prompt_template"].format(**kwargs)

        # Create a version for logging that replaces large content with placeholders
        template_for_logging = prompt_data["user_prompt_template"]
        for key, value in kwargs.items():
            if isinstance(value, str) and len(value) > 1000:  # Large content threshold
                # Replace large content with a summary for logging
                summary = f"[{key.upper()}_CONTENT: {len(value)} chars, preview: {value[:100]}...]"
                template_for_logging = template_for_logging.replace(
                    f"{{{key}}}", summary
                )
            else:
                template_for_logging = template_for_logging.replace(
                    f"{{{key}}}", str(value)
                )

        # Prepare metadata for logging
        metadata = {
            "prompt_type": prompt_type,
            "version": prompt_data["version"],
            "description": prompt_data["description"],
            "created_at": prompt_data["created_at"],
        }

        # Add preprocessor field if specified in prompt
        if "preprocessor" in prompt_data:
            metadata["preprocessor"] = prompt_data["preprocessor"]

        return (
            prompt_data["system_prompt"],
            user_prompt,
            metadata,
            template_for_logging,
        )
