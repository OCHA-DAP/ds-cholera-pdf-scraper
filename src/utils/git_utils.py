"""
Git utilities for retrieving repository information.
"""

import subprocess
from typing import Optional


def get_current_commit_hash(short: bool = True) -> Optional[str]:
    """
    Get the current git commit hash.

    Args:
        short: If True, return abbreviated 7-character hash. If False, return full hash.

    Returns:
        The commit hash string, or None if not in a git repository or on error.
    """
    try:
        if short:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_current_branch() -> Optional[str]:
    """
    Get the current git branch name.

    Returns:
        The branch name, or None if not in a git repository or on error.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_info() -> dict:
    """
    Get comprehensive git repository information.

    Returns:
        Dictionary with commit hash, branch, and other git info.
    """
    return {
        "commit_hash": get_current_commit_hash(short=True),
        "commit_hash_full": get_current_commit_hash(short=False),
        "branch": get_current_branch(),
    }


def is_git_repository() -> bool:
    """
    Check if the current directory is part of a git repository.

    Returns:
        True if in a git repository, False otherwise.
    """
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"], capture_output=True, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
