"""Utilities for git operations (commit, push, tag).

The file is not used at the moment, but is kept for future reference.
If you want to use it, you can uncomment the code in the cli.py file. (ingest function)
For this further adapts regarding SSH keys must be made.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def get_git_commit_hash(project_root: Path) -> Optional[str]:
    """Get the current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_git_repo(project_root: Path) -> bool:
    """Check if the project root is a git repository."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=project_root,
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_remote_url(project_root: Path) -> Optional[str]:
    """Get the remote URL for 'origin' if available."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_commit_and_tag(
    project_root: Path,
    run_id: str,
    push: bool = False,
) -> Optional[str]:
    """
    Commit current changes, create a tag with run_id, and optionally push.
    
    Args:
        project_root: Root directory of the project
        run_id: Identifier for the run (used as commit message and tag name)
        push: Whether to push commits and tags to remote
        
    Returns:
        Commit hash if successful, None otherwise
    """
    if not is_git_repo(project_root):
        return None
    
    try:
        # Check if there are any changes to commit
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        
        if not status_result.stdout.strip():
            # No changes to commit, but we can still create a tag if needed
            # Get current commit hash
            commit_hash = get_git_commit_hash(project_root)
            if commit_hash:
                # Create or update tag
                try:
                    subprocess.run(
                        ["git", "tag", "-a", run_id, "-m", f"Run snapshot: {run_id}"],
                        cwd=project_root,
                        check=True,
                    )
                except subprocess.CalledProcessError:
                    # Tag might already exist, try to update it
                    subprocess.run(
                        ["git", "tag", "-a", "-f", run_id, "-m", f"Run snapshot: {run_id}"],
                        cwd=project_root,
                        check=True,
                    )
                
                if push:
                    # Push tags
                    try:
                        subprocess.run(
                            ["git", "push", "origin", "--tags"],
                            cwd=project_root,
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                    except subprocess.CalledProcessError as e:
                        error_msg = e.stderr or e.stdout or "Unknown error"
                        if "Authentication failed" in error_msg or "Repository not found" in error_msg:
                            raise RuntimeError(
                                f"Git tag push authentication failed. See commit push error above for solutions.\n"
                                f"Error details: {error_msg}"
                            ) from e
                        raise RuntimeError(f"Git tag push failed: {error_msg}") from e
            
            return commit_hash
        
        # Stage all changes
        subprocess.run(
            ["git", "add", "."],
            cwd=project_root,
            check=True,
        )
        
        # Commit with run_id as message
        subprocess.run(
            ["git", "commit", "-m", run_id],
            cwd=project_root,
            check=True,
        )
        
        # Get the commit hash
        commit_hash = get_git_commit_hash(project_root)
        
        # Create tag with run_id
        try:
            subprocess.run(
                ["git", "tag", "-a", run_id, "-m", f"Run snapshot: {run_id}"],
                cwd=project_root,
                check=True,
            )
        except subprocess.CalledProcessError:
            # Tag might already exist, try to update it
            subprocess.run(
                ["git", "tag", "-a", "-f", run_id, "-m", f"Run snapshot: {run_id}"],
                cwd=project_root,
                check=True,
            )
        
        if push:
            # Push commits
            try:
                push_result = subprocess.run(
                    ["git", "push", "origin", "HEAD"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr or e.stdout or "Unknown error"
                # Check if it's an authentication error
                if "Authentication failed" in error_msg or "Repository not found" in error_msg:
                    remote_url = get_remote_url(project_root)
                    suggestions = []
                    
                    if remote_url and remote_url.startswith("https://"):
                        # Extract repo path from HTTPS URL
                        if "github.com/" in remote_url:
                            repo_path = remote_url.split("github.com/")[1].rstrip(".git")
                            ssh_url = f"git@github.com:{repo_path}.git"
                        else:
                            ssh_url = "git@github.com:USER/REPO.git"
                        suggestions.append(
                            f"Your remote uses HTTPS: {remote_url}\n"
                            f"  → Switch to SSH: git remote set-url origin {ssh_url}"
                        )
                    elif remote_url and remote_url.startswith("git@"):
                        suggestions.append(
                            f"Your remote uses SSH: {remote_url}\n"
                            f"  → Ensure SSH key is loaded: ssh-add ~/.ssh/id_rsa (or your key path)"
                        )
                    
                    suggestions.extend([
                        "  → Configure credential helper: git config --global credential.helper manager",
                        "  → Or use a personal access token in HTTPS URL",
                        "  → Or manually push after the commit: git push origin HEAD --tags"
                    ])
                    
                    raise RuntimeError(
                        f"Git push authentication failed.\n\n"
                        f"This happens because subprocess.run() doesn't have access to the same\n"
                        f"authentication context as manual git commands (SSH keys, credential managers).\n\n"
                        f"Solutions:\n" + "\n".join(suggestions) + f"\n\n"
                        f"Error details: {error_msg}"
                    ) from e
                raise RuntimeError(f"Git push failed: {error_msg}") from e
            
            # Push tags
            try:
                subprocess.run(
                    ["git", "push", "origin", "--tags"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr or e.stdout or "Unknown error"
                if "Authentication failed" in error_msg or "Repository not found" in error_msg:
                    raise RuntimeError(
                        f"Git tag push authentication failed. See commit push error above for solutions.\n"
                        f"Error details: {error_msg}"
                    ) from e
                raise RuntimeError(f"Git tag push failed: {error_msg}") from e
        
        return commit_hash
        
    except subprocess.CalledProcessError as e:
        # Git operation failed
        raise RuntimeError(f"Git operation failed: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("Git is not installed or not available in PATH")


def handle_git_snapshot(
    project_root: Path,
    run_id: str,
    choice: str,
) -> Optional[str]:
    """
    Handle git snapshot based on user choice.
    
    Args:
        project_root: Root directory of the project
        run_id: Identifier for the run
        choice: User choice ("1", "2", or "3")
        
    Returns:
        Commit hash if commit was made, None otherwise
    """
    if choice == "1":
        # Commit only
        return git_commit_and_tag(project_root, run_id, push=False)
    elif choice == "2":
        # Commit and push
        return git_commit_and_tag(project_root, run_id, push=True)
    elif choice == "3":
        # No commit, no push
        return None
    else:
        raise ValueError(f"Invalid choice: {choice}")

