
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

from git import Repo, GitCommandError

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPO_DIR = DATA_DIR / "repository"


def validate_github_url(url: str) -> str:
    """Accept ordinary public GitHub repository URLs only."""
    parsed = urlparse(url.strip())

    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("Use an HTTPS URL from github.com.")

    parts = parsed.path.strip("/").split("/")

    if len(parts) != 2:
        raise ValueError("URL must look like https://github.com/owner/repo")

    owner, repo = parts

    if not re.fullmatch(r"[A-Za-z0-9-]+", owner):
        raise ValueError("Invalid GitHub owner.")

    repo = repo.removesuffix(".git")

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid repository name.")

    return f"https://github.com/{owner}/{repo}.git"


def clone_repository(url: str) -> Path:
    safe_url = validate_github_url(url)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # This first version indexes one repository at a time.
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)

    try:
        Repo.clone_from(
            safe_url,
            REPO_DIR,
            depth=1,
            multi_options=["--filter=blob:none"],
        )
    except GitCommandError as exc:
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
        raise RuntimeError(
            "Could not clone the repository. Check the URL and access."
        ) from exc

    print(f"Repository cloned to: {REPO_DIR}")
    return REPO_DIR


if __name__ == "__main__":
    repository_url = input("Enter a public GitHub repository URL: ").strip()

    try:
        path = clone_repository(repository_url)
        print(f"Success: {path}")
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
