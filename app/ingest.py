
import json
from pathlib import Path
from git import Repo

DATA_DIR = Path("data")
REPO_DIR = DATA_DIR / "repository"
OUTPUT_FILE = DATA_DIR / "documents.jsonl"

REPO_URL = "https://github.com/pallets/itsdangerous.git"

ALLOWED_EXTENSIONS = {
    ".py", ".md", ".rst", ".txt", ".toml",
    ".yaml", ".yml", ".json", ".html", ".js",
    ".ts", ".vue", ".sql", ".sh"
}

MAX_FILE_SIZE = 500_000  # 500 KB per file


def main():
    DATA_DIR.mkdir(exist_ok=True)

    
    # Use the repository that has already been cloned
    if not REPO_DIR.exists() or not (REPO_DIR / ".git").exists():
        raise FileNotFoundError(
            "Repository not found. Run app/index_repo.py first."
        )


    repo = Repo(REPO_DIR)
    commit_hash = repo.head.commit.hexsha

    count = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for relative_path in repo.git.ls_files().splitlines():
            file_path = REPO_DIR / relative_path

            if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            if not file_path.is_file():
                continue

            if file_path.stat().st_size > MAX_FILE_SIZE:
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            if not content.strip():
                continue

            document = {
                "file_path": relative_path,
                "content": content,
                "commit": commit_hash,
                "start_line": 1,
                "end_line": len(content.splitlines())
            }

            output.write(json.dumps(document) + "\n")
            count += 1

    print(f"Repository: {repo.remotes.origin.url}")
    print(f"Commit: {commit_hash}")
    print(f"Files indexed: {count}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
