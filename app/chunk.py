
import ast
import json
from pathlib import Path

INPUT_FILE = Path("data/documents.jsonl")
OUTPUT_FILE = Path("data/chunks.jsonl")


def extract_chunks(document):
    source = document["content"]

    try:
        syntax_tree = ast.parse(source)
    except SyntaxError:
        return []

    chunks = []

    def visit(node, parents=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            name = node.name
            qualified_name = ".".join((*parents, name))
            kind = "class" if isinstance(node, ast.ClassDef) else "function"

            code = ast.get_source_segment(source, node)

            if code:
                chunks.append({
                    "file_path": document["file_path"],
                    "name": qualified_name,
                    "type": kind,
                    "content": code,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "commit": document.get("commit")
                })

            # Visit methods, nested classes, and nested functions.
            for child in ast.iter_child_nodes(node):
                visit(child, (*parents, name))
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, parents)

    for node in syntax_tree.body:
        visit(node)

    return chunks


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_files = 0
    total_chunks = 0

    # Write to a temporary file so a failed run doesn't
    # leave a partially written output file.
    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    try:
        with INPUT_FILE.open(encoding="utf-8") as source_file, \
                temp_file.open("w", encoding="utf-8") as output_file:

            for line in source_file:
                document = json.loads(line)
                total_files += 1

                if Path(document["file_path"]).suffix != ".py":
                    continue

                for chunk in extract_chunks(document):
                    output_file.write(
                        json.dumps(chunk, ensure_ascii=False) + "\n"
                    )
                    total_chunks += 1

        temp_file.replace(OUTPUT_FILE)

    finally:
        if temp_file.exists():
            temp_file.unlink()

    print(f"Files processed: {total_files}")
    print(f"Code chunks created: {total_chunks}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
