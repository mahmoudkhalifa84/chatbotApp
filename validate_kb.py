"""Validate documents.json before committing/deploying knowledge base changes.

Usage: python validate_kb.py [path/to/documents.json]
"""
import json
import sys


def validate(path):
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must contain a non-empty JSON array of entries")

    seen_ids = set()
    categories = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} is not a JSON object: {entry!r}")

        entry_id = entry.get("id", "").strip()
        text = entry.get("text", "").strip()
        category = entry.get("category", "").strip() or "uncategorized"

        if not entry_id:
            raise ValueError(f"Entry {i} is missing a non-empty 'id'")
        if not text:
            raise ValueError(f"Entry '{entry_id}' is missing non-empty 'text'")
        if entry_id in seen_ids:
            raise ValueError(f"Duplicate entry id '{entry_id}'")

        seen_ids.add(entry_id)
        categories.setdefault(category, []).append(entry_id)

    return categories


if __name__ == "__main__":
    kb_path = sys.argv[1] if len(sys.argv) > 1 else "documents.json"
    try:
        categories = validate(kb_path)
    except (ValueError, json.JSONDecodeError, OSError) as e:
        print(f"INVALID: {e}")
        sys.exit(1)

    total = sum(len(ids) for ids in categories.values())
    print(f"OK: {total} entries in {kb_path}")
    for category, ids in sorted(categories.items()):
        print(f"  {category} ({len(ids)}): {', '.join(ids)}")
