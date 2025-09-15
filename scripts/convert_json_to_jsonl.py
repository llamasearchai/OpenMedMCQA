import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/convert_json_to_jsonl.py input.json output.jsonl")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be an array of objects")
    with open(outp, "w", encoding="utf-8") as w:
        for obj in data:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


