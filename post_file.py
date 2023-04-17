import argparse, requests
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Path to the source file", required=True)
    parser.add_argument("--url", help="URL of the destination", required=True)
    args = parser.parse_args()

    with open(args.source, "rb") as f:
        r = requests.post(args.url, files={"img": (Path(args.source).name, f)})
        print(r.status_code)