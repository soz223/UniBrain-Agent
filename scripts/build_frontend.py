import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unibrain_agent.frontend import document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist")
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text(document(standalone=True), encoding="utf-8")
    (folder / ".nojekyll").touch()
    print(f"Static frontend written to {folder}")


if __name__ == "__main__":
    main()
