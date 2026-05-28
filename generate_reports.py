from __future__ import annotations

import argparse
import json
from pathlib import Path

from report_pages import save_strongreject_website


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the StrongReject results website from compiled StrongReject outputs."
    )
    parser.add_argument(
        "--compiled-dir",
        default="results/compiled_strongreject_results",
        help="Directory containing strongreject_summary.csv, strongreject_details.csv, and manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="website",
        help="Directory where index.html should be written.",
    )
    parser.add_argument(
        "--compile-first",
        action="store_true",
        help="Run the StrongReject compiler before generating the website.",
    )
    parser.add_argument("--cache-root", default="cache", help="Cache root used only with --compile-first.")
    parser.add_argument(
        "--max-detail-rows",
        type=int,
        default=200,
        help="Maximum detail rows to include in the static HTML sample.",
    )
    args = parser.parse_args()

    if args.compile_first:
        from compile_results import compile_cache_results

        manifest = compile_cache_results(
            cache_root=Path(args.cache_root),
            output_dir=Path(args.compiled_dir),
        )
        print(json.dumps({"compiled": manifest.get("outputs", {})}, indent=2, ensure_ascii=True))

    out_path = save_strongreject_website(
        compiled_dir=Path(args.compiled_dir),
        output_dir=Path(args.output_dir),
        max_detail_rows=args.max_detail_rows,
    )
    print(f"Saved StrongReject website: {out_path}")


if __name__ == "__main__":
    main()
