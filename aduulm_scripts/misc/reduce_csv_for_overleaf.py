#!/usr/bin/env python3
"""
Reduce a Monte-Carlo time-series CSV for use with LaTeX/PGFPlots.

Default behaviour:
- keeps the time/index column "k"
- keeps all aggregate mean columns ending in "_mean"
- removes q05/q50/q95 and other auxiliary columns
- keeps ALL time steps (no temporal downsampling)
- optionally shortens floating-point text representation

Usage:
    python reduce_csv_for_overleaf.py mc_kalman_sa_time_series.csv

Output:
    mc_kalman_sa_time_series_reduced.csv

Examples:
    # Preserve full numeric text precision:
    python reduce_csv_for_overleaf.py mc_kalman_sa_time_series.csv --no-round

    # Keep selected columns only:
    python reduce_csv_for_overleaf.py mc_kalman_sa_time_series.csv \
        --columns k d_norm_radial_mean d_norm_x_mean d_norm_y_mean d_norm_overall_mean

    # Custom output name:
    python reduce_csv_for_overleaf.py mc_kalman_sa_time_series.csv \
        --output plot_data.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_INDEX_COLUMNS = ("k", "time", "time_step", "t")


def format_number(value: str, significant_digits: int | None) -> str:
    """Shorten numeric strings without changing non-numeric cells."""
    if significant_digits is None:
        return value

    if value == "":
        return value

    try:
        number = float(value)
    except ValueError:
        return value

    if not math.isfinite(number):
        return value

    # Keep integer-looking values compact.
    if number.is_integer():
        return str(int(number))

    return f"{number:.{significant_digits}g}"


def choose_columns(
    fieldnames: list[str],
    explicit_columns: list[str] | None,
) -> list[str]:
    """Select columns for the reduced CSV."""
    if explicit_columns:
        missing = [name for name in explicit_columns if name not in fieldnames]
        if missing:
            raise ValueError(
                "Requested columns are missing from the input CSV:\n  "
                + "\n  ".join(missing)
            )
        return explicit_columns

    # Automatically retain the first conventional index/time column found.
    index_columns = [name for name in DEFAULT_INDEX_COLUMNS if name in fieldnames]
    if not index_columns:
        # Fall back to the first column so PGFPlots still has an x-axis candidate.
        index_columns = [fieldnames[0]]

    mean_columns = [
        name
        for name in fieldnames
        if name.endswith("_mean") and name not in index_columns
    ]

    if not mean_columns:
        raise ValueError(
            "No columns ending in '_mean' were found. "
            "Use --columns to specify the desired columns explicitly."
        )

    return index_columns[:1] + mean_columns


def reduce_csv(
    input_path: Path,
    output_path: Path,
    explicit_columns: list[str] | None,
    significant_digits: int | None,
) -> tuple[int, int, int]:
    """Write a reduced CSV and return (rows, input_columns, output_columns)."""
    with input_path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError("The input CSV does not contain a header row.")

        fieldnames = list(reader.fieldnames)
        keep_columns = choose_columns(fieldnames, explicit_columns)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=keep_columns,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()

            row_count = 0
            for row in reader:
                reduced_row = {
                    column: format_number(row.get(column, ""), significant_digits)
                    for column in keep_columns
                }
                writer.writerow(reduced_row)
                row_count += 1

    return row_count, len(fieldnames), len(keep_columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce a large time-series CSV to the columns normally required "
            "for LaTeX/PGFPlots."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <input_stem>_reduced.csv",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help=(
            "Explicit columns to keep. If omitted, the script keeps 'k' "
            "(or a similar index column) and every column ending in '_mean'."
        ),
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=10,
        help=(
            "Significant digits used for numeric output. Default: 10. "
            "This only reduces text size; use --no-round to preserve the "
            "original numeric strings exactly."
        ),
    )
    parser.add_argument(
        "--no-round",
        action="store_true",
        help="Do not shorten numeric values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input_csv
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_reduced{input_path.suffix}"
        )

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output path must be different.")

    significant_digits = None if args.no_round else args.digits

    rows, input_columns, output_columns = reduce_csv(
        input_path=input_path,
        output_path=output_path,
        explicit_columns=args.columns,
        significant_digits=significant_digits,
    )

    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    reduction = 100.0 * (1.0 - output_size / input_size) if input_size else 0.0

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows  : {rows}")
    print(f"Cols  : {input_columns} -> {output_columns}")
    print(
        f"Size  : {input_size / 1024:.1f} KiB -> "
        f"{output_size / 1024:.1f} KiB ({reduction:.1f}% smaller)"
    )


if __name__ == "__main__":
    main()
