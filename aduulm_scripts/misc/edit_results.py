import json
import csv
from pathlib import Path


def export_results_table_custom(
    results_or_path,
    output_dir="results",
    filename_prefix="gof_results_custom",
    columns=None,
    column_labels=None,
    decimals=3,
    include_name=True,
    sort_by=None,
    reverse=False,
    latex=True,
    csv_file=True,
    json_file=True,
):
    """
    Export a customized results table after the evaluation.

    Parameters
    ----------
    results_or_path : dict or str or Path
        Either the results dictionary returned by the evaluation function
        or a path to a saved JSON results file.

    output_dir : str or Path
        Directory where the exported files are stored.

    filename_prefix : str
        Prefix for output filenames.

    columns : list[str] or None
        Result keys to include as table columns.
        Example:
            ["power", "mean_p_ok", "std_p_ok"]
        If None, a default compact set is used.

    column_labels : dict[str, str] or None
        Optional mapping from result keys to displayed labels.
        Example:
            {"power": "Power", "mean_p_ok": "Mean $P_{OK}$"}

    decimals : int or dict[str, int]
        Number of decimals for numeric values.
        Can be a single int or a dict per column.

    include_name : bool
        Whether to include the distribution/scenario name as first column.

    sort_by : str or None
        Optional result key to sort rows by.

    reverse : bool
        If True, sort descending.

    latex : bool
        Export LaTeX table rows.

    csv_file : bool
        Export CSV file.

    json_file : bool
        Export filtered JSON file.

    Returns
    -------
    dict
        Paths of generated files.
    """

    # -----------------------------
    # Load results if path is given
    # -----------------------------
    if isinstance(results_or_path, (str, Path)):
        with open(results_or_path, "r") as f:
            results = json.load(f)
    else:
        results = results_or_path

    if columns is None:
        columns = [
            "power",
            "mean_p_ok",
            "std_p_ok",
            "tau",
            "W",
            "n",
            "alpha",
        ]

    if column_labels is None:
        column_labels = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Prepare rows
    # -----------------------------
    items = list(results.items())

    if sort_by is not None:
        items = sorted(
            items,
            key=lambda item: item[1].get(sort_by, float("nan")),
            reverse=reverse,
        )

    rows = []

    for name, res in items:
        row = {}

        if include_name:
            row["name"] = name

        for col in columns:
            row[col] = res.get(col, None)

        rows.append(row)

    # -----------------------------
    # Formatting helper
    # -----------------------------
    def get_decimals(col):
        if isinstance(decimals, dict):
            return decimals.get(col, 3)
        return decimals

    def format_value(value, col):
        if value is None:
            return ""

        if isinstance(value, float):
            return f"{value:.{get_decimals(col)}f}"

        if isinstance(value, int):
            return str(value)

        return str(value)

    paths = {}

    # -----------------------------
    # Filtered JSON
    # -----------------------------
    if json_file:
        json_path = output_dir / f"{filename_prefix}.json"

        filtered = {}
        for row in rows:
            name = row["name"] if include_name else str(len(filtered))
            filtered[name] = {
                col: row[col]
                for col in columns
            }

        with open(json_path, "w") as f:
            json.dump(filtered, f, indent=2)

        paths["json"] = str(json_path)

    # -----------------------------
    # CSV
    # -----------------------------
    if csv_file:
        csv_path = output_dir / f"{filename_prefix}.csv"

        fieldnames = []
        if include_name:
            fieldnames.append("name")
        fieldnames.extend(columns)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                writer.writerow(row)

        paths["csv"] = str(csv_path)

    # -----------------------------
    # LaTeX rows only
    # -----------------------------
    if latex:
        tex_path = output_dir / f"{filename_prefix}.tex"

        with open(tex_path, "w") as f:
            f.write("% Automatically generated custom result table rows\n")

            if include_name:
                header_cols = ["Distribution"] + [
                    column_labels.get(col, col)
                    for col in columns
                ]
            else:
                header_cols = [
                    column_labels.get(col, col)
                    for col in columns
                ]

            f.write("% " + " & ".join(header_cols) + " \\\\\n")

            for row in rows:
                vals = []

                if include_name:
                    vals.append(str(row["name"]))

                for col in columns:
                    vals.append(format_value(row[col], col))

                f.write(" & ".join(vals) + r" \\" + "\n")

        paths["tex"] = str(tex_path)

    return paths

paths = export_results_table_custom(
    "results/fig1_like_gof_results.json",
    output_dir="results",
    filename_prefix="fig1_like_gof_results_compact",
    columns=["power", "mean_p_ok", "q05_p_ok"],
)