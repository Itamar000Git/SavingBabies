from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("csv_output")
OUTPUT_DIR = Path("Croped_Dataset_CSV")

UC_COLUMN = "UC"
FHR_COLUMN = "FHR"
TIME_COLUMN = "t_sec"

UC_TRAILING_THRESHOLD = 10.0

# Remove another 3 minutes only if trailing low-UC rows were removed.
EXTRA_BACK_MINUTES_IF_CROPPED = 3

# Dataset sampling rate: 4Hz = 4 samples per second.
FS = 4

# If all UC values are below threshold, keep the file unchanged.
KEEP_ALL_LOW_UC_FILES = True


# ============================================================
# CROPPING LOGIC
# ============================================================

def find_last_uc_at_or_above_threshold(df: pd.DataFrame) -> int | None:
    """
    Finds the last row index where UC >= threshold.

    Rows after this index are the trailing low-UC tail.
    """
    uc = pd.to_numeric(df[UC_COLUMN], errors="coerce").fillna(0).to_numpy()

    keep_indices = np.where(uc >= UC_TRAILING_THRESHOLD)[0]

    if len(keep_indices) == 0:
        return None

    return int(keep_indices[-1])


def move_cut_back_3_minutes(df: pd.DataFrame, first_cut_index: int) -> int:
    """
    Moves the cut point 3 minutes earlier.

    If t_sec exists, use actual time.
    Otherwise, use FS rows:
        3 minutes * 60 seconds * 4Hz = 720 rows
    """
    extra_seconds = EXTRA_BACK_MINUTES_IF_CROPPED * 60

    if TIME_COLUMN in df.columns:
        t = pd.to_numeric(df[TIME_COLUMN], errors="coerce")

        cut_time = t.iloc[first_cut_index]
        target_time = cut_time - extra_seconds

        valid_indices = np.where(t.to_numpy() <= target_time)[0]

        if len(valid_indices) == 0:
            return 0

        return int(valid_indices[-1])

    extra_rows = EXTRA_BACK_MINUTES_IF_CROPPED * 60 * FS
    return max(0, first_cut_index - extra_rows)


def crop_csv_file(input_path: Path, output_path: Path) -> dict:
    """
    Crops only from the end.

    Step 1:
        Remove trailing rows while UC < 10.

    Step 2:
        If Step 1 removed anything, remove another 3 minutes before that point.

    If Step 1 removed nothing:
        Keep the file unchanged.

    FHR and all other columns are cut at the same row.
    """
    df = pd.read_csv(input_path)

    if UC_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {UC_COLUMN}")

    if FHR_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {FHR_COLUMN}")

    original_rows = len(df)

    if original_rows == 0:
        df.to_csv(output_path, index=False)
        return {
            "file": input_path.name,
            "original_rows": 0,
            "first_cut_rows": 0,
            "extra_3min_rows": 0,
            "final_rows": 0,
            "total_removed_rows": 0,
            "status": "empty file copied",
        }

    last_keep_index = find_last_uc_at_or_above_threshold(df)

    if last_keep_index is None:
        if KEEP_ALL_LOW_UC_FILES:
            cropped_df = df.copy()
            status = "all UC < threshold - copied unchanged"
        else:
            cropped_df = df.iloc[:0].copy()
            status = "all UC < threshold - saved empty"

        cropped_df.to_csv(output_path, index=False)

        return {
            "file": input_path.name,
            "original_rows": original_rows,
            "first_cut_rows": 0,
            "extra_3min_rows": 0,
            "final_rows": len(cropped_df),
            "total_removed_rows": original_rows - len(cropped_df),
            "status": status,
        }

    # Step 1: remove trailing rows where UC < 10.
    rows_after_first_cut = original_rows - (last_keep_index + 1)
    did_remove_trailing_low_uc = rows_after_first_cut > 0

    if did_remove_trailing_low_uc:
        # Step 2: because we removed trailing low UC,
        # move another 3 minutes backward.
        final_keep_index = move_cut_back_3_minutes(df, last_keep_index)
        status = "cropped trailing low UC + extra 3 minutes"
    else:
        # Important:
        # If no trailing low UC was removed, do NOT remove 3 minutes.
        final_keep_index = last_keep_index
        status = "no trailing low UC - copied unchanged"

    cropped_df = df.iloc[: final_keep_index + 1].copy()

    final_rows = len(cropped_df)
    total_removed_rows = original_rows - final_rows
    extra_3min_rows = total_removed_rows - rows_after_first_cut

    cropped_df.to_csv(output_path, index=False)

    return {
        "file": input_path.name,
        "original_rows": original_rows,
        "first_cut_rows": rows_after_first_cut,
        "extra_3min_rows": max(0, extra_3min_rows),
        "final_rows": final_rows,
        "total_removed_rows": total_removed_rows,
        "status": status,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {INPUT_DIR.resolve()}")
        return

    print(f"Input folder:  {INPUT_DIR.resolve()}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print(f"Found {len(csv_files)} CSV files.")
    print()
    print(f"Rule 1: remove trailing rows only while {UC_COLUMN} < {UC_TRAILING_THRESHOLD}")
    print(f"Rule 2: if Rule 1 removed rows, remove another {EXTRA_BACK_MINUTES_IF_CROPPED} minutes backward")
    print(f"Rule 3: if Rule 1 removed nothing, do not remove extra time")
    print()

    results = []

    for csv_file in csv_files:
        output_path = OUTPUT_DIR / csv_file.name

        try:
            result = crop_csv_file(csv_file, output_path)
            results.append(result)

            print(
                f"{result['file']}: "
                f"{result['original_rows']} -> {result['final_rows']} rows | "
                f"first_cut={result['first_cut_rows']} | "
                f"extra_3min={result['extra_3min_rows']} | "
                f"total_removed={result['total_removed_rows']} "
                f"({result['status']})"
            )

        except Exception as e:
            print(f"FAILED: {csv_file.name} | {e}")
            results.append({
                "file": csv_file.name,
                "original_rows": None,
                "first_cut_rows": None,
                "extra_3min_rows": None,
                "final_rows": None,
                "total_removed_rows": None,
                "status": f"failed: {e}",
            })

    summary_df = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "cropping_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print()
    print("Done.")
    print(f"Summary saved to: {summary_path.resolve()}")

    total_removed = summary_df["total_removed_rows"].dropna().sum()
    print(f"Total removed rows: {int(total_removed)}")


if __name__ == "__main__":
    main()