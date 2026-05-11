# src/feature_engineering.py
"""
Standalone feature-engineering script.

Run this ONCE on raw train/test CSVs to produce engineered Parquet files and a
train_stats pickle.  Subsequent training runs load the Parquet files directly,
skipping the expensive FE step entirely.

Outputs (all paths configurable via CLI):
  <output_dir>/train_engineered.parquet
  <output_dir>/test_engineered.parquet   (optional, only if --test_dataset given)
  <output_dir>/train_stats.pkl
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from utils_data import apply_feature_engineering


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description='taxi_trip_feature_engineering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--train_dataset',
        type=str,
        default=str(project_root / 'split' / 'train.csv'),
        help='Path to raw training CSV.',
    )
    parser.add_argument(
        '--test_dataset',
        type=str,
        default=None,
        help='Path to raw test CSV (optional). If provided, FE is applied using '
             'train_stats fitted on training data.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(project_root / 'split'),
        help='Directory where engineered Parquet files and train_stats.pkl are saved.',
    )
    parser.add_argument(
        '--iqr_factor',
        type=float,
        default=2.5,
        help='IQR multiplier for statistical outlier removal.',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training data -------------------------------------------------- #
    print(f"Loading training data from: {args.train_dataset}")
    train_raw = pd.read_csv(args.train_dataset)
    print(f"  Raw rows: {len(train_raw):,}")

    print("\nApplying feature engineering to training data ...")
    train_fe, train_stats = apply_feature_engineering(
        train_raw, train_stats=None, iqr_factor=args.iqr_factor
    )
    print(f"  Engineered rows: {len(train_fe):,}")

    train_out = output_dir / 'train_engineered.parquet'
    train_fe.to_parquet(train_out, index=False)
    print(f"  Saved → {train_out}")

    stats_out = output_dir / 'train_stats.pkl'
    with open(stats_out, 'wb') as f:
        pickle.dump(train_stats, f)
    print(f"  Saved → {stats_out}")

    # ---- Test data (optional) ------------------------------------------- #
    if args.test_dataset:
        print(f"\nLoading test data from: {args.test_dataset}")
        test_raw = pd.read_csv(args.test_dataset)
        print(f"  Raw rows: {len(test_raw):,}")

        print("Applying feature engineering to test data (using training stats) ...")
        test_fe, _ = apply_feature_engineering(
            test_raw, train_stats=train_stats, iqr_factor=args.iqr_factor
        )
        print(f"  Engineered rows: {len(test_fe):,}")

        test_out = output_dir / 'test_engineered.parquet'
        test_fe.to_parquet(test_out, index=False)
        print(f"  Saved → {test_out}")

    print("\nFeature engineering complete.")


if __name__ == '__main__':
    main()