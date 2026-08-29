"""Collect final metrics from all decay runs of a checkpoint root.

Given a checkpoint root (e.g. ckpts/lr7), finds every decay output directory
produced by decay.sh (ckpts/lr7_decay*), takes the last line of each run's
metrics.csv, and writes them into a single CSV.

Usage:
    python collect.py ckpts/lr7 [-o output.csv]

Default output: <ckpt_root>_decay_results.csv
"""
import argparse
import csv
import glob
import os
import sys


def decay_iters(dirname, prefix):
    # extract NUM_ITERS from "<prefix>_decay<NUM_ITERS>" for numeric sorting
    suffix = os.path.basename(dirname)[len(os.path.basename(prefix) + '_decay'):]
    try:
        return int(suffix)
    except ValueError:
        return sys.maxsize


def main():
    parser = argparse.ArgumentParser(description='Collect final metrics of all decay runs of a ckpt root')
    parser.add_argument('ckpt_root', type=str,
                        help='checkpoint root of the stable run, e.g. ckpts/lr7')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='output csv path (default: <ckpt_root>_decay_results.csv)')
    args = parser.parse_args()

    root = args.ckpt_root.rstrip('/')
    out_path = args.output or root + '_decay_results.csv'
    # never overwrite an existing results file: pick the next free name instead
    if os.path.exists(out_path):
        stem, ext = os.path.splitext(out_path)
        k = 1
        while os.path.exists(f'{stem}_{k}{ext}'):
            k += 1
        out_path = f'{stem}_{k}{ext}'
        print(f'Output file exists, writing to {out_path} instead', file=sys.stderr)

    decay_dirs = sorted((d for d in glob.glob(root + '_decay*') if os.path.isdir(d)),
                        key=lambda d: decay_iters(d, root))
    if not decay_dirs:
        sys.exit(f'No decay directories found matching {root}_decay*')

    header = None
    rows = []
    for d in decay_dirs:
        metrics_path = os.path.join(d, 'metrics.csv')
        if not os.path.isfile(metrics_path):
            print(f'Warning: {metrics_path} not found, skipping', file=sys.stderr)
            continue
        with open(metrics_path, newline='') as f:
            lines = [row for row in csv.reader(f) if row]
        if len(lines) < 2:
            print(f'Warning: {metrics_path} has no data rows, skipping', file=sys.stderr)
            continue
        header = lines[0]
        rows.append(lines[-1])

    if not rows:
        sys.exit('No metrics collected')

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'Collected {len(rows)} decay runs into {out_path}')


if __name__ == '__main__':
    main()
