# Helper Scripts

## prepare.sh

Set up the conda env, install dependencies, and download the FineWeb100B data:

```shell
./prepare.sh
```

## run.sh

Launch a stable WSD run: power LR exponent 7 (effective LR = 2^-7), 8 GPUs, outputs saved to `ckpts/lr7`:

```shell
./run.sh 7 8 ckpts/lr7
```

## decay.sh

Branch a decay run off `ckpts/lr7`: resumes the checkpoint at step 100000 and decays the LR to zero over 5000 steps (total 105000 iterations), saving to `ckpts/lr7_decay105000`:

```shell
./decay.sh 105000 8 ckpts/lr7
```

These helpers use WSD. Run `train.py` directly for cosine, sine, or triangle
schedules and repeat the schedule arguments when resuming.

## collect.py

Collect the final metrics of all decay runs of `ckpts/lr7` (the last line of each `ckpts/lr7_decay*/metrics.csv`) into `ckpts/lr7_decay_results.csv`:

```shell
python collect.py ckpts/lr7
```
