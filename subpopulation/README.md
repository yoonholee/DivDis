# Subpopulation Shift Experiments

All default experiment settings are wrapped in the `--setting` argument.
The defaults are defined in `parser.py`.
One thing is that the baselines in the paper were run on a separate clone of the JTT repo, so it's possible that the numbers differ.

The entry point is `run_expt.py`:

```bash
# ERM on default tasks
python run_expt.py --setting WBIRDS         
python run_expt.py --setting MultiNLI
python run_expt.py --setting CELEBA_1
python run_expt.py --setting CELEBA_2
python run_expt.py --setting CELEBA_3
python run_expt.py --setting CELEBA_4
```

To run the complete correlation tasks in the paper, add the `--majority_only` option.
For example, the following runs ERM on the Waterbirds-CC task:

```bash
# ERM on Waterbirds-CC
python run_expt.py --setting WBIRDS --majority_only     
```

To run DivDis, add `--diversify`.
A few hyperparameter variants are shown below.

```bash
# DivDis on Waterbirds-CC
python run_expt.py --setting WBIRDS --majority_only --diversify             
# DivDis on Waterbirds-CC with 4 heads
python run_expt.py --setting WBIRDS --majority_only --diversify --heads 4
# DivDis on Waterbirds-CC with L1 repulsion loss
python run_expt.py --setting WBIRDS --majority_only --diversify --mode l1
# DivDis on Waterbirds-CC with repulsion loss weight 10
python run_expt.py --setting WBIRDS --majority_only --diversify --diversity_weight 10.0
```
