This directory is heavily based on https://github.com/p-lambda/wilds

The command for the Camelyon-WILDS experiments in the paper is:
```
python run_expt.py --root_dir YOUR_DATA_FOLDER --dataset camelyon17 --unlabeled_split test_unlabeled --additional_train_transform randaugment --randaugment_n 2 --algorithm DivDis --divdis_diversity_weight 0.01 --log_dir logs/camelyon_final_testaug --seed 1
```

After running on all ten seeds, perform final evaluation with he command
```
python evaluate.py logs/camelyon_final_testaug --dataset camelyon17
```