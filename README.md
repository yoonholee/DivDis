# DivDis

Code for the paper "Diversify and Disambiguate: Learning From Underspecified Data" ([arXiv](https://arxiv.org/abs/2202.03418), [project page](https://sites.google.com/view/diversify-and-disambiguate)).

---
<p align="center">
  <img align="middle" src="./assets/fig1.png" alt="Main figure"/>
</p>

The main repulsion loss for DivDis is implemented in `divdis.py` in the root directory, and shared by all experiments.

## Citation

If you find this code useful, please cite our [paper](https://arxiv.org/abs/2202.03418):

```
@inproceedings{lee2022divdis,
  title     = {Diversify and Disambiguate: Learning From Underspecified Data},
  author    = {Lee, Yoonho and Yao, Huaxiu and Finn, Chelsea},
  year      = {2022}
}
```

## Code references

- [MNIST-CIFAR repo](https://github.com/harshays/simplicitybiaspitfalls) (base code for MNIST-CIFAR experiment)
- [GroupDRO repo](https://github.com/kohpangwei/group_DRO) (base code for subpopulation shifts)