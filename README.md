# Variational quantum algorithm for ergotropy estimation in quantum many-body batteries

This repository contains the code and data for the corresponding preprint article [arXiv:2308.03334](https://arxiv.org/abs/2308.03334).

## Summary

Quantum batteries are predicted to have the potential to outperform their classical counterparts and are therefore an important element in the development of quantum technologies. In this work we simulate the charging process and work extraction of many-body quantum batteries on noisy-intermediate scale quantum (NISQ) devices, and devise the Variational Quantum Ergotropy (VQErgo) algorithm which finds the optimal unitary operation that maximises work extraction from the battery. We test VQErgo by calculating the ergotropy of a quantum battery undergoing transverse field Ising dynamics. We investigate the battery for different system sizes and charging times and analyze the minimum required circuit depth of the variational optimization using both ideal and noisy simulators. Finally, we optimize part of the VQErgo algorithm and calculate the ergotropy on one of IBM's quantum devices. 

## Content

* [load_pvqd_runtime_fig9+12/](load_pvqd_runtime_fig9+12/): Script and data for reproducing figures 9 and 12 in the paper.
* [pvqd_vqe_fakeperth_fig11/](pvqd_vqe_fakeperth_fig11/): Script and data for reproducing figure 11 in the paper.
* [pvqd_vqe_statevector_differentM_fig4+5+10/](pvqd_vqe_statevector_differentM_fig4+5+10/): Script and data for reproducing figures 4, 5, and 10 in the paper.
* [pvqd_vqe_statevector_fig8/](pvqd_vqe_statevector_fig8/): Script and data for reproducing figure 8 in the paper.
* [rxx_fakeperth_fig15/](rxx_fakeperth_fig15/): Script and data for reproducing figure 15 in the paper.
* [rxx_runtime_fig16/](rxx_runtime_fig16/): Script and data for reproducing figure 16 in the paper.
* [rxx_statevector_differentM_fig14/](rxx_statevector_differentM_fig14/): Script and data for reproducing figure 14 in the paper.
* [train_pvqd_statevector_fig13/](train_pvqd_statevector_fig13/): Script and data for reproducing figure 13 in the paper.

## Requirements
The code is written in Python and apart from the usual libraries (numpy and matplotlib) you also need to have qiskit installed if you want to regenerate the data itself.

## Citation

If you use our code/models for your research, consider citing our paper:
```
@misc{hoang2023variational,
      title={Variational quantum algorithm for ergotropy estimation in quantum many-body batteries}, 
      author={Duc Tuan Hoang and Friederike Metz and Andreas Thomasen and Tran Duong Anh-Tai and Thomas Busch and Thomás Fogarty},
      year={2023},
      eprint={2308.03334},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
