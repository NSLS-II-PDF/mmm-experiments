XCA
===
Anything within the convex hull gives 10 phases, all but one are experimentally
observed on the materials project. (TiCu_Pm-3m)


New method for isotropic expansion/contraction used. 

Tritium Env Setup to Train
==========================
The cctbx dependency is only avail via conda. 
```shell
conda create -n xca -c conda-forge cctbx-base=2022.3 python=3.9
conda activate xca
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113
pip install git+https://github.com/maffettone/xca@inline_training#egg=xca
```
With all of this, and even trying to use 2020 versions of cctbx, we still get a library error.
GLIBCXX_3.4.25 is the highest numbered version in the libstdc++.
```shell
ImportError: __import__("scitbx_array_family_flex_ext"): /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by /nsls2/users/pmaffetto/conda_envs/xca/lib/python3.9/lib-dynload/scitbx_array_family_flex_ext.so)
```



Model Training List
===================

Low Q, High Fidelity Model
--------------------------
- On lil-bits. 
- wandb: [terrifying-poltergeist-18](https://wandb.ai/phillip-maffettone/proj-xca/runs/3b77bt9s)
- checkpoint path: /home/pmm/project-mmm/proj-xca/3b77bt9s
```python
# BEGIN XRD PARAMETERS #
param_dict = {
    "wavelength": 0.1665,
    "noise_std": 5e-4,
    "instrument_radius": 1000.0,
    "theta_m": 0.0,
    "tth_min": 0.1,
    "tth_max": 12.0,
    "n_datapoints": 3000,
}
kwargs = {
    "bkg_2": (-1e-4, 1e-4),
    "bkg_1": (-1e-4, 1e-4),
    "bkg_0": (0, 1e-3),
    "march_range": (0.5, 1.0),
    "isotropic_expansion": (-0.05, 0.05),
}
shape_limit = 1e-3
```

Low Q, Low Fidelity Model (higher noise and peak shape)
-------------------------------------------------------
- On lil-bits. 
- wandb: [charmed-jazz-19](https://wandb.ai/phillip-maffettone/proj-xca/runs/3rb3fskq)
- checkpoint path: /home/pmm/project-mmm/proj-xca/3rb3fskq
- Expected phase ordering ['Pt1_Fm-3m', 'Zr1Pt3_P6_3mmc', 'Zr1_Ibam', 'Zr1Pt3_Pm-3m', 'Zr1Pt8_I4mmm', 'Zr1_Im-3m', 'Zr1_Fm-3m', 'Zr1_P6_3mmc', 'Zr1Pt1_Cmcm', 'Zr2Pt1_Fd-3m', 'Zr5Pt3_Cmcm', 'Zr7Pt10_Cmce', 'Zr9Pt11_I4m']
```python
# BEGIN XRD PARAMETERS #
param_dict = {
    "wavelength": 0.1665,
    "noise_std": 7e-3,
    "instrument_radius": 1000.0,
    "theta_m": 0.0,
    "tth_min": 0.1,
    "tth_max": 12.0,
    "n_datapoints": 3000,
}
kwargs = {
    "bkg_2": (-1e-4, 1e-4),
    "bkg_1": (-1e-4, 1e-4),
    "bkg_0": (0, 1e-3),
    "march_range": (0.0, 1.0),
    "isotropic_expansion": (-0.05, 0.05),
}
shape_limit = 1e-2
```
