from pathlib import Path

import matplotlib.pyplot as plt
from xca.data_synthesis.builder import single_pattern

# BEGIN XRD PARAMETERS #
param_dict = {
    "wavelength": 0.1665,
    "noise_std": 5e-4,
    "instrument_radius": 1000.0,
    "theta_m": 0.0,
    "tth_min": 0.1,
    "tth_max": 12.0,
    "n_datapoints": 2000,
}
kwargs = {"march_range": (0.5, 1.0), "isotropic_expansion": (-0.05, 0.05)}
shape_limit = 1e-3

cif_paths = list((Path(__file__).parent / "2022-nov" / "phases").glob("*.cif"))


def view_stacked():
    fig, ax = plt.subplots(figsize=(15, 15))
    data = []
    for i, path in enumerate(cif_paths):
        param_dict["input_cif"] = str(path)
        da = single_pattern(param_dict, shape_limit=shape_limit, **kwargs)
        da.data = da.data + i
        data.append(da)
        plt.plot(da.attrs["q"], da.data, label=path.name)
        ax.set_xlabel("q")
        ax.legend(bbox_to_anchor=(1.1, 1.05), loc="upper left")

    fig.tight_layout()
    fig.show()
    return fig, data


def view_single(idx):
    fig, ax = plt.subplots()
    param_dict["input_cif"] = cif_paths[idx]
    da = single_pattern(param_dict, shape_limit=shape_limit, **kwargs)
    plt.plot(da.attrs["q"], da.data)
    ax.set_title(cif_paths[idx].name)
    ax.set_xlabel("q")
    fig.show()
    return fig, da


if __name__ == "__main__":
    fig, data = view_stacked()
    fig2, da = view_single(0)
