"""More minimal version of the pahfit decomposition plot. Good for
presentations. And eventually for the paper."""

from matplotlib import pyplot as plt
import numpy as np


def plot_decomposition(spec, m, with_labels=False):
    """Spec: spectrum that was given to PAHFIT

    m: PAHFIT Model
    """
    plt.figure()

    # data
    plt.plot(
        spec.spectral_axis.value,
        spec.flux.value,
        label="data",
        marker="_",
        linestyle="none",
        color="k",
    )

    #    big_unc = spec.uncertainty >
    plt.errorbar(
        spec.spectral_axis.value,
        spec.flux.value,
        spec.uncertainty.array,
        # capsize=0,
        color="k",
        linestyle="none",
        alpha=0.2,
    )

    # total model
    total = m.tabulate(spec.meta["instrument"], wavelengths=spec.spectral_axis)
    plt.plot(total.spectral_axis.value, total.flux.value, label="model", color="orange")

    # continuum model
    continuum = m.tabulate(
        spec.meta["instrument"],
        wavelengths=spec.spectral_axis,
        feature_mask=(m.features["kind"] == "dust_continuum")
        | (m.features["kind"] == "starlight"),
    )
    plt.plot(
        continuum.spectral_axis.value,
        continuum.flux.value,
        label="continuum",
        color="m",
    )

    def plot_single_component(name, **kwargs):
        component = m.tabulate(
            spec.meta["instrument"],
            wavelengths=spec.spectral_axis,
            feature_mask=m.features["name"] == name,
        )
        plt.plot(component.spectral_axis.value, component.flux.value, **kwargs)

    first = True
    for i in range(len(m.features)):
        if m.features["kind"][i] == "dust_feature" and m.features["power"][i][0] > 0:
            plot_single_component(
                m.features["name"][i],
                color="xkcd:light blue",
                label="dust feature" if first else None,
            )
            if with_labels:
                plt.text(
                    m.features["wavelength"][i, 0],
                    np.amin(spec.flux.value[spec.flux.value > 0]),
                    m.features["name"][i].replace("PAH_", ""),
                    rotation=90,
                    va="top",
                    ha="center",
                )
            first = False

    plt.xlabel("wavelength ($\\mu m$)")
    plt.ylabel("flux (MJy / sr)")
    plt.legend()
    plt.gcf().set_size_inches(9, 5)
    plt.tight_layout()
    return plt.gcf()
