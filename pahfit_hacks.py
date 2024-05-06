from pahfit import instrument
import numpy as np
from specutils import Spectrum1D
from astropy import units as u


def remove_mask_from_spec(spec, mask):
    """Remove data points from a Spectrum1D object according to the given mask."""
    new_spec = Spectrum1D(
        spectral_axis=spec.spectral_axis[~mask],
        flux=spec.flux[..., ~mask],
        uncertainty=spec.uncertainty[..., ~mask],
        meta=spec.meta,
    )
    return new_spec


def remove_lines_from_data(spec, model):
    """Clip out data points where lines are present, based on the wavelengths listed in the model."""
    mask = np.full(spec.shape[-1], False)
    w = spec.spectral_axis.to(u.micron).value

    clean_width = 1.5
    for row in model.features:
        if row["kind"] == "line":
            line_w = row["wavelength"][0]
            if row["fwhm"].mask[0]:
                # get the maximum fwhm for the instrument combination
                fwhm_val_min_max = instrument.fwhm(
                    spec.meta["instrument"], line_w, as_bounded=True
                )[0]
                if fwhm_val_min_max.mask[0]:
                    # do nothing with lines that are outside anyway
                    continue
                else:
                    fwhm = fwhm_val_min_max[0]
            else:
                fwhm = row["fwhm"][0]
            wmin = line_w - clean_width * fwhm
            wmax = line_w + clean_width * fwhm
            at_line = np.logical_and(wmin < w, w < wmax)
            mask = np.logical_or(mask, at_line)

    print("masking ", np.count_nonzero(mask), "out of ", len(mask))

    nolines_spec = remove_mask_from_spec(spec, mask)
    return nolines_spec


def remove_lines_from_data_and_model(spec, model):
    """Modify both spectrum and model, to remove the lines in both."""
    new_model = model.copy()
    new_model.features = new_model.features[new_model.features["kind"] != "line"]
    return remove_lines_from_data(spec, model), new_model
