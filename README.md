# TOI-4479 b ephemeris refinement

Code for *A Refined Orbital Period for TOI-4479 b from an Extended TESS Baseline*, Research Notes of the AAS 10, 231 (2026) ([ADS](https://ui.adsabs.harvard.edu/abs/2026RNAAS..10..231Z/abstract)).

`main.py` downloads the TESS SPOC light curves of TOI-4479 (TIC 126606859; sectors 41, 55 and 82), fits a mid-transit time to each transit with the transit shape fixed to Esparza-Borges et al. (2022), and fits a linear ephemeris that is compared with the ExoClock ephemeris of Kokori et al. (2026).

## Running

```
pip install -r requirements.txt
python main.py
```

The light curves are downloaded from MAST, so this needs a network connection. Tested with Python 3.12 and the versions pinned in `requirements.txt`.

The script writes these files to the current directory, the same ones as in the Zenodo deposit:

- `Figure_1.pdf`, `Figure_1.png`: Figure 1 of the note
- `TOI_4479_OC_table.csv`: mid-transit time, O-C, SNR and red-noise factor for each transit
- `TOI_4479_ephemeris_metadata.json`: refined and literature ephemerides, the comparison between them, the curvature test, and the fixed transit-shape inputs

It also prints a short summary. The refined ephemeris is

```
T0 = 3016.43853 ± 0.00048 BTJD  (BJD_TDB - 2457000)
P  = 1.1589128 ± 0.0000012 d
```

## Limb darkening

`toi4479b_ld_coeffs.json` contains the quadratic limb-darkening coefficients, computed with LDTk from the stellar parameters of Esparza-Borges et al. (2022). The script uses this file when it is present. To recompute the coefficients, delete it; that requires `ldtk` and downloads the PHOENIX spectra the first time. The sampling is seeded (`LDTK_SAMPLE_SEED`), so the result is reproducible.

## License

The code is under the MIT license. The data products are on Zenodo under CC-BY 4.0.
