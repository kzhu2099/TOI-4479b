"""Refined transit ephemeris of TOI-4479 b from TESS SPOC light curves (Zhu 2026, RNAAS 10, 231).

Fits a mid-transit time to each TESS transit, fits a weighted linear ephemeris and compares it with Kokori et al. (2026). Writes Figure_1 (.pdf/.png), the per-transit O-C table and the ephemeris metadata JSON from the Zenodo deposit.

Usage: python main.py
"""

import csv
import functools
import json
import os
import warnings
from datetime import datetime

import batman
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.timeseries import BoxLeastSquares
from matplotlib.lines import Line2D
from scipy import stats
from scipy.optimize import curve_fit
from wotan import flatten, transit_mask

warnings.filterwarnings('ignore')

sns.set_theme(style='ticks', context='paper', font_scale=1.1, palette='colorblind')

FIGURE_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Tinos', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.3,
    'lines.markeredgewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'pdf.fonttype': 42,  # embed TrueType fonts (AAS)
    'ps.fonttype': 42,
}

STAR_ID = 'TOI_4479'
MAST_SEARCH_ID = 'TIC 126606859'

# Ephemeris and T14 from Kokori et al. (2026, ExoClock IV): Tc = 2460016.43991 +/- 0.00088 BJD_TDB, P = 1.158929 +/- 4.8e-6 d, T14 = 1.0981 +/- 0.0078 hr.
LIT_PERIOD, LIT_PERIOD_ERR = 1.158929, 4.8e-06    # days
LIT_T0_BTJD, LIT_T0_ERR = 3016.43991, 0.00088      # BTJD, days
LIT_DURATION_DAYS = 1.0981 / 24.0

# Transit shape, fixed to Esparza-Borges et al. (2022), Table 2. They give both the apparent radius ratio k_app = 0.0572 and the dilution-corrected k_true = 0.062, but adopt k_app themselves (Sec. 6.1: contamination is negligible), so we do too. measure_transit_depth() checks this with a free Rp/Rs fit.
RP_RS, A_RS, INC = 0.0572, 7.8, 86.36

# Stellar parameters from Esparza-Borges et al. (2022), Table 1, used as LDTk inputs. They only give lower bounds on logg and [Fe/H], so we center on the bound with a 0.2 dex width.
TEFF, TEFF_ERR = 3400.0, 100.0   # K
LOGG, LOGG_ERR = 4.5, 0.2        # dex
FEH, FEH_ERR = 0.0, 0.2          # dex
LDTK_SAMPLE_SEED = 44790622      # arbitrary; makes u1/u2 reproducible
LD_CACHE_FILE = 'toi4479b_ld_coeffs.json'

TIME_STANDARD = 'BJD_TDB'
LIT_CITATION = 'Kokori, A., Tsiaras, A., Pantelidou, G., et al. 2026, ApJS, 283, 5'
SHAPE_CITATION = 'Esparza-Borges, E., Parviainen, H., Murgas, F., et al. 2022, A&A, 666, A10'
LD_CITATION = 'Parviainen, H. & Aigrain, S. 2015, MNRAS, 453, 3821 (LDTk); Husser, T.-O., Wende-von Berg, S., Dreizler, S., et al. 2013, A&A, 553, A6 (PHOENIX spectra)'

QUALITY_BITMASK = 'default'
DETREND_WINDOW = 0.3         # days, ~6x the transit duration
DETREND_MASK_FACTOR = 2.2    # width of the in-transit mask, in transit durations
FIT_HALF_WINDOW_HOURS = 4.0  # fit +/-4 hr around each predicted T0
MIN_DATAPOINTS = 20
MAX_T0_ERR = 0.03            # days; only meant to catch failed fits
JACKKNIFE_SIGMA_THRESH = 4.0

PHASE_FOLD_WINDOW_HR = 3     # half-width of the phase-fold panels
PHASE_BIN_MINUTES = 5.0
OUTLINE_EDGE_PAD_HR = 2.0    # model curves extend past the panel edges by this much


# ---- Data ----

def fetch_data():
    """Download and stitch the SPOC light curves. Returns time, flux, flux_err, sector."""
    import lightkurve as lk

    print(f'Downloading TESS SPOC light curves for {MAST_SEARCH_ID}...')
    search = lk.search_lightcurve(MAST_SEARCH_ID, mission='TESS', author='SPOC', exptime=[20, 120])
    times, fluxes, flux_errs, sectors = [], [], [], []
    for lc in search.download_all(quality_bitmask=QUALITY_BITMASK):
        lc = lc.remove_nans().normalize()
        times.append(lc.time.value)
        fluxes.append(lc.flux.value)
        flux_errs.append(lc.flux_err.value)
        sectors.append(np.full(len(lc.time), int(lc.sector), dtype=int))

    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    flux_err = np.concatenate(flux_errs)
    sector = np.concatenate(sectors)

    order = np.argsort(time)
    time, flux, flux_err, sector = time[order], flux[order], flux_err[order], sector[order]

    # Clip gross (>5 sigma) outliers; detrending comes later
    mask = (np.abs(flux - np.median(flux)) < (5 * np.std(flux))) & np.isfinite(flux_err)
    return time[mask], flux[mask], flux_err[mask], sector[mask]


def detrend(time, flux, sector):
    """Transit-masked biweight filter, then each sector rescaled so its out-of-transit median is 1."""
    in_transit = transit_mask(time=time, period=LIT_PERIOD, duration=LIT_DURATION_DAYS * DETREND_MASK_FACTOR, T0=LIT_T0_BTJD)
    flat_flux, _ = flatten(time, flux, window_length=DETREND_WINDOW, method='biweight', mask=in_transit, return_trend=True)
    for s in np.unique(sector):
        sel = sector == s
        flat_flux[sel] = flat_flux[sel] / np.median(flat_flux[sel & ~in_transit])
    return flat_flux


def bls_seed(time, flat_flux):
    """BLS search near the literature period, to seed the individual T0 fits."""
    periods = np.linspace(LIT_PERIOD - 0.15, LIT_PERIOD + 0.15, 100000)
    durations = np.linspace(LIT_DURATION_DAYS * 0.7, LIT_DURATION_DAYS * 1.3, 5)
    res = BoxLeastSquares(time, flat_flux).power(periods, durations)
    best = np.argmax(res.power)
    period, t0 = res.period[best], res.transit_time[best]

    # Move T0 to the transit closest to the literature T0
    t0 = t0 - np.round((t0 - LIT_T0_BTJD) / period) * period
    return t0, period


# ---- Transit model ----

@functools.cache
def limb_darkening():
    """Quadratic TESS-band limb darkening from LDTk, cached to LD_CACHE_FILE."""
    if os.path.exists(LD_CACHE_FILE):
        with open(LD_CACHE_FILE) as f:
            return json.load(f)

    from ldtk import LDPSetCreator
    from ldtk.filters import tess

    sc = LDPSetCreator(teff=(TEFF, TEFF_ERR), logg=(LOGG, LOGG_ERR), z=(FEH, FEH_ERR), filters=[tess])
    # create_profiles() samples the stellar parameters with numpy's global RNG
    np.random.seed(LDTK_SAMPLE_SEED)
    ps = sc.create_profiles(nsamples=2000)
    qc, qe = ps.coeffs_qd(do_mc=False)  # deterministic ML fit
    meta = {
        'u1': float(qc[0][0]),
        'u2': float(qc[0][1]),
        'u1_err': float(qe[0][0]),
        'u2_err': float(qe[0][1]),
        'teff': TEFF,
        'logg': LOGG,
        'feh': FEH,
        'sample_seed': LDTK_SAMPLE_SEED,
        'coeffs_qd_do_mc': False,
    }
    with open(LD_CACHE_FILE, 'w') as f:
        json.dump(meta, f)
    return meta


def batman_params(rp_rs=RP_RS, period=100.0):
    """batman parameters for the fixed Esparza-Borges et al. (2022) shape."""
    ld = limb_darkening()
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = period
    params.rp = rp_rs
    params.a = A_RS
    params.inc = INC
    params.ecc = 0.0
    params.w = 90.0
    params.u = [ld['u1'], ld['u2']]
    params.limb_dark = 'quadratic'
    return params


def model_depth_ppm(rp_rs):
    """Mid-transit depth (ppm) of the limb-darkened model. With this star's limb darkening and b ~ 0.5 this is ~3710 ppm for k_app, not (Rp/Rs)^2 = 3272 ppm."""
    params = batman_params(rp_rs)
    flux_mid = batman.TransitModel(params, np.array([0.0])).light_curve(params)[0]
    return (1.0 - flux_mid) * 1e6


def measure_transit_depth(time, flat_flux, flux_err, t0, period):
    """Fit Rp/Rs freely (a/Rs, inc and LD fixed) to the phase-folded light curve, as a check on RP_RS."""
    phase_hr = fold_phase(time, period, t0)
    keep = np.abs(phase_hr) < FIT_HALF_WINDOW_HOURS
    params = batman_params(period=period)

    def model(t, rp, baseline):
        params.rp = rp
        return batman.TransitModel(params, t).light_curve(params) * baseline

    popt, pcov = curve_fit(model, phase_hr[keep] / 24.0, flat_flux[keep], p0=[RP_RS, 1.0], sigma=flux_err[keep], bounds=([0.001, 0.9], [0.3, 1.1]), absolute_sigma=True, maxfev=5000)
    rp_fit, rp_err = float(popt[0]), float(np.sqrt(pcov[0, 0]))

    depth_ppm = model_depth_ppm(rp_fit)
    depth_hi = model_depth_ppm(rp_fit + rp_err)
    depth_lo = model_depth_ppm(max(rp_fit - rp_err, 1e-6))
    depth_kapp_ppm = model_depth_ppm(0.0572)
    depth_ktrue_ppm = model_depth_ppm(0.062)
    closer = abs(depth_ppm - depth_ktrue_ppm) < abs(depth_ppm - depth_kapp_ppm)
    return {
        'rp_rs_fit': rp_fit,
        'rp_rs_fit_err': rp_err,
        'depth_ppm': depth_ppm,
        'depth_err_ppm': abs(depth_hi - depth_lo) / 2.0,
        'depth_kapp_ppm': depth_kapp_ppm,
        'depth_ktrue_ppm': depth_ktrue_ppm,
        'closer_to': 'k_true' if closer else 'k_app',
        'rp_rs_used_for_timing': RP_RS,
        'half_window_hr': FIT_HALF_WINDOW_HOURS,
    }


# ---- Transit timing ----

def red_noise_beta(residuals, cadence_days, duration_days):
    """Red-noise beta factor (Winn et al. 2008): binned scatter over the white-noise expectation."""
    bin_size = int(np.round(duration_days / cadence_days))
    if bin_size < 2 or len(residuals) < 2 * bin_size:
        return 1.0
    n_bins = len(residuals) // bin_size
    binned = np.mean(residuals[:n_bins * bin_size].reshape(n_bins, bin_size), axis=1)
    beta = np.std(binned) / (np.std(residuals) / np.sqrt(bin_size))
    return max(1.0, beta)


def fit_transit_times(time, flux, flux_err, sector, t0_seed, period_seed):
    """Fit T0 for each transit (fixed-shape model times a local linear baseline). Errors are inflated by the red-noise beta factor."""
    params = batman_params(period=period_seed)
    depth = model_depth_ppm(RP_RS) / 1e6

    first = int(np.floor((time.min() - t0_seed) / period_seed))
    last = int(np.ceil((time.max() - t0_seed) / period_seed))
    t0s, t0_errs, snrs, betas, sectors = [], [], [], [], []

    for n in range(first, last + 1):
        expected_t0 = t0_seed + n * period_seed
        win = np.abs(time - expected_t0) < (FIT_HALF_WINDOW_HOURS / 24.0)
        if np.sum(win) < MIN_DATAPOINTS:
            continue

        t_win = time[win]
        local_median = np.median(flux[win])
        f_win = flux[win] / local_median
        e_win = np.clip(flux_err[win] / local_median, 1e-8, None)

        params.t0 = expected_t0
        transit_model = batman.TransitModel(params, t_win)

        def joint_model(t, t0_fit, offset, slope):
            params.t0 = t0_fit
            return transit_model.light_curve(params) * (offset + slope * (t - expected_t0))

        lo, hi = expected_t0 - 1.5 / 24.0, expected_t0 + 1.5 / 24.0
        try:
            popt, pcov = curve_fit(joint_model, t_win, f_win, p0=[expected_t0, 1.0, 0.0], sigma=e_win, bounds=([lo, 0.90, -0.5], [hi, 1.10, 0.5]), maxfev=3000, absolute_sigma=True)
        except (RuntimeError, ValueError):
            continue

        # Skip failed fits and T0s stuck at the bounds
        err = np.sqrt(pcov[0][0])
        if not (np.isfinite(err) and 1e-5 < err < MAX_T0_ERR and lo + 1e-4 < popt[0] < hi - 1e-4):
            continue

        resid = f_win - joint_model(t_win, *popt)
        beta = red_noise_beta(resid, np.median(np.diff(t_win)), LIT_DURATION_DAYS)
        n_in = int(np.sum(np.abs(t_win - popt[0]) < (LIT_DURATION_DAYS / 2.0)))
        vals, counts = np.unique(sector[win], return_counts=True)

        t0s.append(popt[0])
        t0_errs.append(err * beta)
        snrs.append(depth / (np.std(resid) / np.sqrt(n_in)))
        betas.append(beta)
        sectors.append(int(vals[np.argmax(counts)]))

    t0s = np.array(t0s)
    epochs = np.round((t0s - LIT_T0_BTJD) / LIT_PERIOD)  # E = 0 is the literature T0
    return epochs, t0s, np.array(t0_errs), np.array(snrs), np.array(betas), np.array(sectors)


def jackknife_mask(epochs, t0s, t0_errs):
    """True for epochs within JACKKNIFE_SIGMA_THRESH sigma of a linear fit to the others."""
    n = len(epochs)
    w = 1.0 / t0_errs
    resid = np.zeros(n)
    for i in range(n):
        others = np.arange(n) != i
        coeffs, cov = np.polyfit(epochs[others], t0s[others], 1, w=w[others], cov='unscaled')
        # Include the error of the leave-one-out prediction, so epochs at the ends of the baseline aren't rejected too easily
        e = epochs[i]
        pred_var = (e ** 2) * cov[0, 0] + 2.0 * e * cov[0, 1] + cov[1, 1]
        resid[i] = np.abs(t0s[i] - np.polyval(coeffs, e)) / np.sqrt(t0_errs[i] ** 2 + max(pred_var, 0.0))
    return resid < JACKKNIFE_SIGMA_THRESH


def fit_linear_ephemeris(epochs, t0s, t0_errs):
    """Weighted linear fit of T0 vs. epoch. Epochs count from LIT_T0_BTJD, so the intercept is the refined T0. Errors come from the unscaled covariance; chi2/dof ('scale_factor') is only a diagnostic."""
    w = 1.0 / t0_errs
    (period, t0), cov = np.polyfit(epochs, t0s, deg=1, w=w, cov='unscaled')
    _, cov_scaled = np.polyfit(epochs, t0s, deg=1, w=w, cov=True)
    period_err, t0_err = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    return {
        't0': t0,
        'period': period,
        't0_err': t0_err,
        'period_err': period_err,
        'cov': cov,
        'scale_factor': cov_scaled[0, 0] / cov[0, 0],
        'delta_p_sec': (period - LIT_PERIOD) * 86400.0,
        'delta_p_sigma': (period - LIT_PERIOD) / np.sqrt(period_err ** 2 + LIT_PERIOD_ERR ** 2),
        't0_diff_min': (t0 - LIT_T0_BTJD) * 24 * 60,
        't0_diff_sigma': (t0 - LIT_T0_BTJD) / np.sqrt(t0_err ** 2 + LIT_T0_ERR ** 2),
    }


def curvature_test(epochs, t0s, t0_errs, period):
    """Linear vs. quadratic ephemeris (F-test, delta BIC) and the implied Pdot."""
    w = 1.0 / t0_errs
    n = len(epochs)
    # Centered epochs keep the quadratic fit well conditioned; chi2 and the quadratic term don't change
    e_c = epochs - float(np.mean(epochs))

    c1 = np.polyfit(e_c, t0s, deg=1, w=w)
    c2, cov2 = np.polyfit(e_c, t0s, deg=2, w=w, cov='unscaled')
    chi2_1 = float(np.sum(((t0s - np.polyval(c1, e_c)) * w) ** 2))
    chi2_2 = float(np.sum(((t0s - np.polyval(c2, e_c)) * w) ** 2))

    f_stat = (chi2_1 - chi2_2) / (chi2_2 / (n - 3))
    p_value = float(stats.f.sf(f_stat, 1, n - 3))
    delta_bic = (chi2_2 + 3 * np.log(n)) - (chi2_1 + 2 * np.log(n))

    # T(E) = T0 + P*E + 0.5*P*Pdot*E^2, so Pdot = 2*c2[0]/P (converted to ms/yr)
    pdot = 2.0 * float(c2[0]) / period
    pdot_err = 2.0 * float(np.sqrt(cov2[0, 0])) / period
    # Rough 3-sigma detection limit from the median T0 error and the epoch span
    min_quad = 3.0 * float(np.median(t0_errs)) / (0.5 * float(epochs.max() - epochs.min()) ** 2)

    return {
        'preferred': 'linear (no significant curvature)' if delta_bic > 0 else 'quadratic',
        'delta_bic': delta_bic,
        'p_value': p_value,
        'pdot_ms_yr': pdot * 86400e3 * 365.25,
        'pdot_err_ms_yr': pdot_err * 86400e3 * 365.25,
        'pdot_limit_ms_yr': 2.0 * min_quad / period * 86400e3 * 365.25,
    }


# ---- Outputs ----

def write_oc_table(epochs, t0s, t0_errs, snrs, sectors, betas, used, eph):
    """Per-transit T0s and O-C against the refined ephemeris."""
    oc_min = (t0s - (eph['t0'] + epochs * eph['period'])) * 24.0 * 60.0
    with open(f'{STAR_ID}_OC_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'T0_BTJD', 'T0_err_days', 'O_minus_C_min', 'transit_SNR', 'sector', 'red_noise_beta', 'used_in_fit'])
        for i in np.argsort(epochs, kind='stable'):
            writer.writerow([int(epochs[i]), f'{t0s[i]:.6f}', f'{t0_errs[i]:.6f}', f'{oc_min[i]:.3f}', f'{snrs[i]:.1f}', int(sectors[i]), f'{betas[i]:.3f}', bool(used[i])])


def write_metadata(eph, curvature, depth, fig_stats, rms_lit, rms_ref, epochs, n_rej, sector, transit_baseline_days, photometric_baseline_days):
    """Summary JSON for the Zenodo deposit."""
    sectors = np.unique(sector)
    payload = {
        'dataset_title': 'A Refined Orbital Period for TOI-4479 b from an Extended TESS Baseline: Transit-Timing Data',
        'star': {'star_id': STAR_ID, 'mast_search_id': MAST_SEARCH_ID},
        'time_standard': TIME_STANDARD,
        'refined_ephemeris': {
            't0_btjd': eph['t0'], 't0_err_days': eph['t0_err'],
            'period_days': eph['period'], 'period_err_days': eph['period_err'],
            'chi2_dof_scale_factor': eph['scale_factor'],
        },
        'literature_ephemeris': {
            't0_btjd': LIT_T0_BTJD, 't0_err_days': LIT_T0_ERR,
            'period_days': LIT_PERIOD, 'period_err_days': LIT_PERIOD_ERR,
            'citation': LIT_CITATION,
        },
        'comparison_to_literature': {
            'delta_period_sec': eph['delta_p_sec'],
            'delta_period_sigma': eph['delta_p_sigma'],
            't0_diff_min': eph['t0_diff_min'],
            't0_diff_sigma': eph['t0_diff_sigma'],
            'rms_vs_literature_min': rms_lit,
            'rms_vs_refined_min': rms_ref,
            'projected_offset': {
                'label': fig_stats['future_label'],
                'offset_min': fig_stats['future_offset_min'],
                'offset_sigma_min': fig_stats['future_offset_sigma_min'],
            },
        },
        'curvature_test': {
            'preferred_model': curvature['preferred'], 'delta_bic': curvature['delta_bic'],
            'f_test_p_value': curvature['p_value'],
            'pdot_ms_yr': curvature['pdot_ms_yr'], 'pdot_err_ms_yr': curvature['pdot_err_ms_yr'],
        },
        'observations': {
            'n_transits_used': len(epochs), 'n_transits_rejected': n_rej,
            'epoch_range': [int(np.min(epochs)), int(np.max(epochs))],
            'n_sectors': len(sectors), 'sectors': [int(s) for s in sectors],
            'transit_baseline_days': transit_baseline_days,          # first to last adopted T0
            'photometric_baseline_days': photometric_baseline_days,  # full light-curve span
        },
        'data_source': {
            'facility': 'TESS', 'pipeline': 'SPOC', 'quality_bitmask': QUALITY_BITMASK,
            'mast_dataset_doi': '10.17909/T9-NMC8-F686',
        },
        'transit_shape': {
            'rp_rs_fixed': RP_RS, 'a_rs_fixed': A_RS, 'inc_deg_fixed': INC,
            'shape_citation': SHAPE_CITATION,
            'limb_darkening': limb_darkening(),
            'limb_darkening_citation': LD_CITATION,
        },
        'depth_measurement': depth,
        'license': 'CC-BY-4.0',
    }
    with open(f'{STAR_ID}_ephemeris_metadata.json', 'w') as f:
        json.dump(payload, f, indent=2)


# ---- Figure ----

def fold_phase(time, period, t0):
    """Phase-fold on `period` around `t0`; returns hours from mid-transit."""
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    return phase * 24.0


def bin_phase(phase_hours, flux, bin_width_min=PHASE_BIN_MINUTES):
    """Bin a phase-folded light curve. Returns bin centers, median flux and SEM."""
    bin_width_hr = bin_width_min / 60.0
    edges = np.arange(phase_hours.min(), phase_hours.max() + bin_width_hr, bin_width_hr)
    idx = np.digitize(phase_hours, edges)
    centers, meds, sems = [], [], []
    for i in range(1, len(edges)):
        m = idx == i
        n = int(np.sum(m))
        if n == 0:
            continue
        centers.append(np.mean(phase_hours[m]))
        meds.append(np.median(flux[m]))
        sems.append(np.std(flux[m]) / np.sqrt(n) if n > 1 else np.nan)
    return np.array(centers), np.array(meds), np.array(sems)


# Paul Tol's (2021) colorblind-safe "muted" palette, without sand and wine
_SECTOR_PALETTE = ["#332288", "#44AA99", "#CC6677", "#117733", "#88CCEE", "#999933"]
_SECTOR_MARKERS = ['o', 's', '^', 'D', 'v', 'P']
_SECTOR_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]


def _cluster_epochs(epochs):
    """Group epochs into clusters (one per sector, in practice) split at large gaps."""
    order = np.argsort(epochs)
    diffs = np.diff(epochs[order])
    threshold = max(float(np.median(diffs[diffs > 0])) * 6.0, 15.0)
    groups = [[order[0]]]
    for i, d in enumerate(diffs):
        if d > threshold:
            groups.append([])
        groups[-1].append(order[i + 1])
    return groups


def _cluster_layout(epochs, cluster_idx, pad_frac=0.3, min_pad=3.0):
    """(lo, hi, pad) x-range for each epoch cluster, plus their relative widths."""
    layout = []
    for idxs in cluster_idx:
        e_sub = epochs[np.asarray(idxs)]
        lo, hi = float(e_sub.min()), float(e_sub.max())
        layout.append((lo, hi, max((hi - lo) * pad_frac, min_pad)))
    widths = [(hi - lo) + 2 * pad for lo, hi, pad in layout]
    return layout, widths


def _style_broken_axes(axes_group, cluster_layout):
    """Hide the inner spines of a broken-axis group and draw the diagonal break marks."""
    n = len(axes_group)
    d = 0.018
    # d is in axes-fraction units, so scale dx by panel width to keep the marks the same size
    rel_widths = [(hi - lo) + 2 * pad for lo, hi, pad in cluster_layout]
    ref_width = float(np.median(rel_widths))

    for i, ax in enumerate(axes_group):
        if i > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        if i < n - 1:
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', which='both', right=False)

        # Evenly spaced ticks; MaxNLocator bunches them up in narrow clusters
        lo, hi, _ = cluster_layout[i]
        span = hi - lo
        n_ticks = 2 if span < 5 else (3 if span < 12 else 4)
        n_ticks = max(2, min(n_ticks, int(np.floor(hi) - np.ceil(lo)) + 1))
        ax.set_xticks(np.unique(np.round(np.linspace(lo, hi, n_ticks))))
        ax.tick_params(axis='x', labelrotation=40, labelsize=10)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
            lbl.set_rotation_mode('anchor')

        dx = float(np.clip(d * (ref_width / rel_widths[i]), d * 0.4, d * 2.5))
        kwargs = dict(transform=ax.transAxes, color='0.45', clip_on=False, lw=1.1)
        if i < n - 1:
            ax.plot([1 - dx, 1 + dx], [-d, d], **kwargs)
            ax.plot([1 - dx, 1 + dx], [1 - d, 1 + d], **kwargs)
        if i > 0:
            ax.plot([-dx, dx], [-d, d], **kwargs)
            ax.plot([-dx, dx], [1 - d, 1 + d], **kwargs)


def _width_ratios(cluster_widths, cluster_layout, period):
    """Column widths for the broken axis, with a spacer between clusters that grows (logarithmically) with the time skipped."""
    gap_days = [(cluster_layout[i + 1][0] - cluster_layout[i][1]) * period for i in range(len(cluster_layout) - 1)]
    ref = float(np.median(cluster_widths))
    spacers = [float(np.clip(ref * 0.07 * np.log10(g + 1.0), ref * 0.06, ref * 0.22)) for g in gap_days]
    widths = [cluster_widths[0]]
    col_idx = [0]
    for sw, cw in zip(spacers, cluster_widths[1:]):
        widths.append(sw)
        widths.append(cw)
        col_idx.append(len(widths) - 1)
    return widths, col_idx


def _plot_oc_panel(axes_group, cluster_idx, cluster_layout, epochs, t0s, t0_errs, sec, rej_epochs, rej_t0s, rej_errs, ref_t0, ref_p, unique_sectors, sector_colors, sector_markers, ylim, fit_trend):
    """O-C vs. epoch on a broken-axis group, colored by sector, optionally with a linear trend and its 1-sigma band."""
    reject_color, trend_color = '#767676', '#262B33'
    oc_min = (t0s - (ref_t0 + epochs * ref_p)) * 24.0 * 60.0
    oc_err_min = t0_errs * 24.0 * 60.0
    rej_oc_min = (rej_t0s - (ref_t0 + rej_epochs * ref_p)) * 24.0 * 60.0

    if fit_trend:
        # Unscaled covariance, so the band reflects the actual timing errors (as in Patra et al. 2017; Ivshina & Winn 2022)
        c_lin, cov_lin = np.polyfit(epochs, oc_min, deg=1, w=1.0 / oc_err_min, cov='unscaled')

    h_rej, h_trend = None, None
    for ax, idxs, (lo, hi, pad) in zip(axes_group, cluster_idx, cluster_layout):
        idxs = np.asarray(idxs)
        ax.axhline(0, color='0.75', lw=1.0, ls='--', alpha=0.9, zorder=1)
        ax.grid(True, axis='y', color='0.90', lw=0.6, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)

        if fit_trend:
            e_line = np.linspace(lo - pad, hi + pad, 60)
            fit_line = np.polyval(c_lin, e_line)
            var_line = (e_line ** 2) * cov_lin[0, 0] + 2.0 * e_line * cov_lin[0, 1] + cov_lin[1, 1]
            sigma_line = np.sqrt(np.clip(var_line, 0.0, None))
            ax.fill_between(e_line, fit_line - sigma_line, fit_line + sigma_line, color=trend_color, alpha=0.16, lw=0, zorder=1.2)
            h_trend, = ax.plot(e_line, fit_line, '-', color=trend_color, lw=1.8, alpha=0.95, zorder=1.5)

        for s in unique_sectors:
            sel = sec[idxs] == s
            if not np.any(sel):
                continue
            sub_idx = idxs[sel]
            ax.errorbar(epochs[sub_idx], oc_min[sub_idx], yerr=oc_err_min[sub_idx], fmt=sector_markers[s], color=sector_colors[s], ms=5.6, capsize=0, elinewidth=1.1, mec='white', mew=0.7, zorder=3)

        in_range = (rej_epochs >= lo - pad) & (rej_epochs <= hi + pad)
        if np.any(in_range):
            h_rej = ax.errorbar(rej_epochs[in_range], rej_oc_min[in_range], yerr=rej_errs[in_range] * 24.0 * 60.0, fmt='x', color=reject_color, ms=6.5, capsize=0, elinewidth=1.1, mec='none', zorder=2)

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(*ylim)

    axes_group[0].set_ylabel('O $-$ C (min)')
    if h_rej is not None:
        h_rej.set_label('Rejected')
    if h_trend is not None:
        h_trend.set_label('Drift fit ($\\pm1\\sigma$)')
    return h_rej, h_trend


def _plot_phase_panel(ax, t, f, sec, per, t0, unique_sectors, sector_colors, sector_markers, sector_linestyles, model_ph_hr, model_flux, adopted_epochs, adopted_t0, adopted_sectors):
    """Phase-fold on (t0, per): faint binned data plus one model outline per sector, shifted by the mean O-C of that sector's transits. Returns the spread (min) between the sectors' shifts."""
    phase = fold_phase(t, per, t0)
    keep = np.abs(phase) < PHASE_FOLD_WINDOW_HR
    phase, f, sec = phase[keep], f[keep], sec[keep]

    shifts_hr = []
    for s in unique_sectors:
        sel = sec == s
        bc, bf, be = bin_phase(phase[sel], f[sel])
        ax.errorbar(bc, (bf - 1) * 1e6, yerr=be * 1e6, fmt=sector_markers[s], color=sector_colors[s], ms=2.5, capsize=0, elinewidth=0.6, mec='none', zorder=2, alpha=0.3)

        asel = adopted_sectors == s
        oc_days = adopted_t0[asel] - (t0 + adopted_epochs[asel] * per)
        shift_hr = float(np.mean(oc_days)) * 24.0
        shifts_hr.append(shift_hr)

        # A float markevery keeps the marker spacing even after the shift
        ax.plot(model_ph_hr + shift_hr, (model_flux - 1) * 1e6, ls=sector_linestyles[s], color=sector_colors[s], lw=1.5, zorder=4, alpha=0.9, solid_capstyle='round', dash_capstyle='round', marker=sector_markers[s], markevery=0.14, ms=5.5, mec='white', mew=0.6, markerfacecolor=sector_colors[s])

    ax.axvline(0, color='0.6', lw=0.8, ls=':', alpha=0.8, zorder=1)
    ax.set_xlim(-PHASE_FOLD_WINDOW_HR, PHASE_FOLD_WINDOW_HR)
    ax.set_ylabel('Relative flux (ppm)')
    return (np.max(shifts_hr) - np.min(shifts_hr)) * 60.0


def _draw_depth_label(ax, depth_val_ppm):
    """Dotted guide line at the model depth, labelled on the y-axis."""
    depth_color = "#EE7733"  # Tol 'orange', not used by any sector

    ax.figure.canvas.draw()  # so the y-ticks are final
    ylo, yhi = ax.get_ylim()
    yticks = np.array([t for t in ax.get_yticks() if ylo <= t <= yhi])

    def _fmt(v):
        return f'{v:.0f}'.replace('-', '−')  # same minus sign as the tick labels

    ax.plot([-PHASE_FOLD_WINDOW_HR, 0], [depth_val_ppm, depth_val_ppm], ls=':', color=depth_color, lw=1.1, zorder=6, clip_on=False)

    # Blank the tick label that would overlap the depth label
    spacing = float(np.median(np.diff(yticks)))
    nearest = int(np.argmin(np.abs(yticks - depth_val_ppm)))
    if abs(yticks[nearest] - depth_val_ppm) / spacing <= 1 / 3:
        ax.set_yticks(yticks)
        labels = [_fmt(yt) for yt in yticks]
        labels[nearest] = ''
        ax.set_yticklabels(labels)

    ax.annotate(_fmt(depth_val_ppm), xy=(0, depth_val_ppm), xycoords=('axes fraction', 'data'), xytext=(-4, 0), textcoords='offset points', ha='right', va='center', fontsize=13, color=depth_color, zorder=7, annotation_clip=False)
    ax.figure.canvas.draw()


def make_figure(time, flat_flux, sector, epochs, t0s, t0_errs, epoch_sectors, rej_epochs, rej_t0s, rej_errs, eph):
    """Figure 1. Left: O-C vs. epoch for the literature (a) and refined (b) ephemerides. Right: phase folds on the literature (c) and refined (d) ephemerides."""
    refined_t0, refined_p = eph['t0'], eph['period']

    # Leave jackknife-rejected transits out of the phase folds
    keep = ~np.isin(np.round((time - LIT_T0_BTJD) / LIT_PERIOD), rej_epochs)
    t_fold, f_fold, sec_fold = time[keep], flat_flux[keep], sector[keep]

    unique_sectors = [int(s) for s in np.unique(sector)]
    sector_colors = dict(zip(unique_sectors, _SECTOR_PALETTE))
    sector_markers = dict(zip(unique_sectors, _SECTOR_MARKERS))
    sector_linestyles = dict(zip(unique_sectors, _SECTOR_LINESTYLES))

    # Model curve on a grid that runs past the panel edges, so shifted copies still reach them
    half_window_hr = PHASE_FOLD_WINDOW_HR + OUTLINE_EDGE_PAD_HR
    model_params = batman_params(period=refined_p)
    t_grid = np.linspace(-half_window_hr / 24.0, half_window_hr / 24.0, int(400 * half_window_hr / PHASE_FOLD_WINDOW_HR))
    model_flux = batman.TransitModel(model_params, t_grid).light_curve(model_params)
    model_ph_hr = t_grid * 24.0

    # The O-C panels get a broken x-axis, one piece per sector
    cluster_idx = _cluster_epochs(epochs)
    cluster_layout, cluster_widths = _cluster_layout(epochs, cluster_idx)
    col_widths, cluster_col_idx = _width_ratios(cluster_widths, cluster_layout, refined_p)

    # Top row holds the legends
    fig = plt.figure(figsize=(10.6, 7.0), layout='constrained')
    fig.get_layout_engine().set(h_pad=0.035, w_pad=0.035, hspace=0.03, wspace=0.025)
    outer_all = fig.add_gridspec(2, 1, height_ratios=[0.14, 1.0], hspace=0.01)
    ax_head = fig.add_subplot(outer_all[0])
    ax_head.axis('off')
    outer = outer_all[1].subgridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.10, wspace=0.06)

    oc_gs_lit = outer[0, 0].subgridspec(1, len(col_widths), width_ratios=col_widths, wspace=0.015)
    oc_gs_ref = outer[1, 0].subgridspec(1, len(col_widths), width_ratios=col_widths, wspace=0.015)
    ax_oc_lit, ax_oc_ref = [], []
    for col in cluster_col_idx:
        ax_oc_lit.append(fig.add_subplot(oc_gs_lit[0, col], sharey=ax_oc_lit[0] if ax_oc_lit else None))
    for col in cluster_col_idx:
        ax_oc_ref.append(fig.add_subplot(oc_gs_ref[0, col], sharey=ax_oc_ref[0] if ax_oc_ref else None))
    _style_broken_axes(ax_oc_lit, cluster_layout)
    _style_broken_axes(ax_oc_ref, cluster_layout)

    ax_ph_lit = fig.add_subplot(outer[0, 1])  # (c)
    ax_ph_ref = fig.add_subplot(outer[1, 1])  # (d)
    for ax in (ax_ph_lit, ax_ph_ref):
        ax.grid(True, axis='both', color='0.90', lw=0.6, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)

    # Same y-range for (a) and (b)
    oc_min_lit = (t0s - (LIT_T0_BTJD + epochs * LIT_PERIOD)) * 24.0 * 60.0
    oc_min_ref = (t0s - (refined_t0 + epochs * refined_p)) * 24.0 * 60.0
    oc_err_min = t0_errs * 24.0 * 60.0
    lo = float(np.min(np.concatenate([oc_min_lit - oc_err_min, oc_min_ref - oc_err_min])))
    hi = float(np.max(np.concatenate([oc_min_lit + oc_err_min, oc_min_ref + oc_err_min])))
    oc_ylim = (lo - 0.22 * (hi - lo), hi + 0.22 * (hi - lo))

    oc_args = (cluster_idx, cluster_layout, epochs, t0s, t0_errs, epoch_sectors, rej_epochs, rej_t0s, rej_errs)
    style_args = (unique_sectors, sector_colors, sector_markers, oc_ylim)
    h_rej, h_trend = _plot_oc_panel(ax_oc_lit, *oc_args, LIT_T0_BTJD, LIT_PERIOD, *style_args, fit_trend=True)
    _plot_oc_panel(ax_oc_ref, *oc_args, refined_t0, refined_p, *style_args, fit_trend=False)

    for ax in ax_oc_lit:
        ax.tick_params(axis='x', labelbottom=False)
    ax_oc_ref[len(cluster_idx) // 2].set_xlabel('Epoch, $E$')

    adopted = (epochs, t0s, epoch_sectors)
    fold_args = (unique_sectors, sector_colors, sector_markers, sector_linestyles, model_ph_hr, model_flux, *adopted)
    spread_lit = _plot_phase_panel(ax_ph_lit, t_fold, f_fold, sec_fold, LIT_PERIOD, LIT_T0_BTJD, *fold_args)
    spread_ref = _plot_phase_panel(ax_ph_ref, t_fold, f_fold, sec_fold, refined_p, refined_t0, *fold_args)

    # Offset of the literature ephemeris from the refined one by Jan 2027, shown in (c). The error is from the refined ephemeris covariance.
    days_since_j2000 = (datetime(2027, 1, 1) - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
    jan2027 = 2451545.0 + days_since_j2000 - 2457000.0  # BTJD
    E_fut = np.round((jan2027 - refined_t0) / refined_p)
    T_fut = refined_t0 + E_fut * refined_p
    E_fut_lit = np.round((T_fut - LIT_T0_BTJD) / LIT_PERIOD)
    future_shift_hr = (T_fut - (LIT_T0_BTJD + E_fut_lit * LIT_PERIOD)) * 24.0
    cov = eph['cov']
    future_var = (E_fut ** 2) * cov[0, 0] + 2.0 * E_fut * cov[0, 1] + cov[1, 1]
    future_sigma_min = float(np.sqrt(future_var) * 24.0 * 60.0)
    (h_future,) = ax_ph_lit.plot(model_ph_hr + future_shift_hr, (model_flux - 1) * 1e6, ls=(0, (1, 1)), color='#0077BB', lw=1.7, zorder=6, alpha=0.95, label='Jan 2027 (extrapolated)')
    ax_ph_lit.legend(handles=[h_future], loc='upper right', frameon=False, fontsize=9, handlelength=1.7, borderaxespad=0.3)

    ax_ph_lit.tick_params(axis='x', labelbottom=False)
    ax_ph_ref.set_xlabel('Time from mid-transit (hr)')

    # Same y-range for (c) and (d): the union of their autoscaled ranges
    ph_ylo = min(ax_ph_lit.get_ylim()[0], ax_ph_ref.get_ylim()[0])
    ph_yhi = max(ax_ph_lit.get_ylim()[1], ax_ph_ref.get_ylim()[1])
    ax_ph_lit.set_ylim(ph_ylo, ph_yhi)
    ax_ph_ref.set_ylim(ph_ylo, ph_yhi)

    handles = [h for h in [h_rej, h_trend] if h is not None]
    leg_main = ax_head.legend(handles=handles, loc='center left', bbox_to_anchor=(0.0, 0.5), bbox_transform=ax_head.transAxes, ncol=3, frameon=False, handlelength=1.7, columnspacing=1.3, handletextpad=0.5)
    ax_head.add_artist(leg_main)

    # Marker, line style and color per sector, so the legend works in grayscale
    sector_handles = [Line2D([0], [0], marker=sector_markers[s], linestyle=sector_linestyles[s], color=sector_colors[s], mec='white', mew=0.6, ms=7.5, lw=1.7, label=str(s)) for s in unique_sectors]
    ax_head.legend(handles=sector_handles, loc='center right', bbox_to_anchor=(1.0, 0.5), bbox_transform=ax_head.transAxes, title='TESS sector', ncol=1, frameon=False, fontsize=13, title_fontsize=13, handlelength=2.6, handleheight=1.1, columnspacing=1.3, labelspacing=0.45)

    _draw_depth_label(ax_ph_ref, float((np.min(model_flux) - 1.0) * 1e6))

    panels = [ax_oc_lit[0], ax_oc_ref[0], ax_ph_lit, ax_ph_ref]
    for ax, label in zip(panels, ['(a)', '(b)', '(c)', '(d)']):
        ax.annotate(label, xy=(0, 1), xycoords='axes fraction', xytext=(6, -6), textcoords='offset points', fontweight='bold', fontsize=13, ha='left', va='top', zorder=10, annotation_clip=False)

    fig.savefig('Figure_1.png', dpi=400, bbox_inches='tight')
    fig.savefig('Figure_1.pdf', bbox_inches='tight')
    plt.close(fig)

    return {'spread_lit': spread_lit, 'spread_ref': spread_ref, 'future_label': 'Jan 2027', 'future_offset_min': future_shift_hr * 60.0, 'future_offset_sigma_min': future_sigma_min}


def main():
    time, flux, flux_err, sector = fetch_data()
    flat_flux = detrend(time, flux, sector)
    t0_seed, period_seed = bls_seed(time, flat_flux)

    epochs, t0s, t0_errs, snrs, betas, transit_sectors = fit_transit_times(time, flux, flux_err, sector, t0_seed, period_seed)
    used = jackknife_mask(epochs, t0s, t0_errs)
    rej = ~used

    eph = fit_linear_ephemeris(epochs[used], t0s[used], t0_errs[used])
    curvature = curvature_test(epochs[used], t0s[used], t0_errs[used], eph['period'])
    depth = measure_transit_depth(time, flat_flux, flux_err, eph['t0'], eph['period'])

    oc_lit = (t0s[used] - (LIT_T0_BTJD + epochs[used] * LIT_PERIOD)) * 24.0 * 60.0
    oc_ref = (t0s[used] - (eph['t0'] + epochs[used] * eph['period'])) * 24.0 * 60.0
    rms_lit = float(np.sqrt(np.mean(oc_lit ** 2)))
    rms_ref = float(np.sqrt(np.mean(oc_ref ** 2)))

    with plt.rc_context(FIGURE_RC):
        fig_stats = make_figure(time, flat_flux, sector, epochs[used], t0s[used], t0_errs[used], transit_sectors[used], epochs[rej], t0s[rej], t0_errs[rej], eph)

    write_oc_table(epochs, t0s, t0_errs, snrs, transit_sectors, betas, used, eph)
    write_metadata(eph, curvature, depth, fig_stats, rms_lit, rms_ref, epochs[used], int(np.sum(rej)), sector, t0s[used].max() - t0s[used].min(), time.max() - time.min())

    print(f'{np.sum(used)} transits used, {np.sum(rej)} rejected')
    print(f"T0 = {eph['t0']:.5f} +/- {eph['t0_err']:.5f} BTJD, "
          f"P = {eph['period']:.7f} +/- {eph['period_err']:.7f} d, chi2/dof = {eph['scale_factor']:.2f}")
    print(f"vs. Kokori et al. (2026): dP = {eph['delta_p_sec']:+.1f} s ({eph['delta_p_sigma']:+.2f} sigma), "
          f"dT0 = {eph['t0_diff_min']:+.2f} min ({eph['t0_diff_sigma']:+.2f} sigma)")
    print(f'O-C RMS: {rms_lit:.2f} min (literature), {rms_ref:.2f} min (refined)')
    print(f"Sector spread in the phase fold: {fig_stats['spread_lit']:.1f} min (literature), "
          f"{fig_stats['spread_ref']:.1f} min (refined)")
    print(f"Literature ephemeris offset by Jan 2027: {fig_stats['future_offset_min']:+.1f} "
          f"+/- {fig_stats['future_offset_sigma_min']:.1f} min")
    print(f"Curvature: delta BIC = {curvature['delta_bic']:+.2f}, p = {curvature['p_value']:.2f}, "
          f"Pdot = {curvature['pdot_ms_yr']:+.1f} +/- {curvature['pdot_err_ms_yr']:.1f} ms/yr "
          f"(3-sigma limit ~{curvature['pdot_limit_ms_yr']:.0f} ms/yr)")
    print(f"Free Rp/Rs = {depth['rp_rs_fit']:.4f} +/- {depth['rp_rs_fit_err']:.4f} "
          f"(depth {depth['depth_ppm']:.0f} +/- {depth['depth_err_ppm']:.0f} ppm)")


if __name__ == '__main__':
    main()
