from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Union

import numpy as np
import pandas as pd
from nkutils import read_named_range_to_dataframe as rnr2df


VALUATION_YEAR = 2026
PROJECTION_TERM = 40
NA_SENTINEL = -999.0
CASH_FACTOR = 0.25
RETIREMENT_AGE = 65
INFLATION_INPUT_FILE = "infl_toy_paths.csv"

qx_scheme = {
    65:0.0072,66:0.00774,67:0.00828,68:0.00891,69:0.00963,
    70:0.01044,71:0.01134,72:0.01242,73:0.01359,74:0.01494,
    75:0.01647,76:0.01818,77:0.02016,78:0.02232,79:0.02484,
    80:0.02763,81:0.03078,82:0.03429,83:0.03825,84:0.04266,
    85:0.04761,86:0.05310,87:0.05922,88:0.06606,89:0.07362,
    90:0.08208,91:0.09144,92:0.10179,93:0.11331,94:0.12609,
    95:0.14022,96:0.15588,97:0.17316,98:0.19233,99:0.21348,
    100:0.23679,101:0.26100,102:0.28800,103:0.31770,104:0.35010,
    105:0.38520,106:0.42300,107:0.46350,108:0.50670,109:0.55260,
    110:0.60030,111:0.64890,112:0.69750,113:0.74340,114:0.78390,
    115:0.81900,116:0.84780,117:0.86850,118:0.88380,119:0.89370,120:1.0,
}


@dataclass(frozen=True)
class ProjectionConfig:
    valuation_year: int = VALUATION_YEAR
    projection_term: int = PROJECTION_TERM
    cash_factor: float = CASH_FACTOR


REQUIRED_COLUMNS = [
    "age_vd8",
    "status",
    "members_alive",
    "leaving_year",
    "retirement_year",
    "gmp_fixed_revaluation_rate",
    "gmp_post88_amount_leave",
    "amount_XS_86to97_leave",
    "amount_97_05_rpi_0_5_leave",
    "amount_05_09_leave",
    "amount_09_22_cpi_0_25_leave",
    "pre09_cap5pc_hist_factor",
    "post09_cap2_5pc_hist_factor",
    "pension_fixed_at_valnd8",
    "pension_gmp_post88_at_valnd8",
    "pension_LPI5pc_at_valnd8",
    "pension_LPI2_5pc_at_valnd8",
]


OUTPUT_COLUMNS = [
    "total_cashflow",
    "pensioner_fixed_pension",
    "pensioner_cpi3_gmp_pension",
    "pensioner_rpi5_pension",
    "pensioner_cpi2p5_pension",
    "deferred_retirement_cash",
    "deferred_fixed_pension",
    "deferred_cpi3_gmp_pension",
    "deferred_rpi5_pension_9705",
    "deferred_cpi2p5_pension_0509",
    "deferred_cpi2p5_pension_0922",
    "defer_exp_rpi5_pre97_xs",
    "defer_exp_rpi5_9705",
    "defer_exp_rpi5_0509",
    "defer_exp_cpi2p5_0922",
]


def _load_dataset(dataset: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
    else:
        df = pd.read_csv(dataset)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df.reset_index(drop=True)


def _load_rpi_cpi_paths(
    rpi_file: Union[str, Path],
    valuation_year: int,
    projection_term: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rpi = pd.read_csv(str(rpi_file)).to_numpy(dtype=float)
    if rpi.ndim != 2:
        raise ValueError("RPI input must be 2D")
    if rpi.shape[1] < projection_term:
        raise ValueError(
            f"RPI input has {rpi.shape[1]} years, expected at least {projection_term}."
        )
    rpi = rpi[:, :projection_term]
    cpi = 0.85 * rpi
    years = np.arange(valuation_year, valuation_year + projection_term, dtype=int)
    return rpi, cpi, years


def _build_qx_vector(qx_table: Mapping[int, float], max_age: int = 130) -> np.ndarray:
    qx_vec = np.zeros(max_age + 1, dtype=float)
    lo = min(qx_table)
    hi = max(qx_table)
    for age in range(max_age + 1):
        if age < lo:
            qx_vec[age] = float(qx_table[lo])
        elif age > hi:
            qx_vec[age] = 1.0
        else:
            qx_vec[age] = float(qx_table[age])
    return qx_vec


def _cumprod_base_one(step_arr: np.ndarray) -> np.ndarray:
    out = np.ones_like(step_arr, dtype=float)
    if step_arr.shape[1] > 1:
        out[:, 1:] = np.cumprod(step_arr[:, :-1], axis=1)
    return out


def _build_alive_matrices(  # VECTORISED VERSION
    start_alive: np.ndarray,
    start_age: np.ndarray,
    qx_vec: np.ndarray,
    years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(start_alive)
    T = len(years)

    year_offsets = years - VALUATION_YEAR                      # (T,)
    age_matrix = start_age[:, None] + year_offsets[None, :]   # (n_rows, T)
    capped_age = np.minimum(age_matrix, len(qx_vec) - 1)

    qx_matrix = qx_vec[capped_age]
    qx_matrix = np.where(age_matrix < RETIREMENT_AGE, 0.0, qx_matrix)
    one_year_survival = 1.0 - qx_matrix                       # (n_rows, T)

    alive_start = np.empty((n_rows, T), dtype=float)
    alive_start[:, 0] = start_alive
    if T > 1:
        alive_start[:, 1:] = (
            start_alive[:, None]
            * np.cumprod(one_year_survival[:, :-1], axis=1)
        )

    alive_mid = alive_start * (1.0 - 0.5 * qx_matrix)

    return alive_start, alive_mid


def _safe_power(base: float, exponent: np.ndarray) -> np.ndarray:
    exp = np.maximum(exponent, 0)
    return np.power(base, exp, dtype=float)

def _annual_capped_steps(rate_paths: np.ndarray, cap: float) -> np.ndarray:
    return 1.0 + np.minimum(rate_paths, cap)

def _precompute_scaffold(
    df: pd.DataFrame,
    qx_table: Mapping[int, float],
    config: ProjectionConfig,
) -> dict[str, object]:
    years = np.arange(config.valuation_year, config.valuation_year + config.projection_term, dtype=int)
    qx_vec = _build_qx_vector(qx_table)

    pens = df[df["status"] == "Pensioner"].copy().reset_index(drop=True)
    defs = df[df["status"] == "Deferred"].copy().reset_index(drop=True)

    scaffold: dict[str, object] = {
        "years": years,
        "pensioners": pens,
        "deferreds": defs,
    }

    if len(pens):
        _, p_alive_mid = _build_alive_matrices(
            pens["members_alive"].to_numpy(dtype=float),
            pens["age_vd8"].to_numpy(dtype=int),
            qx_vec,
            years,
        )
    else:
        p_alive_mid = np.zeros((0, len(years)), dtype=float)
    scaffold["pensioner_alive_mid"] = p_alive_mid

    if len(defs):
        ret_year = defs["retirement_year"].to_numpy(dtype=int)
        leave_year = defs["leaving_year"].to_numpy(dtype=int)
        ret_idx = np.clip(ret_year - config.valuation_year, 0, config.projection_term - 1)

        before_mask = (years[None, :] < ret_year[:, None]).astype(float)
        retire_mask = (years[None, :] == ret_year[:, None]).astype(float)
        post_mask = (years[None, :] >= ret_year[:, None]).astype(float)

        d_alive_mid_post = np.zeros((len(defs), len(years)), dtype=float)
        d_alive_start_post = np.zeros((len(defs), len(years)), dtype=float)
        for i in range(len(defs)):
            n = float(defs.loc[i, "members_alive"])
            for j, year in enumerate(years):
                if year < ret_year[i]:
                    continue
                age = 65 + (year - ret_year[i])
                qx = qx_vec[min(age, len(qx_vec) - 1)]
                d_alive_start_post[i, j] = n
                d_alive_mid_post[i, j] = n * (1.0 - 0.5 * qx)
                n *= (1.0 - qx)
    else:
        leave_year = np.zeros(0, dtype=int)
        ret_idx = np.zeros(0, dtype=int)
        before_mask = np.zeros((0, len(years)), dtype=float)
        retire_mask = np.zeros((0, len(years)), dtype=float)
        post_mask = np.zeros((0, len(years)), dtype=float)
        d_alive_mid_post = np.zeros((0, len(years)), dtype=float)
        d_alive_start_post = np.zeros((0, len(years)), dtype=float)

    scaffold["deferred_leave_year"] = leave_year
    scaffold["deferred_ret_idx"] = ret_idx
    scaffold["deferred_before_mask"] = before_mask
    scaffold["deferred_retire_mask"] = retire_mask
    scaffold["deferred_post_mask"] = post_mask
    scaffold["deferred_alive_mid_post"] = d_alive_mid_post
    scaffold["deferred_alive_start_post"] = d_alive_start_post
    return scaffold


def project_cashflows_all_sims(
    dataset: Union[str, Path, pd.DataFrame],
    rpi_file: Union[str, Path] = INFLATION_INPUT_FILE,
    qx_table: Mapping[int, float] = qx_scheme,
    config: ProjectionConfig = ProjectionConfig(),
) -> pd.DataFrame:
    df = _load_dataset(dataset)
    scaffold = _precompute_scaffold(df, qx_table, config)
    rpi_rates, cpi_rates, years = _load_rpi_cpi_paths(rpi_file, config.valuation_year, config.projection_term)

    n_sims, T = rpi_rates.shape

    # stochastic arrays built once for all sims
    rpi_cum = _cumprod_base_one(1.0 + rpi_rates)
    cpi_cum = _cumprod_base_one(1.0 + cpi_rates)
    rpi5_pay_cum = _cumprod_base_one(_annual_capped_steps(rpi_rates, 0.05))
    cpi25_pay_cum = _cumprod_base_one(_annual_capped_steps(cpi_rates, 0.025))
    cpi3_pay_cum = _cumprod_base_one(_annual_capped_steps(cpi_rates, 0.03))

    out = {k: np.zeros((n_sims, T), dtype=float) for k in OUTPUT_COLUMNS}

    # Existing pensioners
    pens: pd.DataFrame = scaffold["pensioners"]
    p_alive_mid: np.ndarray = scaffold["pensioner_alive_mid"]
    if len(pens):
        amt_fixed = pens["pension_fixed_at_valnd8"].to_numpy(dtype=float)
        amt_c3 = pens["pension_gmp_post88_at_valnd8"].to_numpy(dtype=float)
        amt_r5 = pens["pension_LPI5pc_at_valnd8"].to_numpy(dtype=float)
        amt_c25 = pens["pension_LPI2_5pc_at_valnd8"].to_numpy(dtype=float)

        out["pensioner_fixed_pension"][:] = np.broadcast_to((p_alive_mid * amt_fixed[:, None]).sum(axis=0), (n_sims, T))
        out["pensioner_cpi3_gmp_pension"][:] = np.einsum("rt,st->st", p_alive_mid * amt_c3[:, None], cpi3_pay_cum)
        out["pensioner_rpi5_pension"][:] = np.einsum("rt,st->st", p_alive_mid * amt_r5[:, None], rpi5_pay_cum)
        out["pensioner_cpi2p5_pension"][:] = np.einsum("rt,st->st", p_alive_mid * amt_c25[:, None], cpi25_pay_cum)

    # Deferreds
    defs: pd.DataFrame = scaffold["deferreds"]
    if len(defs):
        leave_year = scaffold["deferred_leave_year"]
        ret_idx = scaffold["deferred_ret_idx"]
        before_mask = scaffold["deferred_before_mask"]
        retire_mask = scaffold["deferred_retire_mask"]
        post_mask = scaffold["deferred_post_mask"]
        alive_mid_post = scaffold["deferred_alive_mid_post"]

        hist_pre09 = defs["pre09_cap5pc_hist_factor"].to_numpy(dtype=float)
        hist_post09 = defs["post09_cap2_5pc_hist_factor"].to_numpy(dtype=float)
        duration = (defs["retirement_year"].to_numpy(dtype=int) - leave_year).astype(int)

        amt_gmp = defs["gmp_post88_amount_leave"].to_numpy(dtype=float)
        amt_pre97 = defs["amount_XS_86to97_leave"].to_numpy(dtype=float)
        amt_9705 = defs["amount_97_05_rpi_0_5_leave"].to_numpy(dtype=float)
        amt_0509 = defs["amount_05_09_leave"].to_numpy(dtype=float)
        amt_0922 = defs["amount_09_22_cpi_0_25_leave"].to_numpy(dtype=float)
        gmp_rate = defs["gmp_fixed_revaluation_rate"].to_numpy(dtype=float)

        rpi_to_ret = rpi_cum[:, ret_idx]
        cpi_to_ret = cpi_cum[:, ret_idx]

        pre09_ret_factor = np.minimum(hist_pre09[None, :] * rpi_to_ret, _safe_power(1.05, duration)[None, :])
        post09_ret_factor = np.minimum(hist_post09[None, :] * cpi_to_ret, _safe_power(1.025, duration)[None, :])
        gmp_ret_factor = np.power(1.0 + gmp_rate, duration, dtype=float)[None, :]

        ret_pre97 = amt_pre97[None, :] * pre09_ret_factor
        ret_9705 = amt_9705[None, :] * pre09_ret_factor
        ret_0509 = amt_0509[None, :] * pre09_ret_factor
        ret_0922 = amt_0922[None, :] * post09_ret_factor
        ret_gmp = amt_gmp[None, :] * gmp_ret_factor

        # deferment exposures before retirement
        cap5_year = _safe_power(1.05, years[None, :] - leave_year[:, None])
        cap25_year = _safe_power(1.025, years[None, :] - leave_year[:, None])
        pre09_factor_by_year = np.minimum(
            hist_pre09[None, :, None] * rpi_cum[:, None, :],
            cap5_year[None, :, :],
        )
        post09_factor_by_year = np.minimum(
            hist_post09[None, :, None] * cpi_cum[:, None, :],
            cap25_year[None, :, :],
        )

        out["defer_exp_rpi5_pre97_xs"][:] = np.einsum("srt,rt->st", amt_pre97[None, :, None] * pre09_factor_by_year, before_mask)
        out["defer_exp_rpi5_9705"][:] = np.einsum("srt,rt->st", amt_9705[None, :, None] * pre09_factor_by_year, before_mask)
        out["defer_exp_rpi5_0509"][:] = np.einsum("srt,rt->st", amt_0509[None, :, None] * pre09_factor_by_year, before_mask)
        out["defer_exp_cpi2p5_0922"][:] = np.einsum("srt,rt->st", amt_0922[None, :, None] * post09_factor_by_year, before_mask)

        # retirement cash
        total_ret = ret_pre97 + ret_9705 + ret_0509 + ret_0922 + ret_gmp
        out["deferred_retirement_cash"][:] = np.einsum("sr,rt->st", config.cash_factor * total_ret, retire_mask)

        # pension in retirement year and after
        pay_fixed_base = (1.0 - config.cash_factor) * ret_pre97
        pay_c3_base = (1.0 - config.cash_factor) * ret_gmp
        pay_r5_base = (1.0 - config.cash_factor) * ret_9705
        pay_c25_0509_base = (1.0 - config.cash_factor) * ret_0509
        pay_c25_0922_base = (1.0 - config.cash_factor) * ret_0922

        rpi5_from_ret = np.zeros((n_sims, len(defs), T), dtype=float)
        cpi25_from_ret = np.zeros((n_sims, len(defs), T), dtype=float)
        cpi3_from_ret = np.zeros((n_sims, len(defs), T), dtype=float)
        for j, r in enumerate(ret_idx):
            rpi5_from_ret[:, j, :] = rpi5_pay_cum / rpi5_pay_cum[:, [r]]
            cpi25_from_ret[:, j, :] = cpi25_pay_cum / cpi25_pay_cum[:, [r]]
            cpi3_from_ret[:, j, :] = cpi3_pay_cum / cpi3_pay_cum[:, [r]]

        rpi5_from_ret *= post_mask[None, :, :]
        cpi25_from_ret *= post_mask[None, :, :]
        cpi3_from_ret *= post_mask[None, :, :]

        out["deferred_fixed_pension"][:] = np.einsum("sr,rt->st", pay_fixed_base, alive_mid_post)
        out["deferred_cpi3_gmp_pension"][:] = np.einsum("sr,srt,rt->st", pay_c3_base, cpi3_from_ret, alive_mid_post)
        out["deferred_rpi5_pension_9705"][:] = np.einsum("sr,srt,rt->st", pay_r5_base, rpi5_from_ret, alive_mid_post)
        out["deferred_cpi2p5_pension_0509"][:] = np.einsum("sr,srt,rt->st", pay_c25_0509_base, cpi25_from_ret, alive_mid_post)
        out["deferred_cpi2p5_pension_0922"][:] = np.einsum("sr,srt,rt->st", pay_c25_0922_base, cpi25_from_ret, alive_mid_post)

    out["total_cashflow"] = (
        out["pensioner_fixed_pension"]
        + out["pensioner_cpi3_gmp_pension"]
        + out["pensioner_rpi5_pension"]
        + out["pensioner_cpi2p5_pension"]
        + out["deferred_retirement_cash"]
        + out["deferred_fixed_pension"]
        + out["deferred_cpi3_gmp_pension"]
        + out["deferred_rpi5_pension_9705"]
        + out["deferred_cpi2p5_pension_0509"]
        + out["deferred_cpi2p5_pension_0922"]
    )

    parts = []
    for sim in range(n_sims):
        sim_df = pd.DataFrame({
            "sim": sim,
            "year": years,
            "cashflow_date": [f"{y}-06-30" for y in years],
        })
        for col in OUTPUT_COLUMNS:
            sim_df[col] = out[col][sim, :]
        parts.append(sim_df)

    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    dataset_path = "Schemes/lpi_scheme_dataset_2026_with_amounts_v2.csv"
    result = project_cashflows_all_sims(dataset=dataset_path)
    result.to_csv("projected_cashflows.csv", index=False)
    pivot_cf = result.pivot(index="sim", columns="year", values="total_cashflow")
    pivot_cf.to_csv("projected_total_cashflows_matrix.csv")
    print(result.head())
