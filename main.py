"""GTA VI trophy earn-rate forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy, five quantiles of its Exophase earn rate at the
case target.  The model works in COUNTS, not just rates, because Exophase
refreshes its counts in jumps: when the tracked-player base grows, every
rate but the 100% one falls (the newly tracked players hold fewer trophies),
so a rate-only persistence forecast is systematically too high.

Method, per trophy i and tracked-player count P(t):
  * t is hours since the game's release (2026-11-19T00:00Z).
  * Deduplicate consecutive identical reads; keep every refresh event
    (a read whose players_tracked changed).
  * P(t): least-squares fit of ln P against ln(1 + t) on distinct-P points
    (last-24h window when it holds >= 3 distinct points, else all points),
    extrapolated to the target, capped to [P_now, 8 * P_now].
  * Earn counts E_i(t) = r_i * P / 100, running-max enforced (counts never
    fall).  Per trophy, least-squares slope s_i of ln E_i against ln(1 + t),
    shrunk toward the cross-trophy median slope with weight n / (n + 3),
    then E_i(T) = E_i(now) * ((1 + T) / (1 + t_now)) ** s_i, floored at
    E_i(now) and capped at 0.999 * P(T).  r_i(T) = 100 * E_i(T) / P(T).
  * Flat history (no players_tracked change at all, as on launch night):
    assume one refresh before the target, sized like the first refresh the
    house recorded: the player base grows about 6x (638 -> 3,868) and the
    new cohort holds each trophy at about 0.07x the current rate.  That
    0.07 comes from the recorded totals: trophies earned 943 -> 4,276,
    and sum(rates) * P / 100 equals trophies earned exactly, so outside
    the one 100% trophy the non-100 rates sum 47.78 before the refresh
    and 10.55 after it, a ratio of 0.2208; solving
    (1 + 5.06 * c) / 6.06 = 0.2208 gives c ~= 0.07.  A uniform 0.2208x
    scaling of the practice rates reproduces the house's stated practice
    baseline (loss ~1.51, typical miss ~0.68pp, coverage 0) at 1.40 /
    0.68pp / 0, which corroborates it.  Flat histories also get a sigma
    floor of 0.35, since a refresh's size is itself uncertain.
  * Spread on the logit scale: sigma = sqrt(sigma_fit^2
    + (0.35 * sqrt(d))^2 + sigma_P^2), d the horizon in days, sigma_fit the
    per-trophy residual std of the ln E fit shrunk to the global median,
    sigma_P the residual std of the ln P fit; sigma clamped to [0.25, 1.5].
    Quantiles are inv_logit(logit(median) + z_tau * sigma).
  * Edges: a trophy at 100% stays near 100 (median 99.9, sigma 0.15);
    a trophy at 0% throughout gets median 0.05 with sigma 1.2.

Deterministic, standard library only, no network, no wall clock, no
randomness.  Reads every *.in in the input directory (sorted) and writes
one *.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# New-cohort assumptions used only when history is perfectly flat.
# Calibrated to the first refresh the house recorded (see module docstring):
# the player base grew ~6x and the new cohort's per-trophy rate is ~0.07x.
FLAT_PLAYER_GROWTH = 6.0
FLAT_NEW_COHORT_RATE = 0.07
# Extra logit spread for a flat history: a refresh's size is uncertain.
FLAT_SIGMA_FLOOR = 0.35


def parse_time(text):
    """Parse an ISO 8601 timestamp (trailing Z allowed) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def least_squares(xs, ys):
    """Return (slope, intercept, residual std) of the y ~ x line, or None."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        resid = [(y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)]
        std = math.sqrt(sum(resid) / (n - 2))
    else:
        std = 0.0
    return slope, intercept, std


def distinct_points(reads, value_fn):
    """(x, y) points from reads whose value changed vs the previous read."""
    points = []
    previous = None
    for read in reads:
        value = value_fn(read)
        if value is None:
            continue
        if previous is None or value != previous:
            points.append((math.log1p(read["t"]), value))
            previous = value
    return points


def recent_points(points, t_now, window_hours=24.0):
    """Points within the last window_hours before t_now (x is ln(1+t))."""
    cutoff = math.log1p(max(0.0, t_now - window_hours))
    kept = [pt for pt in points if pt[0] >= cutoff]
    return kept if len(kept) >= 3 else points


def forecast_case(case, name):
    if case.get("levels", LEVELS) != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    t_target = hours_since_release(target)

    if not history:
        # No data at all: emit a valid, very wide forecast rather than crash.
        return {"trophies": [
            {"id": tr["id"], "q": [round4(inv_logit(logit(1.0) + z * 1.0)) for z in Z]}
            for tr in trophies
        ]}

    # --- Normalise reads: drop consecutive duplicates, keep refresh events.
    reads = []
    prev_key = None
    for raw in history:
        key = (raw["players_tracked"], tuple(raw["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        reads.append({
            "t": hours_since_release(parse_time(raw["at"])),
            "players": float(raw["players_tracked"]),
            "rates": raw["rates"],
        })
    last = reads[-1]
    t_now = last["t"]
    p_now = last["players"]
    n_trophies = len(trophies)

    # Last non-null rate per trophy (usually the last read's rate).
    last_rate = []
    for i in range(n_trophies):
        rate = None
        for read in reversed(reads):
            if read["rates"][i] is not None:
                rate = float(read["rates"][i])
                break
        last_rate.append(0.0 if rate is None else rate)

    flat = all(read["players"] == reads[0]["players"] for read in reads)

    # --- Player-count model: ln P ~ ln(1 + t).
    p_points = distinct_points(reads, lambda r: math.log(r["players"]))
    p_window = recent_points(p_points, t_now)
    p_fit = least_squares([x for x, _ in p_window], [y for _, y in p_window])
    if p_fit is not None and not flat:
        p_slope, p_intercept, sigma_p = p_fit
        p_target = math.exp(p_intercept + p_slope * math.log1p(t_target))
    else:
        p_target = p_now
        sigma_p = 0.0
    if flat:
        p_target = FLAT_PLAYER_GROWTH * p_now
    p_target = min(max(p_target, p_now), 8.0 * p_now)

    # --- Per-trophy earn-count series (running max, counts never fall).
    counts = []  # counts[i] = list of (t, E) with E > 0, distinct E values
    for i in range(n_trophies):
        series = []
        running_max = 0.0
        for read in reads:
            rate = read["rates"][i]
            if rate is None:
                continue
            e = float(rate) * read["players"] / 100.0
            if e > running_max:
                running_max = e
            series.append((read["t"], running_max))
        # Keep only points where the count changed (distinct E).
        distinct = []
        prev_e = None
        for t, e in series:
            if e > 0.0 and (prev_e is None or e != prev_e):
                distinct.append((math.log1p(t), math.log(e)))
                prev_e = e
        counts.append(distinct)

    # --- Per-trophy slopes of ln E ~ ln(1 + t), shrunk to the median.
    raw_slopes = []
    raw_sigmas = []
    for series in counts:
        window = recent_points(series, t_now)
        fit = least_squares([x for x, _ in window], [y for _, y in window])
        if fit is None:
            raw_slopes.append(None)
            raw_sigmas.append(None)
        else:
            raw_slopes.append(fit[0])
            raw_sigmas.append(fit[2])
    known_slopes = [s for s in raw_slopes if s is not None]
    median_slope = median(known_slopes) if known_slopes else 0.0
    known_sigmas = [s for s in raw_sigmas if s is not None]
    median_sigma = median(known_sigmas) if known_sigmas else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        rate_now = last_rate[i]
        if rate_now >= 99.995:
            # Everybody holds it and refreshes leave it at 100%.
            centre, sigma = 99.9, 0.15
        elif rate_now <= 0.0:
            # Never earned by anyone tracked so far.
            centre, sigma = 0.05, 1.2
        else:
            e_now = rate_now * p_now / 100.0
            if flat:
                # One refresh before the target, sized like the first one
                # the house recorded: the base grows ~6x and the new
                # cohort holds each trophy at ~0.07x the current rate.
                e_target = e_now + (FLAT_PLAYER_GROWTH - 1.0) * p_now \
                    * FLAT_NEW_COHORT_RATE * rate_now / 100.0
                sigma_fit = 0.0
            else:
                n = len(counts[i])
                slope = raw_slopes[i]
                if slope is None:
                    slope = median_slope
                else:
                    slope = (n * slope + 3.0 * median_slope) / (n + 3.0)
                slope = max(0.0, slope)
                e_target = e_now * ((1.0 + t_target) / (1.0 + t_now)) ** slope
                e_target = max(e_target, e_now)
                e_target = min(e_target, 0.999 * p_target)
                fit_sigma = raw_sigmas[i]
                if fit_sigma is None:
                    sigma_fit = median_sigma
                else:
                    sigma_fit = (n * fit_sigma + 3.0 * median_sigma) / (n + 3.0)
            centre = clamp_rate(100.0 * e_target / p_target)
            sigma = math.sqrt(
                sigma_fit ** 2
                + (0.35 * math.sqrt(horizon_days)) ** 2
                + sigma_p ** 2
            )
            sigma = min(1.5, max(0.25, sigma))
            if flat:
                # A refresh's size is uncertain: widen the flat-case band.
                sigma = max(sigma, FLAT_SIGMA_FLOOR)
        centre_logit = logit(centre)
        quantiles = []
        prev = None
        for z in Z:
            q = round4(inv_logit(centre_logit + z * sigma))
            q = min(99.99, max(0.01, q))
            if prev is not None and q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
