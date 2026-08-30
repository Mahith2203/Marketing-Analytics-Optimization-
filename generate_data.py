"""
Generates a 2-year, multi-geo, multi-channel synthetic marketing dataset
with realistic adstock (carryover) and saturation (diminishing returns)
dynamics baked in, plus a geo-holdout incrementality experiment on one
channel. Ground-truth parameters are stored so the modeling pipeline in
analysis.py can be validated against them.
"""

import numpy as np
import pandas as pd
import json

np.random.seed(42)

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
N_DAYS = 730  # 2 years
START_DATE = pd.Timestamp("2023-01-01")
dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")

GEOS = [f"GEO_{i:02d}" for i in range(1, 11)]  # 10 geos (e.g. DMAs)
GEO_SIZE = np.random.dirichlet(np.ones(len(GEOS)) * 3) * 10  # relative market size weights

CHANNELS = {
    # channel: (true_decay(adstock retention), true_alpha(saturation curvature),
    #           true_half_sat(spend at half-max effect, national $/day), true_beta(max $ revenue/day at full saturation))
    "TV_OTT":       dict(decay=0.55, alpha=0.9, half_sat=9000,  beta=42000, base_spend=7000, spend_noise=0.15),
    "Paid_Search":  dict(decay=0.15, alpha=1.1, half_sat=6000,  beta=28000, base_spend=5500, spend_noise=0.12),
    "Paid_Social":  dict(decay=0.35, alpha=1.0, half_sat=5000,  beta=19000, base_spend=4500, spend_noise=0.18),
    "Display":      dict(decay=0.25, alpha=1.3, half_sat=4000,  beta=6000,  base_spend=3800, spend_noise=0.20),
    "Email":        dict(decay=0.05, alpha=0.8, half_sat=1400,  beta=4200,  base_spend=650,  spend_noise=0.05),
    "Affiliate":    dict(decay=0.10, alpha=1.0, half_sat=2800,  beta=8500,  base_spend=2600, spend_noise=0.10),
    "Influencer":   dict(decay=0.40, alpha=1.2, half_sat=2500,  beta=5200,  base_spend=2200, spend_noise=0.25),
    "Direct_Mail":  dict(decay=0.60, alpha=1.4, half_sat=1800,  beta=3600,  base_spend=1500, spend_noise=0.10),
}

with open("data/ground_truth_params.json", "w") as f:
    json.dump(CHANNELS, f, indent=2)

# ---------------------------------------------------------------
# National daily spend per channel (with seasonality + noise)
# ---------------------------------------------------------------
t = np.arange(N_DAYS)
weekly = 1 + 0.08 * np.sin(2 * np.pi * t / 7)
yearly_holiday = 1 + 0.55 * np.exp(-((t % 365 - 330) ** 2) / (2 * 20 ** 2))  # Nov/Dec bump
trend = 1 + 0.00035 * t  # slow organic budget growth

spend = {}
for ch, p in CHANNELS.items():
    noise = np.random.normal(1, p["spend_noise"], N_DAYS)
    s = p["base_spend"] * weekly * yearly_holiday * trend * noise
    s = np.clip(s, p["base_spend"] * 0.2, None)
    spend[ch] = s

# ---------------------------------------------------------------
# GEO HOLDOUT EXPERIMENT: pause Paid_Social in 3 test geos, days 400-434
# ---------------------------------------------------------------
HOLDOUT_CHANNEL = "Paid_Social"
HOLDOUT_START, HOLDOUT_END = 400, 434  # 5-week holdout window
TEST_GEOS = GEOS[:3]
CONTROL_GEOS = GEOS[3:]

# ---------------------------------------------------------------
# Build adstocked + saturated response, then allocate to geos
# ---------------------------------------------------------------
def adstock(x, decay):
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + decay * out[i - 1]
    return out

def hill_saturation(x, half_sat, alpha, beta):
    return beta * (x ** alpha) / (x ** alpha + half_sat ** alpha)

channel_effect_national = {}
for ch, p in CHANNELS.items():
    ad = adstock(spend[ch], p["decay"])
    eff = hill_saturation(ad, p["half_sat"], p["alpha"], p["beta"])
    channel_effect_national[ch] = eff

baseline_national = 55000 * weekly * yearly_holiday * (1 + 0.0006 * t) + np.random.normal(0, 2500, N_DAYS)

rows = []
for gi, geo in enumerate(GEOS):
    geo_w = GEO_SIZE[gi] / GEO_SIZE.sum()
    for di, d in enumerate(dates):
        row = {"date": d, "geo": geo}
        geo_spend_total = 0
        geo_revenue = baseline_national[di] * geo_w * np.random.normal(1, 0.03)
        for ch in CHANNELS:
            ch_spend = spend[ch][di] * geo_w * np.random.normal(1, 0.05)
            # apply geo holdout: zero out spend for test geos in the window
            if ch == HOLDOUT_CHANNEL and geo in TEST_GEOS and HOLDOUT_START <= di < HOLDOUT_END:
                ch_spend = 0.0
            row[f"spend_{ch}"] = round(max(ch_spend, 0), 2)
            geo_spend_total += ch_spend
            # geo-level effect approximated as national effect scaled by geo weight and
            # this geo's share of channel spend (down-weighted if holdout zeroed it)
            geo_share_of_national_spend = ch_spend / (spend[ch][di] + 1e-9)
            row[f"revenue_{ch}"] = channel_effect_national[ch][di] * geo_share_of_national_spend
            geo_revenue += row[f"revenue_{ch}"]
        row["revenue"] = round(max(geo_revenue, 0), 2)
        row["total_spend"] = round(geo_spend_total, 2)
        rows.append(row)

df = pd.DataFrame(rows)

# National daily aggregate (what analysis.py will primarily model)
agg_cols = ["revenue", "total_spend"] + [f"spend_{ch}" for ch in CHANNELS]
national = df.groupby("date")[agg_cols].sum().reset_index()
national.to_csv("data/national_daily.csv", index=False)

# Full geo-level panel (for the incrementality test)
df.to_csv("data/geo_daily_panel.csv", index=False)

# Save holdout metadata
meta = dict(
    holdout_channel=HOLDOUT_CHANNEL,
    holdout_start_day_index=HOLDOUT_START,
    holdout_end_day_index=HOLDOUT_END,
    holdout_start_date=str(dates[HOLDOUT_START].date()),
    holdout_end_date=str(dates[HOLDOUT_END - 1].date()),
    test_geos=TEST_GEOS,
    control_geos=CONTROL_GEOS,
)
with open("data/holdout_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Generated {len(national)} national daily rows and {len(df)} geo-day rows.")
print(f"Geo holdout: {HOLDOUT_CHANNEL} paused in {TEST_GEOS} from {meta['holdout_start_date']} to {meta['holdout_end_date']}")
