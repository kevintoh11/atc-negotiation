"""Cut the full DOT flights.csv (~590 MB) down to what the scenarios need.

    python scripts/prepare_data.py path/to/flights.csv               # every disrupted airport-hour
    python scripts/prepare_data.py path/to/flights.csv --sample 300  # 300 random airport-hours

For each airport-hour that has at least one disrupted arrival (WEATHER_DELAY > 15
or LATE_AIRCRAFT_DELAY > 20), it keeps EVERY arrival at that airport in that hour
and the hours either side, disrupted or not. The non-disrupted ones are the other
traffic that takes landing slots. Hours are by ETA (scheduled arrival + departure
delay), not by schedule. October-2015 rows with numeric airport IDs are dropped.
"""
import argparse
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
COLS = ["YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER", "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "SCHEDULED_ARRIVAL", "ARRIVAL_DELAY",
        "DISTANCE", "AIR_TIME", "TAXI_IN", "TAXI_OUT", "DIVERTED", "CANCELLED", "WEATHER_DELAY",
        "LATE_AIRCRAFT_DELAY"]
DAY_KEY = ["YEAR", "MONTH", "DAY", "DESTINATION_AIRPORT"]


def eta_hour(df: pd.DataFrame) -> pd.Series:
    sa, sd = df.SCHEDULED_ARRIVAL.astype(int), df.SCHEDULED_DEPARTURE.astype(int)
    sched = (sa // 100) * 60 + sa % 100 + 1440 * (sa < sd)          # overnight arrivals: next day
    eta = sched + (df.DEPARTURE_DELAY.fillna(0).clip(lower=0) * 0.9).astype(int)
    return eta // 60


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("flights_csv")
    ap.add_argument("--sample", type=int, default=0, help="number of disrupted airport-hours to keep (0 = all)")
    ap.add_argument("--out", default=str(DATA / "disruptions.csv"))
    a = ap.parse_args()

    df = pd.read_csv(a.flights_csv, usecols=COLS, low_memory=False)
    codes = set(pd.read_csv(DATA / "airports.csv").IATA_CODE)
    df = df[df.ORIGIN_AIRPORT.astype(str).isin(codes) & df.DESTINATION_AIRPORT.astype(str).isin(codes)]
    df = df[(df.CANCELLED == 0) & df.SCHEDULED_ARRIVAL.notna()].copy()
    df["ETA_HOUR"] = eta_hour(df)

    hit = df[(df.WEATHER_DELAY > 15) | (df.LATE_AIRCRAFT_DELAY > 20)]
    keys = hit[DAY_KEY + ["ETA_HOUR"]].drop_duplicates()
    if a.sample:
        keys = keys.sample(min(len(keys), a.sample), random_state=42)
    # keep the hour itself plus the hours either side, so traffic around the window is present
    wide = pd.concat([keys.assign(ETA_HOUR=keys.ETA_HOUR + d) for d in (-1, 0, 1)]).drop_duplicates()
    out = df.merge(wide, on=DAY_KEY + ["ETA_HOUR"]).drop(columns="ETA_HOUR")
    out.to_csv(a.out, index=False)
    n_dis = ((out.WEATHER_DELAY > 15) | (out.LATE_AIRCRAFT_DELAY > 20)).sum()
    print(f"{len(keys):,} airport-hours -> {len(out):,} arrivals ({n_dis:,} disrupted, "
          f"{len(out) - n_dis:,} other traffic) written to {a.out}")
