"""Write data/sample_flights.csv: a few made-up rows in the exact flights.csv format.

Replace with the real filtered sample (scripts/prepare_data.py) as soon as you
have it - the rest of the code reads both the same way.
"""
import csv
from pathlib import Path

COLS = ["YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER", "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT",
        "WHEELS_OFF", "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME", "DISTANCE", "WHEELS_ON", "TAXI_IN",
        "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "ARRIVAL_DELAY", "DIVERTED", "CANCELLED", "CANCELLATION_REASON",
        "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]


def hhmm(m):
    m %= 1440
    return m // 60 * 100 + m % 60


def mins(v):
    return v // 100 * 60 + v % 100


def row(y, mo, d, dow, al, fn, tail, org, dst, sdep, dep_delay, stime, dist, arr_delay,
        late=0, wx=0, nas=0, carrier=0):
    sarr = hhmm(mins(sdep) + stime)
    return [y, mo, d, dow, al, fn, tail, org, dst, sdep, hhmm(mins(sdep) + dep_delay), dep_delay, 15,
            hhmm(mins(sdep) + dep_delay + 15), stime, stime + arr_delay - dep_delay, stime - 25, dist,
            hhmm(mins(sarr) + arr_delay - 8), 8, sarr, hhmm(mins(sarr) + arr_delay), arr_delay, 0, 0, "",
            nas, 0, carrier, late, wx]


ROWS = [
    # ORD, 18 Jun 2015, thunderstorms ~15:00 -> weather contention (4 arrivals)
    row(2015, 6, 18, 4, "UA", 1544, "N37281", "DEN", "ORD", 1150, 12, 155, 888, 71, wx=52, nas=7),
    row(2015, 6, 18, 4, "AA", 2216, "N4YCAA", "DFW", "ORD", 1215, 5, 140, 802, 38, wx=26, nas=7),
    row(2015, 6, 18, 4, "OO", 4630, "N618SK", "MSP", "ORD", 1250, 0, 85, 334, 33, wx=18, nas=15),
    row(2015, 6, 18, 4, "UA", 938, "N77012", "SFO", "ORD", 942, 20, 263, 1846, 44, wx=17, late=20, nas=7),
    # ATL, 5 Mar 2015, evening -> late-aircraft contention (3 arrivals)
    row(2015, 3, 5, 4, "DL", 1972, "N6705Y", "LGA", "ATL", 1559, 48, 151, 762, 52, late=44, nas=8),
    row(2015, 3, 5, 4, "DL", 2445, "N335NB", "MCO", "ATL", 1720, 35, 88, 404, 39, late=31, nas=8),
    row(2015, 3, 5, 4, "EV", 5191, "N16954", "CHA", "ATL", 1800, 29, 51, 106, 33, late=25, nas=8),
    # DFW single flight
    row(2015, 7, 22, 3, "AA", 1301, "N3EUAA", "LAX", "DFW", 1255, 41, 190, 1235, 46, late=41, nas=5),
    # JFK, 10 Aug 2015: late inbounds pushed past midnight (tests the continuous clock)
    row(2015, 8, 10, 1, "B6", 615, "N593JB", "MCO", "JFK", 2120, 44, 150, 944, 41, late=40, nas=1),
    row(2015, 8, 10, 1, "B6", 1205, "N809JB", "FLL", "JFK", 2105, 38, 170, 1069, 36, late=35, nas=1),

    # ---- not disrupted: other traffic arriving in the same windows (takes landing slots) ----
    row(2015, 6, 18, 4, "WN", 812, "N7714B", "MDW", "ORD", 1320, 3, 70, 17, 4),
    row(2015, 6, 18, 4, "AA", 1188, "N3DLAA", "LGA", "ORD", 1205, 4, 128, 733, 6),
    row(2015, 6, 18, 4, "UA", 1712, "N76508", "IAH", "ORD", 1150, 2, 147, 925, 3),
    row(2015, 6, 18, 4, "OO", 4420, "N712SK", "DSM", "ORD", 1310, 5, 72, 299, 7),
    row(2015, 6, 18, 4, "AA", 2310, "N567AA", "BOS", "ORD", 1205, 9, 155, 867, 11),
    row(2015, 6, 18, 4, "UA", 466, "N38451", "EWR", "ORD", 1240, 0, 140, 719, -2),
    row(2015, 6, 18, 4, "MQ", 3301, "N619AE", "CLE", "ORD", 1340, 3, 75, 316, 2),
    row(2015, 6, 18, 4, "UA", 1290, "N69818", "PHX", "ORD", 1105, 6, 220, 1440, 9),
    row(2015, 6, 18, 4, "AA", 1455, "N3BGAA", "DCA", "ORD", 1255, 2, 125, 612, 4),
    row(2015, 6, 18, 4, "UA", 1021, "N47280", "DEN", "ORD", 1135, 0, 160, 888, 1),
    row(2015, 6, 18, 4, "OO", 4778, "N441SW", "GRR", "ORD", 1400, 0, 32, 137, -1),
    row(2015, 6, 18, 4, "AA", 1734, "N4YBAA", "STL", "ORD", 1305, 3, 75, 258, 5),
    row(2015, 3, 5, 4, "DL", 1101, "N906DE", "BOS", "ATL", 1530, 0, 170, 946, -5),
    row(2015, 3, 5, 4, "DL", 870, "N3743H", "DTW", "ATL", 1710, 6, 120, 594, 8),
    row(2015, 3, 5, 4, "WN", 455, "N8312C", "BWI", "ATL", 1700, 12, 125, 577, 14),
    row(2015, 3, 5, 4, "EV", 4877, "N12552", "SAV", "ATL", 1835, 3, 60, 214, 1),
    row(2015, 3, 5, 4, "DL", 2088, "N932DL", "MSP", "ATL", 1625, 4, 165, 907, 6),
    row(2015, 3, 5, 4, "DL", 1530, "N6713Y", "LAX", "ATL", 1210, 10, 270, 1946, 12),
    row(2015, 7, 22, 3, "AA", 2401, "N3LGAA", "ORD", "DFW", 1410, 4, 145, 802, 5),
    row(2015, 7, 22, 3, "AA", 1907, "N4XMAA", "MIA", "DFW", 1415, 0, 170, 1121, -3),
    row(2015, 7, 22, 3, "MQ", 3517, "N822AE", "AUS", "DFW", 1600, 2, 50, 190, 4),
    row(2015, 8, 10, 1, "DL", 2212, "N3765", "ATL", "JFK", 2200, 5, 140, 760, 6),
    row(2015, 8, 10, 1, "B6", 88, "N623JB", "BUF", "JFK", 2315, 0, 75, 301, -4),
    row(2015, 8, 10, 1, "AA", 300, "N789AN", "ORD", "JFK", 2230, 2, 125, 740, 3),
]

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "sample_flights.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(ROWS)
    print(f"wrote {len(ROWS)} rows to {out}")
