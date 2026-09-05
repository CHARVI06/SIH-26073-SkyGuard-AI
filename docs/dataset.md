# IMD Maitri Dataset

## Source files and preservation

The supplied raw files remain exactly where provided in `Dataset/`:
`imd_maitri.csv` and `imd_maitri.nc`. They are not edited by this project. The
pipeline accepts the supplied location first and also supports a future
`data/raw/` copy without requiring an absolute path.

## Verified schema

The CSV has **no header row**. Its six columns were matched exactly against the
overlapping NetCDF observations (2015-01-01 through 2016-12-19):

| CSV position | Source name | Verified use |
| --- | --- | --- |
| 0 | `timestamp` | observation timestamp |
| 1 | `tempr` | temperature (`temperature_c`) |
| 2 | `rh` | atmospheric pressure (`pressure_hpa`) |
| 3 | `ws` | wind variable; not used by Member 1 baseline |
| 4 | `wd` | wind variable; not used by Member 1 baseline |
| 5 | `ap` | relative humidity (`humidity_pct`) |

The raw data does not include a station identifier or sensor identifier. The
provided NetCDF variable attributes contain no explicit units for the weather
variables. The project uses the problem statement's required labels (C, hPa,
and percent) only after the cross-file value mapping above; it does not invent
additional fields or units.

## Observed data quality

- CSV coverage: 1985-01-01 00:00 through 2016-12-19 12:00, 155,170 rows.
- Observations are time-ordered and timestamps are unique; timestamps are
  generally hourly but contain outages/irregular intervals.
- `-999` is the missing-reading marker, not a genuine weather value.
- In the three baseline inputs, sentinel counts are 8,297 each for `tempr` and
  `rh`, and 138,656 for `ap`; early humidity coverage is therefore sparse.
- NetCDF coverage: 16,514 values from 2015-01-01 to 2016-12-19, encoded as
  minutes since 2000-01-15 00:00:00. It has no NaN or `-999` values.

The preprocessing pipeline converts sentinel values to missing values, permits
at most three short internal time interpolations, and excludes remaining
incomplete rows from model-ready output. It never reindexes across outages.
