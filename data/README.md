# Data

| File | PVOD station | Days | SHA-256 |
| --- | --- | --- | --- |
| `S-1.csv` | `station00` | 302 | `102bf1033d86cf8de8866a2fcdf0509aed19637ac30b3eff81710bfe6233274b` |
| `S-2.csv` | `station01` | 348 | `a6d24acf27933d8ec82433764578349a26ad89379d6286bb6d08f7d6caaa72b0` |
| `S-3.csv` | `station07` | 348 | `bb0e56312458ccbf8300e0ee170b07fe491918110cac41844c695dbca4241e55` |
| `S-4.csv` | `station08` | 348 | `cfb933934780b51fae1cccf6abbabb9af7bab7f28a657433c318d1c3e83c6e20` |

## Provenance

The files come from PVOD (Yao et al., 2021), a public dataset of onsite measurements and WRF-generated NWP for
Chinese PV stations at 15-min resolution. They are byte-for-byte identical to the files in the preprocessed
`full_dataset.zip` archive released with Cross-Unet (Zhang et al., 2026). No values were changed for this
repository.

- T. Yao et al., "A photovoltaic power output dataset: Multi-source photovoltaic power output dataset with
  Python toolkit," *Solar Energy*, vol. 230, pp. 122–130, 2021. doi:10.1016/j.solener.2021.09.050
- Y. Zhang et al., "Rethinking the use of deep learning methods for photovoltaic power forecasting,"
  *Nature Communications*, vol. 17, 7613, 2026.

Use of the data is subject to the terms of the original PVOD release.

## Columns used

| Column | Role | Unit |
| --- | --- | --- |
| `Time` | timestamp (UTC+8, 8 h ahead of `Timestamp`) | – |
| `lmd_totalirrad`, `lmd_diffuseirrad` | measured global / diffuse irradiance (history) | W/m² |
| `lmd_temperature`, `lmd_pressure` | measured temperature / pressure (history) | °C / hPa |
| `lmd_winddirection`, `lmd_windspeed` | measured wind direction / speed (history) | ° / m/s |
| `nwp_globalirrad`, `nwp_directirrad` | NWP global / direct irradiance (history and forecast window) | W/m² |
| `nwp_temperature`, `nwp_humidity` | NWP temperature / relative humidity | °C / % |
| `nwp_windspeed`, `nwp_winddirection` | NWP wind speed / direction | m/s / ° |
| `power` | PV power, the forecast target | MW |

Other columns (`Timestamp`, `date_time`, `nwp_pressure`, `Location`, `Longitude`, `Latitude`, `SWR`) are
not used. The NWP series is aligned to valid time; the paper discusses what this implies for horizons longer
than one day.

To use another station, provide a CSV with the columns above and pass `--data-dir`; the file name
(without `.csv`) is the station name.
