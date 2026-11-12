"""Signed distance to land (km) on a 0.5° grid, from the GEFS land-sea mask (one field, ~0.25 MB).

Positive over water (distance to the nearest land cell), negative over land (distance to the nearest water cell).
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from tcic.config import INTERIM_DIR

EARTH_R_KM = 6371.0
PATH = INTERIM_DIR / "dist2land_0p5.nc"


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def build() -> xr.DataArray:
    import dynamical_catalog as dc

    ds = dc.open("noaa-gefs-forecast-16-day-0-5-degree-virtual", chunks=None)
    lsm = ds["land_sea_mask_surface"].sel(init_time=pd.Timestamp("2025-06-01T00:00"), ensemble_member=0).isel(lead_time=0).load()
    land = lsm.values > 0.5
    lat2, lon2 = np.meshgrid(lsm.latitude.values, lsm.longitude.values, indexing="ij")
    pts = _xyz(lat2.ravel(), lon2.ravel())
    d_land, _ = cKDTree(pts[land.ravel()]).query(pts[~land.ravel()])
    d_water, _ = cKDTree(pts[~land.ravel()]).query(pts[land.ravel()])
    dist = np.empty(land.size)
    dist[~land.ravel()] = 2 * EARTH_R_KM * np.arcsin(d_land / 2)
    dist[land.ravel()] = -2 * EARTH_R_KM * np.arcsin(d_water / 2)
    da = xr.DataArray(dist.reshape(land.shape).astype("float32"), coords={"latitude": lsm.latitude.values, "longitude": lsm.longitude.values},
                      dims=("latitude", "longitude"), name="dist2land_km")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(PATH)
    return da


def dist2land(lat, lon) -> np.ndarray:
    """Nearest-gridpoint signed distance to land (km); NaN where lat/lon is NaN."""
    da = xr.open_dataarray(PATH) if PATH.exists() else build()
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    out = np.full(lat.shape, np.nan, dtype="float32")
    lons = da.longitude.values
    lon_c = np.where(lons.min() >= 0, np.mod(lon, 360), lon)
    ilat = np.clip(np.rint((da.latitude.values[0] - lat) / 0.5), 0, da.sizes["latitude"] - 1)
    ilon = np.mod(np.rint((lon_c - lons[0]) / 0.5), da.sizes["longitude"])
    out[ok] = da.values[ilat[ok].astype(int), ilon[ok].astype(int)]
    return out


if __name__ == "__main__":
    build()
    print("Jamaica (18.1N 77.3W):", dist2land([18.1], [-77.3]), " open Atlantic (25N 50W):", dist2land([25.0], [-50.0]),
          " Melissa landfall approach (17.0N 77.6W):", dist2land([17.0], [-77.6]))
