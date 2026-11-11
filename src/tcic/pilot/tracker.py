"""Simple per-member TC tracker on regional (member, lead, y, x) arrays.

At each lead, the first guess is the previous position advanced by the previous 6-h motion (or, for the first step, the
run's first-guess track). The centre is the MSLP minimum within `search_km` of the guess. A member's storm is "lost" when
the minimum jumps more than `max_jump_km` from the guess, isn't a closed low (the 200-400 km ring mean is less than
`min_depth_hpa` above it), or leaves the region. Intensity: max 10-m wind (and gust) within `rmax_km`.
"""
import numpy as np
from scipy.ndimage import uniform_filter

from tcic.pilot.grid import Region, haversine_km, ring_mean

MS_TO_KT = 1.943844


def track_members(region: Region, msl_pa, u10, v10, gust, lat0: float, lon0: float, fg_lat, fg_lon, leads_h,
                  search_km: float = 350, max_jump_km: float = 450, min_depth_hpa: float = 1.0, rmax_km: float = 300):
    """Returns a structured dict of arrays (member, lead): lat, lon, pmin, vmax, gust, rmw_km, depth, lost."""
    nm, nl = msl_pa.shape[:2]
    out = {k: np.full((nm, nl), np.nan, dtype=np.float32) for k in ("lat", "lon", "pmin", "vmax", "gust", "rmw_km", "depth")}
    out["lost"] = np.zeros((nm, nl), dtype=bool)
    la2, lo2 = np.meshgrid(region.lat, region.lon, indexing="ij")
    wind = np.hypot(u10, v10)
    for m in range(nm):
        lat, lon = lat0, lon0
        prev = None
        for k in range(nl):
            if k == 0:
                g_lat, g_lon = lat0, lon0
            elif prev is not None:
                g_lat, g_lon = 2 * lat - prev[0], 2 * lon - prev[1]
            else:
                g_lat, g_lon = lat + (fg_lat[k] - fg_lat[k - 1]), lon + (fg_lon[k] - fg_lon[k - 1])
            field = msl_pa[m, k]
            if not np.isfinite(field).any() or not region.contains(g_lat, g_lon):
                out["lost"][m, k:] = True
                break
            d = haversine_km(g_lat, g_lon, la2, lo2)
            sm = uniform_filter(np.nan_to_num(field, nan=np.nanmax(field)), size=3, mode="nearest")
            cand = np.where((d <= search_km) & np.isfinite(field), sm, np.inf)
            if not np.isfinite(cand).any():
                out["lost"][m, k:] = True
                break
            r, c = np.unravel_index(np.argmin(cand), cand.shape)
            c_lat, c_lon = float(region.lat[r]), float(region.lon[c])
            dc = haversine_km(c_lat, c_lon, la2, lo2)
            pmin = float(np.nanmin(np.where(dc <= 100, field, np.nan))) / 100.0
            depth = ring_mean(field, dc, 200, 400) / 100.0 - pmin
            if haversine_km(g_lat, g_lon, c_lat, c_lon) > max_jump_km or not depth >= min_depth_hpa:
                out["lost"][m, k:] = True
                break
            inner = dc <= rmax_km
            w = np.where(inner, wind[m, k], np.nan)
            out["lat"][m, k], out["lon"][m, k] = c_lat, c_lon
            out["pmin"][m, k], out["depth"][m, k] = pmin, depth
            out["vmax"][m, k] = np.nanmax(w) * MS_TO_KT
            if gust is not None:
                g = np.where(inner, gust[m, k], np.nan)
                out["gust"][m, k] = np.nanmax(g) * MS_TO_KT if np.isfinite(g).any() else np.nan
            out["rmw_km"][m, k] = float(dc.ravel()[np.nanargmax(w)])
            if k > 0:
                prev = (lat, lon)
            lat, lon = c_lat, c_lon
    return out
