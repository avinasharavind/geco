"""Small helpers for regular lat/lon grids that run north->south and -180->180 (both dynamical and ECMWF open data)."""
import numpy as np

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class Grid:
    """Global grid with lat0 at row 0 (north) and lon0 at column 0, spacing `res` degrees."""

    def __init__(self, res: float, lat0: float = 90.0, lon0: float = -180.0, nlat: int | None = None, nlon: int | None = None):
        self.res, self.lat0, self.lon0 = res, lat0, lon0
        self.nlat = nlat or int(round(180 / res)) + 1
        self.nlon = nlon or int(round(360 / res))

    def row(self, lat):
        return np.clip(np.rint((self.lat0 - np.asarray(lat)) / self.res).astype(int), 0, self.nlat - 1)

    def col(self, lon):
        return np.mod(np.rint((np.asarray(lon) - self.lon0) / self.res).astype(int), self.nlon)

    def lats(self, rows):
        return self.lat0 - np.asarray(rows) * self.res

    def lons(self, cols):
        return self.lon0 + np.asarray(cols) * self.res


class Region:
    """A rectangular sub-array of a Grid: rows [r0, r1), cols [c0, c1) (no dateline wrap; NA/EP storms don't need it)."""

    def __init__(self, grid: Grid, r0: int, r1: int, c0: int, c1: int):
        self.grid, self.r0, self.r1, self.c0, self.c1 = grid, r0, r1, c0, c1
        self.lat = grid.lats(np.arange(r0, r1))
        self.lon = grid.lons(np.arange(c0, c1))

    @property
    def shape(self):
        return (self.r1 - self.r0, self.c1 - self.c0)

    def contains(self, lat, lon) -> bool:
        r, c = self.grid.row(lat), self.grid.col(lon)
        return self.r0 <= r < self.r1 and self.c0 <= c < self.c1

    def distance_km(self, lat, lon):
        """(ny, nx) great-circle distance from (lat, lon) to every point of the region."""
        la, lo = np.meshgrid(self.lat, self.lon, indexing="ij")
        return haversine_km(lat, lon, la, lo)

    def window(self, lat, lon, half_deg: float):
        """Row/col slices (relative to the region) of a (2*half/res+1)^2 box centred on the nearest gridpoint, or None if it doesn't fit."""
        n = int(round(half_deg / self.grid.res))
        r = int(self.grid.row(lat)) - self.r0
        c = int(self.grid.col(lon)) - self.c0
        if r - n < 0 or c - n < 0 or r + n + 1 > self.shape[0] or c + n + 1 > self.shape[1]:
            return None
        return slice(r - n, r + n + 1), slice(c - n, c + n + 1)


def crop_global(field, grid: Grid, lat, lon, half_deg: float, fill=np.nan):
    """Crop a centred box from a global (nlat, nlon) array, wrapping in longitude and padding beyond the poles."""
    n = int(round(half_deg / grid.res))
    r, c = int(grid.row(lat)), int(grid.col(lon))
    rows = np.arange(r - n, r + n + 1)
    cols = np.mod(np.arange(c - n, c + n + 1), grid.nlon)
    out = np.full((rows.size, cols.size), fill, dtype=np.float32)
    ok = (rows >= 0) & (rows < grid.nlat)
    out[ok] = field[rows[ok]][:, cols]
    return out


def ring_mean(field, dist_km, r_in: float, r_out: float):
    m = (dist_km >= r_in) & (dist_km <= r_out) & np.isfinite(field)
    return float(field[m].mean()) if m.any() else np.nan
