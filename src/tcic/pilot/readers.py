"""Data readers for the pilot.

- `NativeReader`: per-member surface fields from dynamical's tiled (native) stores. Reads only the tiles touched by a
  storm's first-guess swath, once each, with all members and leads in each tile, and assembles a regional array.
- `GefsControl`: control-member pressure-level fields from the GEFS 0.5° virtual store (full-globe chunks).
- `EcmwfControl`: control-member pressure-level fields from ECMWF open data on the GCS mirror (GRIB byte ranges).
All network reads are debited against the global download throttle.
"""
import functools
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from tcic.io.throttle import NetCounter, bucket, throttled_get
from tcic.pilot.grid import Grid, Region


def _retry(fn, tries: int = 4, base_s: float = 5.0):
    for k in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - network hiccups; re-raise after the last try
            if k == tries - 1:
                raise
            time.sleep(base_s * 2**k)


@functools.lru_cache(maxsize=8)
def _open(dataset_id: str, group: str | None = None):
    import dynamical_catalog as dc

    return dc.open(dataset_id, chunks=None, **({"group": group} if group else {}))


class NativeReader:
    def __init__(self, dataset_id: str, var_map: dict[str, str], threads: int = 16):
        self.ds = _open(dataset_id)
        self.dataset_id = dataset_id
        self._group, self._arrays = None, {}
        self.var_map = var_map
        self.threads = threads
        da = self.ds[next(iter(var_map.values()))]
        self.grid = Grid(0.25, float(self.ds.latitude[0]), float(self.ds.longitude[0]), self.ds.sizes["latitude"], self.ds.sizes["longitude"])
        ch = dict(zip(da.dims, da.encoding["chunks"]))
        self.tile = (ch["latitude"], ch["longitude"])

    def tiles_for(self, lats, lons, margins_deg) -> set[tuple[int, int]]:
        tiles = set()
        for la, lo, m in zip(lats, lons, margins_deg):
            r0, r1 = self.grid.row(la + m), self.grid.row(la - m)
            c0, c1 = self.grid.col(lo - m), self.grid.col(lo + m)
            if c1 < c0:  # don't wrap the dateline
                c1 = self.grid.nlon - 1
            for tr in range(r0 // self.tile[0], r1 // self.tile[0] + 1):
                for tc in range(c0 // self.tile[1], c1 // self.tile[1] + 1):
                    tiles.add((tr, tc))
        return tiles

    def region_for(self, tiles) -> Region:
        rows = [t[0] for t in tiles]
        cols = [t[1] for t in tiles]
        return Region(self.grid, min(rows) * self.tile[0], min((max(rows) + 1) * self.tile[0], self.grid.nlat),
                      min(cols) * self.tile[1], min((max(cols) + 1) * self.tile[1], self.grid.nlon))

    def read(self, var: str, init: pd.Timestamp, leads_h, tiles, region: Region) -> np.ndarray:
        """(member, lead, y, x) float32 over `region`; NaN outside the requested tiles.

        Reads the zarr array directly with plain slices, one inner chunk (tile) per request. Going through xarray's lazy
        indexing cost ~2.5 GB of transient memory per variable; this costs ~the size of the tiles themselves.
        """
        arr = self._zarr(var)
        dims = self.ds[self.var_map[var]].dims
        it = int(np.searchsorted(self.ds.init_time.values, np.datetime64(init)))
        if self.ds.init_time.values[it] != np.datetime64(init):
            raise KeyError(f"{init} not in {self.var_map[var]}")
        lead_idx = np.searchsorted(self.ds.lead_time.values, pd.to_timedelta(list(leads_h), unit="h").values)
        lead_slice = slice(int(lead_idx.min()), int(lead_idx.max()) + 1)  # only the lead chunk(s) we need (GEFS stores 0-840 h)
        lead_idx = lead_idx - lead_slice.start
        order = [d for d in dims if d != "init_time"]
        perm = [order.index(d) for d in ("ensemble_member", "lead_time", "latitude", "longitude")]
        out = np.full((self.ds.sizes["ensemble_member"], len(lead_idx)) + region.shape, np.nan, dtype=np.float32)

        def one(t):
            rs = slice(t[0] * self.tile[0], min((t[0] + 1) * self.tile[0], self.grid.nlat))
            cs = slice(t[1] * self.tile[1], min((t[1] + 1) * self.tile[1], self.grid.nlon))
            sel = {"init_time": it, "lead_time": lead_slice, "ensemble_member": slice(None), "latitude": rs, "longitude": cs}
            block = _retry(lambda: arr[tuple(sel[d] for d in dims)])
            block = np.transpose(block, perm)[:, lead_idx]
            out[:, :, rs.start - region.r0: rs.stop - region.r0, cs.start - region.c0: cs.stop - region.c0] = block

        with NetCounter() as net, ThreadPoolExecutor(self.threads) as ex:
            list(ex.map(one, sorted(tiles)))
        bucket().consume(net.nbytes)
        return out

    def _zarr(self, var: str):
        if var not in self._arrays:
            import dynamical_catalog as dc
            import zarr

            if self._group is None:
                self._group = zarr.open_group(dc.get_store(self.dataset_id), mode="r")
            self._arrays[var] = self._group[self.var_map[var]]
        return self._arrays[var]


GEFS_LEVELS = {"u": ("wind_u", [200, 500, 850]), "v": ("wind_v", [200, 500, 850]), "t": ("temperature", [200, 500]),
               "r": ("relative_humidity", [500, 700]), "q": ("specific_humidity", [850]), "absv": ("absolute_vorticity", [850])}
GEFS_SFC = {"skt": "temperature_surface", "tcw": "precipitable_water_atmosphere"}


class GefsControl:
    """Control member (0) of the GEFS 0.5° virtual store; returns {name: (lead, nlat, nlon)} with names like 'u200'."""

    grid = Grid(0.5)
    member = "c00"

    def read(self, init: pd.Timestamp, leads_h) -> dict[str, np.ndarray]:
        pl = _open("noaa-gefs-forecast-16-day-0-5-degree-virtual", "pressure_level")
        sfc = _open("noaa-gefs-forecast-16-day-0-5-degree-virtual")
        lt = pd.to_timedelta(list(leads_h), unit="h")
        out = {}
        with NetCounter() as net:
            for short, (name, levels) in GEFS_LEVELS.items():
                da = pl[name].sel(init_time=init, ensemble_member=0, lead_time=lt, pressure_level=levels)
                vals = _retry(lambda: da.transpose("pressure_level", "lead_time", "latitude", "longitude").values)
                for i, lev in enumerate(levels):
                    out[f"{short}{lev}"] = vals[i].astype(np.float32)
            for short, name in GEFS_SFC.items():
                da = sfc[name].sel(init_time=init, ensemble_member=0, lead_time=lt)
                out[short] = _retry(lambda: da.transpose("lead_time", "latitude", "longitude").values).astype(np.float32)
        bucket().consume(net.nbytes)
        # absolute -> relative vorticity (subtract f) so it matches ECMWF 'vo'
        f = 2 * 7.292e-5 * np.sin(np.radians(self.grid.lats(np.arange(self.grid.nlat))))[None, :, None]
        out["vo850"] = out.pop("absv850") - f
        return out


ECMWF_URL = "https://storage.googleapis.com/ecmwf-open-data/{d:%Y%m%d}/{d:%H}z/ifs/0p25/enfo/{d:%Y%m%d%H}0000-{h}h-enfo-ef"
ECMWF_FIELDS = [("u", "200"), ("u", "500"), ("u", "850"), ("v", "200"), ("v", "500"), ("v", "850"), ("t", "200"), ("t", "500"),
                ("r", "500"), ("r", "700"), ("q", "850"), ("vo", "850"), ("d", "200"), ("skt", None), ("tcwv", None)]


class EcmwfControl:
    """Control forecast ('cf') of IFS-ENS from ECMWF open data; returns {name: (lead, 721, 1440)}.

    From about mid-May 2026 the open-data ENS files no longer include the control forecast (only members 1-50), so we
    fall back to perturbed member 1 and record that in `self.member` ("cf" or "pf1").
    """

    grid = Grid(0.25)

    def __init__(self, threads: int = 16):
        self.threads = threads
        self.member = None

    def read(self, init: pd.Timestamp, leads_h) -> dict[str, np.ndarray]:
        import gribberish

        leads_h = list(leads_h)
        jobs = []
        for i, h in enumerate(leads_h):
            base = ECMWF_URL.format(d=init, h=h)
            idx = [json.loads(l) for l in _retry(lambda: throttled_get(base + ".index")).decode().splitlines() if l.strip()]
            has_cf = any(x.get("type") == "cf" for x in idx)
            member = "cf" if has_cf else "pf1"
            if self.member not in (None, member):
                raise RuntimeError(f"mixed ECMWF members within one run {init}: {self.member} and {member}")
            self.member = member
            for x in idx:
                if has_cf and x.get("type") != "cf":
                    continue
                if not has_cf and not (x.get("type") == "pf" and str(x.get("number")) == "1"):
                    continue
                key = (x["param"], x.get("levelist"))
                if key in ECMWF_FIELDS:
                    name = x["param"] + (x.get("levelist") or "")
                    jobs.append((i, name, base + ".grib2", int(x["_offset"]), int(x["_length"])))
        names = [p + (l or "") for p, l in ECMWF_FIELDS]
        out = {n: np.full((len(leads_h), self.grid.nlat, self.grid.nlon), np.nan, dtype=np.float32) for n in names}

        def one(j):
            i, name, url, off, length = j
            raw = _retry(lambda: throttled_get(url, (off, off + length - 1)))
            out[name][i] = np.asarray(gribberish.parse_grib_message(raw, 0).data(), dtype=np.float32).reshape(self.grid.nlat, self.grid.nlon)

        with ThreadPoolExecutor(self.threads) as ex:
            list(ex.map(one, jobs))
        missing = [n for n in names if np.isnan(out[n]).all()]
        if missing:
            raise RuntimeError(f"ECMWF {self.member} fields missing for {init}: {missing}")
        return out
