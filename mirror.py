"""Keep a small public copy of the newest TWDB BayCAST surface currents that a web page can read.

Run:  python mirror.py site          (needs: numpy, requests, h5py)
Out:  site/index.json                the list of days, rewritten after every day (so a stopped run loses nothing)
      site/days/YYYYMMDD.bin         one day: u then v, int8, value x 0.02 = m/s, (time step, node) order
      site/mesh_<key>.bin            lon, lat (float32), triangles (uint32), neighbours (int32: -1 land, -2 open water)
Log:  printed one line per day, with the time.

Source: https://midgewater.twdb.texas.gov/bays_estuaries/baycast/ (file list: item.json). Each daily file
hydro_2D_surf_YYYYMMDD.nc holds u, v in m/s on a triangular mesh every 2 hours, time in UTC (file attributes).
A day already in site/ with the same TWDB etag is skipped, so running again only fetches what changed.

Machine use: 4 download threads (the work is network-bound); packing is one process, under 1 GB, no GPU.
"""
import base64, concurrent.futures as cf, datetime as dt, json, os, sys, tempfile, time, zlib
import numpy as np
import requests
import h5py

BASE = "https://midgewater.twdb.texas.gov/bays_estuaries/baycast/"
UA = {"User-Agent": "Mozilla/5.0"}
DAYS_KEPT = int(os.environ.get("DAYS_KEPT", 60))           # days before today that stay in the copy (size choice; newer and forecast days are always kept)
LON_MAX = -93.3          # keep the mesh west of this longitude (Sabine Pass is near -93.84): the Texas side
VEL_STEP = 0.02          # m/s per stored unit (int8), so stored speeds are exact to 0.01 m/s; larger than 2.54 m/s is clipped and counted


def log(msg):
    print(f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}Z | {msg}", flush=True)


def get(url, tries=4, **kw):
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=600, **kw)
            r.raise_for_status()
            return r
        except Exception as e:
            if k == tries - 1:
                raise
            log(f"retry {k + 1} for {url.rsplit('/', 1)[-1]}: {type(e).__name__}")
            time.sleep(10 * (k + 1))


def crop(lon, nv):
    """nodes and triangles on the Texas side; triangles counter-clockwise; nv is (triangles, 3), counted from 0"""
    nk = lon <= LON_MAX
    tk = nk[nv[:, 0]] & nk[nv[:, 1]] & nk[nv[:, 2]]
    tri = nv[tk]
    used = np.zeros(len(lon), bool); used[tri.ravel()] = True
    return used, tk, tri


def build_mesh(lon, lat, nv, bnd):
    used, tk, tri_old = crop(lon, nv)
    new = -np.ones(len(lon), np.int64); new[used] = np.arange(used.sum())
    tri = new[tri_old]
    lon2, lat2 = lon[used].astype("<f4"), lat[used].astype("<f4")
    ax, ay = lon2[tri[:, 0]], lat2[tri[:, 0]]
    area2 = (lon2[tri[:, 1]] - ax) * (lat2[tri[:, 2]] - ay) - (lon2[tri[:, 2]] - ax) * (lat2[tri[:, 1]] - ay)
    flip = area2 < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    nt = len(tri)
    # neighbours: side k of a triangle is the edge from corner k to corner k+1
    ea = tri.ravel(); eb = tri[:, [1, 2, 0]].ravel()
    key = np.minimum(ea, eb).astype(np.int64) * len(lon2) + np.maximum(ea, eb)
    o = np.argsort(key, kind="stable"); ks = key[o]
    same = ks[1:] == ks[:-1]
    assert not (same[1:] & same[:-1]).any(), "an edge is shared by 3 triangles"
    nb = -np.ones(nt * 3, np.int32)            # -1 = land edge
    i = np.nonzero(same)[0]
    nb[o[i]] = o[i + 1] // 3; nb[o[i + 1]] = o[i] // 3
    # open edges: the model's own open boundary (bnd column 4 = 1) and the edge made by the crop
    fa = nv.ravel(); fb = nv[:, [1, 2, 0]].ravel()
    fkey = np.minimum(fa, fb).astype(np.int64) * len(lon) + np.maximum(fa, fb)
    fu, fc = np.unique(fkey, return_counts=True)
    full_boundary = set(fu[fc == 1].tolist())
    open_seg = bnd[bnd[:, 3] == 1]
    open_keys = set(((np.minimum(open_seg[:, 0], open_seg[:, 1]).astype(np.int64) - 1) * len(lon)
                     + (np.maximum(open_seg[:, 0], open_seg[:, 1]) - 1)).tolist())
    old = np.nonzero(used)[0]
    be = np.nonzero(nb == -1)[0]
    oa, ob = old[ea[be]], old[eb[be]]
    okey = (np.minimum(oa, ob).astype(np.int64) * len(lon) + np.maximum(oa, ob)).tolist()
    nb[be[np.array([(k not in full_boundary) or (k in open_keys) for k in okey])]] = -2
    blob = lon2.tobytes() + lat2.tobytes() + tri.astype("<u4").tobytes() + nb.astype("<i4").tobytes()
    return blob, int(used.sum()), int(nt), int((nb == -1).sum()), int((nb == -2).sum())


def main(site):
    os.makedirs(os.path.join(site, "days"), exist_ok=True)
    ipath = os.path.join(site, "index.json")
    old = json.load(open(ipath, encoding="utf-8")) if os.path.exists(ipath) else {}
    if old.get("vel_step") != VEL_STEP or old.get("lon_max") != LON_MAX:
        old = {}
    days, meshes = dict(old.get("days", {})), dict(old.get("meshes", {}))

    listing = get(BASE + "item.json").json()
    files = {f["name"][14:22]: f for f in listing["files"]
             if f["name"].startswith("hydro_2D_surf_2") and f["name"].endswith(".nc") and len(f["name"]) == 25}
    first = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=DAYS_KEPT)).strftime("%Y%m%d")
    want = sorted(d for d in files if d >= first)
    todo = [d for d in want if not (days.get(d, {}).get("etag") == files[d]["etag"]
                                    and (days[d]["nt"] == 0 or os.path.exists(os.path.join(site, "days", d + ".bin"))))]
    log(f"TWDB lists {len(files)} daily surface files; window {want[0]} to {want[-1]} = {len(want)} days; to fetch now: {len(todo)}")

    def save_index():
        out = dict(built_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), source=BASE, vel_step=VEL_STEP, step_h=2,
                   lon_max=LON_MAX, note="TWDB says BayCAST is for research and education, not for real-time decisions.",
                   days={d: days[d] for d in sorted(days)}, meshes=meshes)
        with open(ipath + ".tmp", "w", encoding="utf-8") as f:
            json.dump(out, f)
        os.replace(ipath + ".tmp", ipath)

    tmp = tempfile.mkdtemp()

    def fetch(d):
        p = os.path.join(tmp, d + ".nc")
        with open(p, "wb") as f:
            f.write(get(BASE + files[d]["name"]).content)
        return d, p

    used_by_key = {}
    with cf.ThreadPoolExecutor(4) as pool:
        for n, (d, p) in enumerate(pool.map(fetch, todo), 1):
            meta = dict(etag=files[d]["etag"], size=files[d]["size"])
            with h5py.File(p, "r") as f:
                nt = int(f["time"].shape[0])
                if nt == 0:
                    days[d] = dict(nt=0, **meta)
                    log(f"{n}/{len(todo)} {d}: 0 time steps (TWDB has not filled this day yet)")
                else:
                    un = f["time"].attrs["units"]; un = un.decode() if isinstance(un, bytes) else un
                    assert un.startswith("hours since "), un
                    t0 = dt.datetime.strptime(un[12:], "%Y-%m-%dT%H:%M:%SZ")
                    hrs = [int(h) for h in f["time"][:]]
                    assert all(b - a == 2 for a, b in zip(hrs, hrs[1:])), f"{d}: steps are not 2 hours apart: {hrs}"
                    for name in ("u", "v"):
                        uu = f[name].attrs["units"]; assert (uu.decode() if isinstance(uu, bytes) else uu) == "m/s"
                    lon = f["lon"][:]
                    key = f"{len(lon)}_{zlib.crc32(lon.astype('<f4').tobytes()):08x}"
                    nv = f["nv"][:].T.astype(np.int64) - 1
                    if key not in meshes or not os.path.exists(os.path.join(site, meshes[key]["file"])):
                        blob, nn, ntri, nland, nopen = build_mesh(lon, f["lat"][:], nv, f["bnd"][:])
                        with open(os.path.join(site, f"mesh_{key}.bin"), "wb") as g:
                            g.write(blob)
                        meshes[key] = dict(file=f"mesh_{key}.bin", nn=nn, ntri=ntri, land_edges=nland, open_edges=nopen)
                        log(f"mesh {key}: {nn} of {len(lon)} nodes, {ntri} triangles, {nland} land edges, {nopen} open edges")
                    if key not in used_by_key:
                        used_by_key[key] = crop(lon, nv)[0]
                    used = used_by_key[key]
                    assert int(used.sum()) == meshes[key]["nn"]
                    u, v = f["u"][:][:, used], f["v"][:][:, used]
                    nan = int(np.isnan(u).sum() + np.isnan(v).sum())      # missing values are stored as 0 and counted
                    u, v = np.nan_to_num(u), np.nan_to_num(v)
                    lim = 127 * VEL_STEP
                    nclip = int((np.abs(u) > lim).sum() + (np.abs(v) > lim).sum())
                    q = lambda a: np.clip(np.rint(a / VEL_STEP), -127, 127).astype(np.int8)
                    with open(os.path.join(site, "days", d + ".bin"), "wb") as g:
                        g.write(q(u).tobytes() + q(v).tobytes())
                    c = f.attrs.get("comment", b""); c = c.decode() if isinstance(c, bytes) else str(c)
                    days[d] = dict(nt=nt, t0=(t0 + dt.timedelta(hours=hrs[0])).strftime("%Y-%m-%dT%H:%M:%SZ"), nn=meshes[key]["nn"], mesh=key,
                                   clipped=nclip, missing=nan, comment=c, **meta)
                    log(f"{n}/{len(todo)} {d}: {nt} steps from {days[d]['t0']}, clipped {nclip}, missing {nan}")
            os.remove(p)
            save_index()

    for d in [d for d in days if d not in want]:
        days.pop(d)
        p = os.path.join(site, "days", d + ".bin")
        if os.path.exists(p):
            os.remove(p)
    for k in [k for k in meshes if not any(v.get("mesh") == k for v in days.values())]:
        p = os.path.join(site, meshes.pop(k)["file"])
        if os.path.exists(p):
            os.remove(p)
    save_index()
    size = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(site) for x in fs if ".git" not in r)
    log(f"done: {len(days)} days in the copy, {sum(1 for v in days.values() if v['nt'])} with data, {size / 1e6:.1f} MB (10^6 bytes)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "site")
