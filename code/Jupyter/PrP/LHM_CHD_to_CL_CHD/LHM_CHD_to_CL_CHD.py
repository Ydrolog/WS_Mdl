"""
Climatology of constant-head (CHD) boundary conditions (iMOD IDF files).
 
Reads all files named  LHM_HD_<YYYYMMDD>_L<layer>_<area>.idf  from one folder,
computes for every (day, month, layer) the mean head over the selected years
(default 1991-2025) and writes  CL_HD_<DDMM>_L<layer>_<out-area>.idf
(e.g. CL_HD_1401_L1_NBr120.idf)  =>  24 time steps x number of layers files.
Input only exists for odd layers; even layers are written as a copy of the odd layer above
(L2 <- L1, L4 <- L3, ...), unless --no-fill-even is given.
 
Usage:
    python make_chd_climatology.py INPUT_DIR OUTPUT_DIR
    python make_chd_climatology.py INPUT_DIR OUTPUT_DIR --start-year 1991 --end-year 2025 \
           --days 14 28 --in-area NBr111 --out-area NBr120
 
Requires numpy only (no imod package needed).
"""
import argparse
import re
import struct
import sys
import warnings
from collections import defaultdict
from pathlib import Path
 
import numpy as np
 
# Classic iMOD IDF layout (little-endian, float32):
#   int32 lrecl (1271), int32 ncol, int32 nrow,
#   float32 xmin, xmax, ymin, ymax, dmin, dmax, nodata, ...rest of header...
# The data always occupy the last ncol*nrow*4 bytes of the file.
OFF_DMIN, OFF_DMAX, OFF_NODATA = 28, 32, 36
 
NAME_RE = re.compile(r"^LHM_HD_(\d{4})(\d{2})(\d{2})_L(\d+)_(.+)\.idf$", re.IGNORECASE)
 
 
def read_idf(path):
    raw = Path(path).read_bytes()
    lrecl, ncol, nrow = struct.unpack("<3i", raw[:12])
    if lrecl != 1271:
        raise ValueError(f"{path}: unexpected IDF magic number {lrecl} (expected 1271)")
    nbytes = ncol * nrow * 4
    header = raw[: len(raw) - nbytes]
    data = np.frombuffer(raw[len(raw) - nbytes:], dtype="<f4").reshape(nrow, ncol)
    nodata = struct.unpack("<f", header[OFF_NODATA:OFF_NODATA + 4])[0]
    return header, data, nodata
 
 
def write_idf(path, header, data, nodata):
    valid = data != nodata
    h = bytearray(header)
    if valid.any():
        struct.pack_into("<f", h, OFF_DMIN, float(data[valid].min()))
        struct.pack_into("<f", h, OFF_DMAX, float(data[valid].max()))
    with open(path, "wb") as f:
        f.write(bytes(h))
        f.write(np.ascontiguousarray(data, dtype="<f4").tobytes())
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--start-year", type=int, default=1991)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--days", type=int, nargs="+", default=[14, 28])
    ap.add_argument("--in-area", default=None, help="area code in input file names (default: any)")
    ap.add_argument("--out-area", default="NBr120", help="area code in output file names")
    ap.add_argument("--prefix", default="CL_HD")
    ap.add_argument("--n-layers", type=int, default=37,
                    help="total number of model layers (default: highest odd input layer + 1)")
    ap.add_argument("--no-fill-even", action="store_true",
                    help="do NOT create even layers from the odd layer above (L2 <- L1, L4 <- L3, ...)")
    args = ap.parse_args()
 
    args.output_dir.mkdir(parents=True, exist_ok=True)
 
    # 1. Index files: (day, month, layer) -> {year: path}
    groups = defaultdict(dict)
    for p in sorted(args.input_dir.glob("*.idf")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        yr, mo, dy, lay, area = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5]
        if args.in_area and area != args.in_area:
            continue
        if dy not in args.days or not (args.start_year <= yr <= args.end_year):
            continue
        groups[(dy, mo, lay)][yr] = p
 
    if not groups:
        sys.exit("No matching files found (check folder, years and --in-area).")
 
    # Even layers are derived from the odd layer above, so any even-layer input files are ignored
    # (stray files would otherwise replace the average with a single year).
    if not args.no_fill_even:
        stray = sorted({(k, y) for k, v in groups.items() if k[2] % 2 == 0 for y in v})
        if stray:
            print(f"NOTE: ignoring {len(stray)} even-layer input file(s), e.g. "
                  f"day {stray[0][0][0]:02d} month {stray[0][0][1]:02d} L{stray[0][0][2]} year {stray[0][1]}")
        groups = {k: v for k, v in groups.items() if k[2] % 2 == 1}
 
    layers = sorted({k[2] for k in groups})
    n_layers = args.n_layers or (max(layers) + (1 if max(layers) % 2 else 0))
    years_expected = set(range(args.start_year, args.end_year + 1))
    print(f"{len(groups)} groups (day/month/layer), layers: {layers}, "
          f"years {args.start_year}-{args.end_year}")
 
    # 2. Compute the mean per group
    ref_header = ref_shape = ref_nodata = None
    problems = []
    n_written = 0
    for (dy, mo, lay), files in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        missing = sorted(years_expected - set(files))
        if missing:
            problems.append(f"{dy:02d}-{mo:02d} L{lay}: missing years {missing}")
 
        stack = []
        for yr, p in sorted(files.items()):
            header, data, nodata = read_idf(p)
            if ref_header is None:
                ref_header, ref_shape, ref_nodata = header, data.shape, nodata
            # same grid? (bytes 12..28 of the header hold the extent)
            if data.shape != ref_shape or header[12:28] != ref_header[12:28]:
                sys.exit(f"Grid differs from reference in {p.name}")
            a = data.astype("float64")
            a[(data == nodata) | (data >= 1e19)] = np.nan
            stack.append(a)
 
        stack = np.stack(stack)  # (years, nrow, ncol)
        # nodata pattern should be identical across years; warn otherwise
        n_valid = np.sum(~np.isnan(stack), axis=0)
        partial = (n_valid > 0) & (n_valid < stack.shape[0])
        if partial.any():
            problems.append(f"{dy:02d}-{mo:02d} L{lay}: {int(partial.sum())} cells not valid in all years")
 
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(stack, axis=0)
        mean = np.where(np.isnan(mean), ref_nodata, mean).astype("float32")
 
        out = args.output_dir / f"{args.prefix}_{dy:02d}{mo:02d}_L{lay}_{args.out_area}.idf"
        write_idf(out, ref_header, mean, ref_nodata)
        print(f"  {out.name}  (n_years={len(files)})")
        n_written += 1
 
        # Even layers have no data of their own: copy the odd layer above (L2 <- L1, L4 <- L3, ...)
        even = lay + 1
        if not args.no_fill_even and lay % 2 == 1 and even <= n_layers:
            out_e = args.output_dir / f"{args.prefix}_{dy:02d}{mo:02d}_L{even}_{args.out_area}.idf"
            write_idf(out_e, ref_header, mean, ref_nodata)
            print(f"  {out_e.name}  (copy of L{lay})")
            n_written += 1
 
    print(f"\nDone: {n_written} files written to {args.output_dir}")
    if problems:
        print("\nWARNINGS:")
        for s in problems:
            print("  -", s)
 
 
if __name__ == "__main__":
    main()