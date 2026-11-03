"""Download the two public source datasets and record their SHA-256 for lineage.

ACS 2018 1-year PUMS, California (US Census, public domain) -> Folktables ACSIncome task.
NYC TLC yellow taxi trips, January 2023 (NYC Open Data terms).

The census zip is read without unpacking, then deleted. curl is used for
resumable downloads with retries (census.gov connections stall).
"""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from folktables import ACSIncome
from folktables.acs import adult_filter

DATA = Path("data")
ACS_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip"
TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"


def fetch(url, dest):
    """Download to dest (skipped if present) and return its sha256."""
    if not dest.exists():
        print(f"downloading {url}")
        part = dest.with_suffix(".part")
        subprocess.run(["curl", "-fL", "-C", "-", "--retry", "20", "--retry-all-errors",
                        "--speed-time", "60", "--speed-limit", "1000", "-o", part, url], check=True)
        part.rename(dest)
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_acs_income(zip_path, out):
    cols = ACSIncome.features + ["ST", "PINCP", "PWGTP"]
    with zipfile.ZipFile(zip_path) as z:
        parts = [n for n in z.namelist() if n.endswith(".csv")]
        df = pd.concat(pd.read_csv(z.open(n), usecols=cols) for n in parts)
    df = adult_filter(df)  # Folktables ACSIncome population filter
    out_df = df[ACSIncome.features].fillna(-1).astype("int32")
    out_df["ST"] = df["ST"].astype("int16")
    out_df["label"] = ACSIncome.target_transform(df["PINCP"]).astype("int8")
    out_df.reset_index(drop=True).to_parquet(out, index=False)
    return len(out_df)


def main():
    DATA.mkdir(exist_ok=True)
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    acs_out = DATA / "acs_income_ca_2018.parquet"
    if not acs_out.exists():
        zip_path = DATA / "csv_pca_2018.zip"
        sha = fetch(ACS_URL, zip_path)
        rows = build_acs_income(zip_path, acs_out)
        zip_path.unlink()  # the sha256 is what lineage needs
        manifest["acs-income-ca-2018"] = {"source": ACS_URL, "sha256": sha, "rows": rows,
                                       "path": str(acs_out)}

    tlc_out = DATA / "nyc_tlc_yellow_2023_01.parquet"
    sha = fetch(TLC_URL, tlc_out)
    manifest["nyc-tlc-yellow-2023-01"] = {"source": TLC_URL, "sha256": sha,
                                          "rows": pd.read_parquet(tlc_out, columns=["VendorID"]).shape[0],
                                          "path": str(tlc_out)}

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
