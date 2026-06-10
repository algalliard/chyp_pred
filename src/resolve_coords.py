"""
src/resolve_coords.py
---------------------
Resolves genomic coordinates (chromosome, position, reference, alternate)
for VUS variants in data/trio_variants.parquet using the Ensembl REST API.
Writes results to data/vus_coords_resolved.csv.
"""

import os
import re
import time
import pandas as pd
import numpy as np
import requests

DATA_DIR = pd.io.common.Path("data")
IN_PARQUET = DATA_DIR / "trio_variants.parquet"
OUT_CSV = DATA_DIR / "vus_coords_resolved.csv"

def parse_ref_alt(hgvs):
    if not isinstance(hgvs, str):
        return None, None
    # Substitution: c.11777T>C
    m = re.search(r'c\.\d+([A-Z])>([A-Z])', hgvs)
    if m:
        return m.group(1), m.group(2)
    # Deletion: c.72delT or c.1513_1515delCCT
    m = re.search(r'c\.\d+(?:_\d+)?del([A-Z]+)', hgvs)
    if m:
        return m.group(1), '-'
    # Insertion: c.3635_3636insTG
    m = re.search(r'c\.\d+_\d+ins([A-Z]+)', hgvs)
    if m:
        return '-', m.group(1)
    # Delins: c.123_124delinsGA
    m = re.search(r'c\.\d+(?:_\d+)?delins([A-Z]+)', hgvs)
    if m:
        return 'N', m.group(1)
    return None, None

def get_enst_and_pos(row):
    transcripts = row.get("Transcripts", "")
    hgvs = row.get("HGVS_Coding", "")
    if not isinstance(transcripts, str) or not isinstance(hgvs, str):
        return None, None
    m_enst = re.search(r'(ENST\d+)', transcripts)
    m_pos = re.search(r'c\.(\d+)', hgvs)
    enst = m_enst.group(1) if m_enst else None
    pos = int(m_pos.group(1)) if m_pos else None
    return enst, pos

def resolve_coords_via_ensembl(enst, pos, max_retries=3):
    if not enst or pos is None:
        return None, None, None
    url = f"https://rest.ensembl.org/map/cds/{enst}/{pos}..{pos}"
    headers = {"Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                mappings = data.get("mappings", [])
                for mapping in mappings:
                    if mapping.get("coord_system") == "chromosome":
                        chrom = mapping.get("seq_region_name")
                        start = mapping.get("start")
                        assembly = mapping.get("assembly_name")
                        return chrom, start, assembly
                return None, None, None
            elif r.status_code == 429:
                # Rate limit hit, sleep and retry
                time.sleep(2 ** attempt)
            else:
                return None, None, None
        except Exception:
            time.sleep(1)
    return None, None, None

def main():
    print(f"Reading variants from {IN_PARQUET}...")
    df = pd.read_parquet(IN_PARQUET)
    
    # Filter for VUS variants (Uncertain significance)
    vus_all = df[df["Classification"].str.strip().str.lower() == "uncertain significance"].copy()
    print(f"Found {len(vus_all)} VUS rows.")
    
    # We assign row_id to map 1-to-1 to vus_unlabeled_meta.csv
    # In notebooks/06, row_id was assigned sequentially from 0 to len(vus_all)-1
    vus_all = vus_all.reset_index(drop=True)
    vus_all["row_id"] = np.arange(len(vus_all))
    
    # Identify unique ENST+pos pairs
    print("Extracting transcripts and CDS positions...")
    resolved_cache = {}
    
    # Iterate and build a cache of unique transcript-position pairs
    unique_pairs = []
    for idx, row in vus_all.iterrows():
        enst, pos = get_enst_and_pos(row)
        if enst and pos is not None:
            unique_pairs.append((enst, pos))
        else:
            unique_pairs.append((None, None))
            
    vus_all["enst"] = [p[0] for p in unique_pairs]
    vus_all["cds_pos"] = [p[1] for p in unique_pairs]
    
    unique_nonnull_pairs = list(set([p for p in unique_pairs if p[0] is not None and p[1] is not None]))
    print(f"Total unique transcript-position pairs to query: {len(unique_nonnull_pairs)}")
    
    # Fetch coordinates
    for i, (enst, pos) in enumerate(unique_nonnull_pairs):
        if (i + 1) % 20 == 0 or (i + 1) == len(unique_nonnull_pairs):
            print(f"Resolving coordinates: {i + 1}/{len(unique_nonnull_pairs)}...")
        
        chrom, start, assembly = resolve_coords_via_ensembl(enst, pos)
        resolved_cache[(enst, pos)] = (chrom, start, assembly)
        # Be polite to the API
        time.sleep(0.08)
        
    print("Mapping coordinates back to VUS rows...")
    chroms = []
    positions = []
    refs = []
    alts = []
    
    for idx, row in vus_all.iterrows():
        enst = row["enst"]
        pos = row["cds_pos"]
        hgvs = row["HGVS_Coding"]
        
        chrom, start, _ = resolved_cache.get((enst, pos), (None, None, None))
        ref, alt = parse_ref_alt(hgvs)
        
        chroms.append(chrom)
        positions.append(start)
        refs.append(ref)
        alts.append(alt)
        
    vus_all["Chr"] = chroms
    vus_all["Position"] = positions
    vus_all["Ref"] = refs
    vus_all["Alt"] = alts
    
    # Keep only row_id and coordinate mapping columns
    out_df = vus_all[["row_id", "Gene", "HGVS_Coding", "Chr", "Position", "Ref", "Alt"]]
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved resolved coordinates to {OUT_CSV} (done)")

if __name__ == "__main__":
    main()
