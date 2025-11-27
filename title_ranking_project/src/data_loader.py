# src/data_loader.py
import os
import glob
import json
import pandas as pd

def load_arxiv_excel(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_excel(path)
    except Exception as e:
        print("Error reading arxiv xlsx:", e)
        return pd.DataFrame()

def load_parquets(folder, pattern="icml_*-00000-of-00001.parquet"):
    files = glob.glob(os.path.join(folder, pattern))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def read_jsonl_to_df(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(rows)

def extract_title_abstract_from_s2orc(metadata_df):
    # handle nested `metadata` objects commonly in S2ORC
    if metadata_df.empty:
        return pd.DataFrame()
    if 'title' in metadata_df.columns and 'abstract' in metadata_df.columns:
        return metadata_df[['title','abstract']].copy()
    if 'metadata' in metadata_df.columns:
        # flatten nested metadata
        meta_flat = pd.json_normalize(metadata_df['metadata'].apply(lambda x: x if isinstance(x, dict) else {}))
        if 'title' in meta_flat.columns and 'abstract' in meta_flat.columns:
            return meta_flat[['title','abstract']].copy()
    # try common alternative keys
    for col in metadata_df.columns:
        if metadata_df[col].dtype == 'object' and metadata_df[col].apply(lambda x: isinstance(x, dict)).any():
            flat = pd.json_normalize(metadata_df[col].dropna().apply(lambda x: x if isinstance(x, dict) else {}))
            if 'title' in flat.columns and 'abstract' in flat.columns:
                return flat[['title','abstract']].copy()
    return pd.DataFrame()

def build_combined_dataset(datasets_folder):
    combined = pd.DataFrame(columns=['title','abstract'])

    # ArXiv
    arxiv_path = os.path.join(datasets_folder, "arxiv_data_210930-054931.xlsx")
    arxiv = load_arxiv_excel(arxiv_path)
    if not arxiv.empty:
        if 'title' in arxiv.columns and 'abstract' in arxiv.columns:
            combined = pd.concat([combined, arxiv[['title','abstract']]], ignore_index=True)

    # ICML parquets
    icml_df = load_parquets(datasets_folder)
    if not icml_df.empty:
        if 'title' in icml_df.columns and 'abstract' in icml_df.columns:
            combined = pd.concat([combined, icml_df[['title','abstract']]], ignore_index=True)

    # S2ORC-like metadata / pdf_parses
    meta_path = os.path.join(datasets_folder, "metadata", "sample.jsonl")
    pdf_parses_path = os.path.join(datasets_folder, "pdf_parses", "sample.jsonl")
    meta_df = read_jsonl_to_df(meta_path)
    pdf_df = read_jsonl_to_df(pdf_parses_path)

    meta_extracted = extract_title_abstract_from_s2orc(meta_df)
    if not meta_extracted.empty:
        combined = pd.concat([combined, meta_extracted], ignore_index=True)

    pdf_extracted = extract_title_abstract_from_s2orc(pdf_df)
    if not pdf_extracted.empty:
        combined = pd.concat([combined, pdf_extracted], ignore_index=True)

    # basic cleanup
    combined = combined.dropna(subset=['title','abstract'])
    combined['title'] = combined['title'].astype(str).str.strip()
    combined['abstract'] = combined['abstract'].astype(str).str.strip()
    combined = combined[combined['title'] != ""]
    combined = combined.reset_index(drop=True)
    return combined
