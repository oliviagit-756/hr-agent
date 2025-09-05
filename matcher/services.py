# matcher/services.py
import os, pandas as pd, numpy as np, faiss, torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# Where artifacts live (change via env var ARTIFACTS_DIR if needed)
ART_DIR = os.getenv("ARTIFACTS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts"))
JD_INDEX_PATH   = os.path.join(ART_DIR, "jd_index.faiss")
JD_META_PATH    = os.path.join(ART_DIR, "jd_meta.csv")
RES_INDEX_PATH  = os.path.join(ART_DIR, "resume_index.faiss")
RES_META_PATH   = os.path.join(ART_DIR, "resume_meta.csv")
CE_MODEL_DIR    = os.path.join(ART_DIR, "ce_hr_finetuned")  # optional

_bi = None
_ce = None
_jd_index = None
_jd_meta = None
_res_index = None
_res_meta = None

def _load():
    """Lazy-load models and indices once per process."""
    global _bi, _ce, _jd_index, _jd_meta, _res_index, _res_meta
    if _jd_index is None and os.path.exists(JD_INDEX_PATH):
        _jd_index = faiss.read_index(JD_INDEX_PATH)
        _jd_meta  = pd.read_csv(JD_META_PATH)
    if _res_index is None and os.path.exists(RES_INDEX_PATH):
        _res_index = faiss.read_index(RES_INDEX_PATH)
        _res_meta  = pd.read_csv(RES_META_PATH)
    if _bi is None:
        _bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if _ce is None:
        ce_path = CE_MODEL_DIR if os.path.exists(CE_MODEL_DIR) else "cross-encoder/ms-marco-MiniLM-L-6-v2"
        _ce = CrossEncoder(ce_path, max_length=512)

def health_info():
    _load()
    return dict(
        jd_vectors=int(_jd_index.ntotal) if _jd_index is not None else 0,
        res_vectors=int(_res_index.ntotal) if _res_index is not None else 0,
        emb_dim=int((_jd_index or _res_index).d) if (_jd_index or _res_index) is not None else None,
        ce_model="finetuned" if os.path.exists(CE_MODEL_DIR) else "ms-marco-miniLM",
    )

def _contains(s: pd.Series, q: str):
    return s.fillna("").str.contains(q, case=False, regex=False)

def pick_jd_text_by_keywords(keywords: str) -> dict:
    _load()
    if _jd_meta is None:
        return {}
    m = _contains(_jd_meta["Job Title"], keywords) | _contains(_jd_meta["Description"], keywords)
    if not m.any():
        return {}
    hit = _jd_meta[m].iloc[0]
    return {"title": hit["Job Title"], "text": hit["text"]}

def search_jobs_for_resume(resume_text: str, k: int = 25, top_m: int = 10) -> pd.DataFrame:
    """Resume -> JDs (needs JD index)."""
    _load()
    if _jd_index is None:
        raise RuntimeError("JD index not found. Place jd_index.faiss & jd_meta.csv in artifacts/")
    q = _bi.encode([resume_text], convert_to_numpy=True, normalize_embeddings=True)
    D, I = _jd_index.search(q, k)
    df = _jd_meta.iloc[I[0]][["Job Title", "Description", "text"]].copy()
    df["bi_score"] = D[0]
    pairs = [(resume_text, t) for t in df["text"].tolist()]
    logits = _ce.predict(pairs)
    probs  = torch.sigmoid(torch.tensor(logits)).numpy()
    df["ce_prob"] = probs
    df["score"]   = 0.3 * df["bi_score"] + 0.7 * df["ce_prob"]
    df["description_preview"] = df["Description"].str.slice(0, 220) + "..."
    return df.sort_values("score", ascending=False).head(top_m)[
        ["Job Title", "description_preview", "bi_score", "ce_prob", "score"]
    ].reset_index(drop=True)

def search_candidates_for_jd(jd_text: str, k: int = 25, top_m: int = 10) -> pd.DataFrame:
    """JD -> Resumes (needs Resume index)."""
    _load()
    if _res_index is None:
        raise RuntimeError("Resume index not found. Place resume_index.faiss & resume_meta.csv in artifacts/")
    q = _bi.encode([jd_text], convert_to_numpy=True, normalize_embeddings=True)
    D, I = _res_index.search(q, k)
    df = _res_meta.iloc[I[0]][["Category", "Resume"]].copy()
    df["bi_score"] = D[0]
    df["resume_preview"] = df["Resume"].str.slice(0, 220) + "..."
    pairs = [(jd_text, t) for t in df["Resume"].tolist()]
    logits = _ce.predict(pairs)
    probs  = torch.sigmoid(torch.tensor(logits)).numpy()
    df["ce_prob"] = probs
    df["score"]   = 0.3 * df["bi_score"] + 0.7 * df["ce_prob"]
    return df.sort_values("score", ascending=False).head(top_m)[
        ["Category", "resume_preview", "bi_score", "ce_prob", "score"]
    ].reset_index(drop=True)
