"""
AGHR System — Text Cleaner (Phase 1.2)
========================================
Cleans and normalizes raw document text.
"""

import re
import unicodedata
import hashlib
from pathlib import Path
from typing import Dict, List, Set

from loguru import logger
from tqdm import tqdm


class TextCleaner:
    """Text cleaning pipeline for raw documents."""

    def __init__(self, remove_html=True, normalize_unicode=True, 
                 remove_special_chars=True, min_text_length=50, dedup=True):
        self.remove_html = remove_html
        self.normalize_unicode = normalize_unicode
        self.remove_special_chars = remove_special_chars
        self.min_text_length = min_text_length
        self.dedup = dedup
        self._seen_hashes: Set[str] = set()

        self._html_re = re.compile(r"<[^>]+>")
        self._url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        self._multi_ws = re.compile(r"\s+")
        self._multi_nl = re.compile(r"\n{3,}")
        self._special_re = re.compile(r"[^\w\s.,;:!?'\"\-\(\)/\[\]%°\n]")
        self._page_re = re.compile(r"(page\s*\d+|^\d+$|©.*$)", re.IGNORECASE | re.MULTILINE)

    def clean_text(self, text: str) -> str:
        """Apply full cleaning pipeline to a single text."""
        if not text:
            return ""

        # Fix encoding
        for old, new in {"\u2018":"'","\u2019":"'","\u201c":'"',"\u201d":'"',
                         "\u2013":"-","\u2014":"-","\u2026":"...","\u00a0":" ",
                         "\ufeff":"","\u200b":"","\u200c":"","\u200d":""}.items():
            text = text.replace(old, new)

        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        if self.remove_html:
            text = self._html_re.sub(" ", text)
        text = self._page_re.sub("", text)
        text = self._url_re.sub("[URL]", text)
        if self.remove_special_chars:
            text = self._special_re.sub(" ", text)
        text = self._multi_nl.sub("\n\n", text)
        text = self._multi_ws.sub(" ", text)
        return text.strip()

    def clean_documents(self, documents: List[Dict], text_key="text") -> List[Dict]:
        """Clean a list of document dicts, filtering short/duplicate docs."""
        cleaned, removed_short, removed_dup = [], 0, 0

        for doc in tqdm(documents, desc="Cleaning documents"):
            clean = self.clean_text(doc.get(text_key, ""))
            if len(clean) < self.min_text_length:
                removed_short += 1
                continue
            if self.dedup:
                h = hashlib.md5(" ".join(clean.lower().split()).encode()).hexdigest()
                if h in self._seen_hashes:
                    removed_dup += 1
                    continue
                self._seen_hashes.add(h)

            doc_copy = doc.copy()
            doc_copy[text_key] = clean
            doc_copy.setdefault("metadata", {})["cleaned_length"] = len(clean)
            cleaned.append(doc_copy)

        logger.info(f"Cleaning: {len(cleaned)} kept, {removed_short} short, {removed_dup} dups")
        return cleaned

    def fuzzy_dedup(self, documents: List[Dict], text_key="text", threshold=0.8) -> List[Dict]:
        """Remove near-duplicates using MinHash LSH."""
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            logger.warning("datasketch not installed, skipping fuzzy dedup")
            return documents

        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        minhashes = {}
        for i, doc in enumerate(documents):
            m = MinHash(num_perm=128)
            for w in doc.get(text_key, "").lower().split():
                m.update(w.encode("utf-8"))
            minhashes[i] = m
            try:
                lsh.insert(str(i), m)
            except ValueError:
                pass

        keep = set()
        for i in range(len(documents)):
            if i not in keep:
                result = lsh.query(minhashes[i])
                keep.add(min(int(r) for r in result))

        deduped = [documents[i] for i in sorted(keep)]
        logger.info(f"Fuzzy dedup: {len(deduped)} kept, {len(documents)-len(deduped)} removed")
        return deduped
