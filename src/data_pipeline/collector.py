"""
AGHR System — Data Collector (Phase 1.1)
==========================================
Collects documents from multiple sources:
  - PDFs (medical guidelines, reports)
  - Web pages (WHO, NIH articles)
  - HuggingFace datasets (HotpotQA, PubMedQA)
  - Local text/CSV/JSON files

Outputs structured document records with metadata.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from loguru import logger
from tqdm import tqdm


@dataclass
class Document:
    """Represents a collected document with metadata."""
    doc_id: str
    text: str
    source: str              # "pdf", "web", "dataset", "file"
    source_name: str         # filename, URL, or dataset name
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        return cls(**d)


class DataCollector:
    """
    Multi-source data collection engine.

    Supports:
      - PDF extraction (PyPDF2 / pdfplumber)
      - Web scraping (trafilatura)
      - HuggingFace datasets (HotpotQA, PubMedQA)
      - Local files (txt, csv, json)
    """

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.documents: List[Document] = []
        self._doc_counter = 0

    def _next_id(self) -> str:
        self._doc_counter += 1
        return f"DOC_{self._doc_counter:05d}"

    # ── PDF Extraction ────────────────────────────────────────

    def collect_pdfs(self, pdf_dir: str) -> List[Document]:
        """
        Extract text from all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files.

        Returns:
            List of Document objects extracted from PDFs.
        """
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            logger.warning(f"PDF directory not found: {pdf_dir}")
            return []

        pdf_files = list(pdf_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        docs = []
        for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
            try:
                text = self._extract_pdf(pdf_file)
                if text and len(text.strip()) > 50:
                    doc = Document(
                        doc_id=self._next_id(),
                        text=text,
                        source="pdf",
                        source_name=pdf_file.name,
                        metadata={
                            "filepath": str(pdf_file),
                            "size_bytes": pdf_file.stat().st_size,
                            "num_chars": len(text),
                        },
                    )
                    docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to extract {pdf_file.name}: {e}")

        self.documents.extend(docs)
        logger.info(f"Collected {len(docs)} documents from PDFs")
        return docs

    def _extract_pdf(self, pdf_path: Path) -> str:
        """Extract text from a single PDF file."""
        try:
            import pdfplumber

            texts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        texts.append(page_text)
            return "\n\n".join(texts)

        except ImportError:
            # Fallback to PyPDF2
            from PyPDF2 import PdfReader

            reader = PdfReader(str(pdf_path))
            texts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            return "\n\n".join(texts)

    # ── Web Scraping ──────────────────────────────────────────

    def collect_web(self, urls: List[str]) -> List[Document]:
        """
        Scrape text content from web URLs.

        Args:
            urls: List of URLs to scrape.

        Returns:
            List of Document objects from web pages.
        """
        try:
            import trafilatura
        except ImportError:
            logger.error("trafilatura not installed. Run: pip install trafilatura")
            return []

        docs = []
        for url in tqdm(urls, desc="Scraping web pages"):
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(downloaded)
                    if text and len(text.strip()) > 50:
                        doc = Document(
                            doc_id=self._next_id(),
                            text=text,
                            source="web",
                            source_name=url,
                            metadata={"url": url, "num_chars": len(text)},
                        )
                        docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")

        self.documents.extend(docs)
        logger.info(f"Collected {len(docs)} documents from web")
        return docs

    # ── HuggingFace Datasets ──────────────────────────────────

    def collect_hotpotqa(self, split: str = "train[:2000]") -> List[Document]:
        """
        Load HotpotQA dataset for multi-hop reasoning.

        Each record includes: question, answer, supporting_facts, context paragraphs.

        Args:
            split: Dataset split to load (e.g., "train[:2000]" for pilot).

        Returns:
            List of Document objects from HotpotQA contexts + QA pairs.
        """
        from datasets import load_dataset

        logger.info(f"Loading HotpotQA ({split})...")
        dataset = load_dataset("hotpot_qa", "distractor", split=split)

        docs = []
        qa_pairs = []

        for idx, item in enumerate(tqdm(dataset, desc="Processing HotpotQA")):
            # Extract context paragraphs as documents
            titles = item.get("context", {}).get("title", [])
            sentences_list = item.get("context", {}).get("sentences", [])

            for title, sentences in zip(titles, sentences_list):
                text = " ".join(sentences)
                if len(text.strip()) > 30:
                    doc = Document(
                        doc_id=self._next_id(),
                        text=text,
                        source="dataset",
                        source_name="hotpotqa",
                        metadata={
                            "title": title,
                            "dataset_index": idx,
                            "question": item.get("question", ""),
                        },
                    )
                    docs.append(doc)

            # Save QA pair
            qa_pairs.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "type": item.get("type", ""),
                "level": item.get("level", ""),
                "supporting_facts": {
                    "title": item.get("supporting_facts", {}).get("title", []),
                    "sent_id": item.get("supporting_facts", {}).get("sent_id", []),
                },
                "source": "hotpotqa",
            })

        self.documents.extend(docs)

        # Save QA pairs
        qa_path = self.output_dir.parent / "qa_pairs" / "hotpotqa_qa.json"
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Collected {len(docs)} context docs + {len(qa_pairs)} QA pairs from HotpotQA"
        )
        return docs

    def collect_pubmedqa(self, split: str = "train[:1000]") -> List[Document]:
        """
        Load PubMedQA dataset for domain-specific medical QA.

        Args:
            split: Dataset split to load.

        Returns:
            List of Document objects from PubMedQA.
        """
        from datasets import load_dataset

        logger.info(f"Loading PubMedQA ({split})...")

        try:
            dataset = load_dataset("pubmed_qa", "pqa_labeled", split=split)
        except Exception:
            # Fallback: try alternative config
            logger.warning("pqa_labeled not available, trying pqa_artificial...")
            dataset = load_dataset("pubmed_qa", "pqa_artificial", split=split)

        docs = []
        qa_pairs = []

        for idx, item in enumerate(tqdm(dataset, desc="Processing PubMedQA")):
            # Context from the abstract / long_answer
            context_parts = item.get("context", {})
            if isinstance(context_parts, dict):
                contexts = context_parts.get("contexts", [])
                text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
            else:
                text = str(context_parts)

            long_answer = item.get("long_answer", "")
            if long_answer:
                text = text + "\n\n" + long_answer

            if len(text.strip()) > 30:
                doc = Document(
                    doc_id=self._next_id(),
                    text=text,
                    source="dataset",
                    source_name="pubmedqa",
                    metadata={
                        "pubid": item.get("pubid", idx),
                        "question": item.get("question", ""),
                    },
                )
                docs.append(doc)

            # Save QA pair
            qa_pairs.append({
                "question": item.get("question", ""),
                "answer": item.get("final_decision", ""),
                "long_answer": long_answer,
                "source": "pubmedqa",
            })

        self.documents.extend(docs)

        # Save QA pairs
        qa_path = self.output_dir.parent / "qa_pairs" / "pubmedqa_qa.json"
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Collected {len(docs)} context docs + {len(qa_pairs)} QA pairs from PubMedQA"
        )
        return docs

    # ── Local Files ───────────────────────────────────────────

    def collect_text_files(self, text_dir: str) -> List[Document]:
        """Load .txt files from a directory."""
        path = Path(text_dir)
        if not path.exists():
            logger.warning(f"Text directory not found: {text_dir}")
            return []

        docs = []
        for txt_file in path.glob("*.txt"):
            try:
                text = txt_file.read_text(encoding="utf-8")
                if len(text.strip()) > 50:
                    doc = Document(
                        doc_id=self._next_id(),
                        text=text,
                        source="file",
                        source_name=txt_file.name,
                        metadata={"filepath": str(txt_file)},
                    )
                    docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to read {txt_file.name}: {e}")

        self.documents.extend(docs)
        logger.info(f"Collected {len(docs)} text files from {text_dir}")
        return docs

    def collect_json_files(self, json_dir: str) -> List[Document]:
        """Load documents from JSON files (expected: list of {text, metadata} objects)."""
        path = Path(json_dir)
        if not path.exists():
            return []

        docs = []
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    records = json.load(f)

                if isinstance(records, list):
                    for record in records:
                        text = record.get("text", "")
                        if len(text.strip()) > 50:
                            doc = Document(
                                doc_id=self._next_id(),
                                text=text,
                                source="file",
                                source_name=json_file.name,
                                metadata=record.get("metadata", {}),
                            )
                            docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {json_file.name}: {e}")

        self.documents.extend(docs)
        logger.info(f"Collected {len(docs)} docs from JSON files in {json_dir}")
        return docs

    # ── Save / Load ───────────────────────────────────────────

    def save_documents(self, filepath: Optional[str] = None) -> str:
        """Save all collected documents to JSONL."""
        if filepath is None:
            filepath = str(self.output_dir / "all_documents.jsonl")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(self.documents)} documents → {filepath}")
        return filepath

    @classmethod
    def load_documents(cls, filepath: str) -> List[Document]:
        """Load documents from JSONL file."""
        docs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(Document.from_dict(json.loads(line)))
        logger.info(f"Loaded {len(docs)} documents from {filepath}")
        return docs

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        stats = {
            "total_documents": len(self.documents),
            "by_source": {},
            "total_chars": 0,
        }
        for doc in self.documents:
            source = doc.source_name
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            stats["total_chars"] += len(doc.text)

        stats["avg_chars"] = (
            stats["total_chars"] / len(self.documents) if self.documents else 0
        )
        return stats


# ── CLI Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.logger import setup_logger
    from src.utils.helpers import load_config

    setup_logger()
    config = load_config("config/config.yaml")

    collector = DataCollector(output_dir=config["data"]["raw_dir"])

    # Collect HotpotQA (pilot)
    hotpot_cfg = config["data"]["datasets"][0]
    collector.collect_hotpotqa(split=hotpot_cfg["split"])

    # Collect PubMedQA
    pubmed_cfg = config["data"]["datasets"][1]
    collector.collect_pubmedqa(split=pubmed_cfg["split"])

    # Save
    collector.save_documents()

    # Stats
    stats = collector.get_stats()
    logger.info(f"Collection stats: {json.dumps(stats, indent=2)}")
