from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src import Document, EmbeddingStore, RecursiveChunker


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CORPUS_CANDIDATES = [Path("data/chinh-sach-tmdt"), Path("data (1)/chinh-sach-tmdt")]
OUTPUT_PATH = Path("ket_qua_benchmark.txt")
CHUNK_SIZE = 500
TOP_K = 3

BENCHMARKS = [
    {
        "id": 1,
        "query": "Người mua có tối đa bao lâu để gửi yêu cầu trả hàng/hoàn tiền đối với đơn hàng thông thường và thực phẩm tươi sống hoặc đông lạnh?",
        "gold_doc_id": "quy-dinh-chung-tra-hang-hoan-tien",
        "gold_answer": "Đơn thông thường 15 ngày kể từ khi đơn được cập nhật Giao hàng thành công; thực phẩm tươi sống/đông lạnh 24 giờ.",
        "metadata_filter": None,
        "required_strings": ["15 ngày", "24 giờ"],
    },
    {
        "id": 2,
        "query": "Trong những trường hợp nào người mua có thể yêu cầu trả hàng/hoàn tiền? Hãy liệt kê ít nhất bốn trường hợp.",
        "gold_doc_id": "quy-dinh-chung-tra-hang-hoan-tien",
        "gold_answer": "Chưa nhận được hàng; nhận thiếu hàng/phụ kiện/quà tặng; người bán gửi sai hàng; hàng bể vỡ, hư hỏng, rò rỉ; hàng lỗi không hoạt động; khác mô tả của người bán.",
        "metadata_filter": None,
        "required_strings": ["chưa nhận được hàng", "thiếu hàng", "gửi sai hàng", "bể vỡ"],
    },
    {
        "id": 3,
        "query": "Shopee có hỗ trợ đổi sản phẩm trực tiếp không? Người mua nên làm gì nếu sản phẩm nhận được bị sai hoặc hư hỏng?",
        "gold_doc_id": "quy-dinh-chung-tra-hang-hoan-tien",
        "gold_answer": "Shopee chưa hỗ trợ yêu cầu đổi hàng. Người mua từ chối nhận khi đồng kiểm, hoặc gửi yêu cầu Trả hàng/Hoàn tiền trong thời hạn quy định.",
        "metadata_filter": None,
        "required_strings": ["đổi hàng", "trả hàng/hoàn tiền"],
    },
    {
        "id": 4,
        "query": "Sau khi Shopee chấp nhận hoàn tiền, người mua thanh toán khi nhận hàng có thể nhận tiền qua đâu và mất bao lâu?",
        "gold_doc_id": "thoi-gian-nhan-tien-hoan",
        "gold_answer": "Hoàn vào Ví ShopeePay trong 24 giờ nếu ví hoạt động bình thường, hoặc vào tài khoản ngân hàng mặc định đã liên kết trong 2 ngày làm việc.",
        "metadata_filter": None,
        "required_strings": ["24 giờ", "2 ngày làm việc"],
    },
    {
        "id": 5,
        "query": "Khi hệ thống ghi nhận đã trả hàng thành công nhưng Shop chưa nhận được hàng hoặc hàng hoàn gặp vấn đề, người bán phải phản hồi trong thời hạn bao lâu và thực hiện phản hồi ở đâu?",
        "gold_doc_id": "quan-ly-don-tra-hang-nguoi-ban",
        "gold_answer": "Phản hồi trong vòng 2 ngày kể từ ngày hệ thống cập nhật trả hàng thành công, qua Kênh Quản Lý Shop > Trả hàng/Hoàn tiền > Cần phản hồi > Phản hồi đến Shopee.",
        "metadata_filter": {"audience": "seller"},
        "required_strings": ["trong vòng 2 ngày", "phản hồi đến shopee"],
    },
]


class TfidfFallbackEmbedder:
    def __init__(self, texts: list[str]) -> None:
        tokenized = [self._features(text) for text in texts]
        document_frequency = Counter(
            feature for features in tokenized for feature in set(features)
        )
        self.vocabulary = {
            feature: index for index, feature in enumerate(sorted(document_frequency))
        }
        count = len(texts)
        self.idf = {
            feature: math.log((1 + count) / (1 + frequency)) + 1.0
            for feature, frequency in document_frequency.items()
        }

    @staticmethod
    def _features(text: str) -> list[str]:
        words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        return words + [f"{left}_{right}" for left, right in zip(words, words[1:])]

    def __call__(self, text: str) -> list[float]:
        counts = Counter(self._features(text))
        vector = [0.0] * len(self.vocabulary)
        for feature, count in counts.items():
            index = self.vocabulary.get(feature)
            if index is not None:
                vector[index] = (1.0 + math.log(count)) * self.idf[feature]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class CachedBatchEmbedder:
    def __init__(self, texts: list[str]) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)
        unique_texts = list(dict.fromkeys(texts))
        vectors = self.model.encode(
            unique_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.cache = {
            text: vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for text, vector in zip(unique_texts, vectors)
        }

    def __call__(self, text: str) -> list[float]:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def corpus_dir() -> Path:
    for candidate in CORPUS_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Không tìm thấy data/chinh-sach-tmdt hoặc data (1)/chinh-sach-tmdt")


def read_frontmatter(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not text.startswith("---"):
        return metadata
    for line in text.split("---", 2)[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata


def load_chunks(root: Path) -> tuple[list[Document], dict[str, int]]:
    chunker = RecursiveChunker(chunk_size=CHUNK_SIZE)
    documents: list[Document] = []
    chunks_per_file: dict[str, int] = {}
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = chunker.chunk(text)
        chunks_per_file[path.stem] = len(chunks)
        base_metadata = read_frontmatter(text)
        base_metadata.update({"source": path.name, "doc_id": path.stem})
        for index, chunk in enumerate(chunks):
            metadata = dict(base_metadata)
            metadata["chunk_index"] = index
            documents.append(Document(f"{path.stem}#{index}", chunk, metadata))
    return documents, chunks_per_file


def contains_normalized(text: str, expected: str) -> bool:
    normalize = lambda value: re.sub(r"\s+", " ", value.lower()).strip()
    return normalize(expected) in normalize(text)


def evaluate(
    store: EmbeddingStore,
    benchmark: dict,
    metadata_filter: dict | None,
) -> dict:
    results = store.search_with_filter(
        benchmark["query"], top_k=TOP_K, metadata_filter=metadata_filter
    )
    gold_rank = next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if result["metadata"].get("doc_id") == benchmark["gold_doc_id"]
        ),
        None,
    )
    context = "\n".join(result["content"] for result in results)
    missing = [
        value
        for value in benchmark["required_strings"]
        if not contains_normalized(context, value)
    ]
    context_has_answer = not missing
    points = 2 if context_has_answer and gold_rank == 1 else 1 if context_has_answer else 0
    top_3 = []
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        top_3.append(
            {
                "rank": rank,
                "chunk_id": result["id"].split("::", 1)[0],
                "doc_id": metadata.get("doc_id"),
                "audience": metadata.get("audience"),
                "score": round(float(result["score"]), 4),
                "preview": " ".join(result["content"].split())[:200],
            }
        )
    return {
        "gold_rank": gold_rank,
        "doc_id_in_top3": gold_rank is not None,
        "context_has_answer": context_has_answer,
        "missing_strings": missing,
        "points": points,
        "top_3": top_3,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    root = corpus_dir()
    chunks, chunks_per_file = load_chunks(root)
    all_texts = [document.content for document in chunks] + [
        benchmark["query"] for benchmark in BENCHMARKS
    ]

    mock_warning = None
    try:
        embedder = CachedBatchEmbedder(all_texts)
        embedding_backend = MODEL_NAME
        embedding_provider = "local"
    except Exception as exc:
        embedder = TfidfFallbackEmbedder([document.content for document in chunks])
        embedding_backend = "TF-IDF word+bigram fallback"
        embedding_provider = "fallback"
        mock_warning = f"Local embedding unavailable: {type(exc).__name__}: {exc}"

    store = EmbeddingStore("shopee_recursive_benchmark", embedding_fn=embedder)
    store.add_documents(chunks)

    query_results = []
    for benchmark in BENCHMARKS:
        item = {
            key: value
            for key, value in benchmark.items()
            if key != "required_strings"
        }
        item["result"] = evaluate(store, benchmark, benchmark["metadata_filter"])
        if benchmark["metadata_filter"]:
            item["ab_without_filter"] = evaluate(store, benchmark, None)
        query_results.append(item)

    payload = {
        "generated_at": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds"),
        "member_role": "R3 - Strategy",
        "strategy": f"RecursiveChunker(chunk_size={CHUNK_SIZE})",
        "embedding_backend": embedding_backend,
        "embedding_provider": embedding_provider,
        "mock_warning": mock_warning,
        "corpus_dir": str(root),
        "top_k": TOP_K,
        "chunks_per_file": chunks_per_file,
        "total_chunks": len(chunks),
        "queries": query_results,
        "total_points": sum(item["result"]["points"] for item in query_results),
        "max_points": len(query_results) * 2,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nĐã ghi kết quả vào {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
