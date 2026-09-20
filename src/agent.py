from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong kho tri thức."

        context_blocks = []
        for index, result in enumerate(results, start=1):
            source = result["metadata"].get("source", result["metadata"].get("doc_id", "unknown"))
            context_blocks.append(f"[{index}] Nguồn: {source}\n{result['content']}")

        prompt = (
            "Bạn là trợ lý hỏi đáp dựa trên kho tri thức. Chỉ sử dụng ngữ cảnh "
            "được cung cấp; nếu ngữ cảnh không đủ, hãy nói rõ không tìm thấy thông tin. "
            "Khi trả lời, hãy trích dẫn nguồn bằng ký hiệu [1], [2], ...\n\n"
            f"Ngữ cảnh:\n{'\n\n'.join(context_blocks)}\n\n"
            f"Câu hỏi: {question}\n"
            "Trả lời:"
        )
        return str(self.llm_fn(prompt))
