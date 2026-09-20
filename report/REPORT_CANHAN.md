# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Le Thi Cham Anh

**MSSV:** 2A202602846

**Nhóm:** G00

**Ngày:** 20/09/2026

> Báo cáo sử dụng kết quả kiểm thử và benchmark được chạy trực tiếp trên repository ngày 20/09/2026.

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity)

**Độ tương tự cosine cao nghĩa là gì?**

Độ tương tự cosine cao cho biết hai vector embedding có hướng gần nhau, tức là hai đoạn văn có nội dung hoặc ý nghĩa ngữ nghĩa tương đồng. Giá trị càng gần 1 thì mức độ tương đồng càng cao.

**Ví dụ có độ tương tự cao:**

- Câu A: “Người mua có thể yêu cầu trả hàng trong vòng 15 ngày.”
- Câu B: “Thời hạn gửi yêu cầu hoàn trả của khách hàng là 15 ngày.”
- Lý do: Hai câu dùng từ khác nhau nhưng cùng diễn đạt thời hạn trả hàng.

**Ví dụ có độ tương tự thấp:**

- Câu A: “Shopee hoàn tiền vào Ví ShopeePay trong 24 giờ.”
- Câu B: “Recursive chunking chia văn bản theo nhiều cấp separator.”
- Lý do: Hai câu thuộc hai chủ đề hoàn toàn khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

Cosine similarity tập trung vào hướng của vector và ít bị ảnh hưởng bởi độ lớn của vector. Điều này phù hợp với text embedding vì mục tiêu chính là so sánh ý nghĩa, không phải độ dài hay độ lớn tuyệt đối của biểu diễn.

### Bài toán tính toán Chunking

**Tài liệu 10.000 ký tự, `chunk_size=500`, `overlap=50`:**

```text
step = chunk_size - overlap = 500 - 50 = 450
number_of_chunks = ceil((10000 - 500) / 450) + 1
                 = ceil(21,11) + 1
                 = 23 chunks
```

**Nếu overlap tăng lên 100:**

```text
step = 500 - 100 = 400
number_of_chunks = ceil((10000 - 500) / 400) + 1
                 = ceil(23,75) + 1
                 = 25 chunks
```

Overlap lớn hơn làm số chunk tăng từ 23 lên 25. Đổi lại, thông tin nằm tại ranh giới giữa hai chunk có nhiều khả năng được giữ nguyên ngữ cảnh, nhưng chi phí embedding và lưu trữ cũng tăng.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Chiến lược cá nhân: `RecursiveChunker`

Tôi chọn `RecursiveChunker` cho corpus chính sách Shopee. Thuật toán thử các separator theo thứ tự `"\n\n"`, `"\n"`, `". "`, `" "`, `""`, ưu tiên ranh giới đoạn và câu để giữ ngữ nghĩa. Nếu một mảnh vẫn dài hơn `chunk_size`, hàm `_split` tiếp tục đệ quy bằng separator nhỏ hơn.

Các base case gồm: văn bản rỗng trả `[]`; văn bản không vượt giới hạn trả một chunk; khi hết separator hoặc gặp separator rỗng thì chia cố định theo `chunk_size`. Sau bước chia sâu, các mảnh nhỏ liền kề được gom lại đến gần giới hạn nhằm tránh tạo nhiều chunk vụn.

### `SentenceChunker.chunk`

Tôi dùng regex lookbehind `(?<=[.!?])(?:[ \t]+|\n+)` để tách tại khoảng trắng hoặc xuống dòng nằm sau dấu kết câu, nhờ đó dấu câu không bị mất. Sau khi loại câu rỗng và strip khoảng trắng, các câu được gom theo `max_sentences_per_chunk`. Hạn chế còn lại là chữ viết tắt và một số số thập phân có thể bị nhận diện sai là ranh giới câu.

### `EmbeddingStore`

Mỗi `Document` được chuẩn hóa thành record gồm ID duy nhất, nội dung, metadata có thêm `doc_id`, và vector embedding. Khi search, query được embedding, tính cosine similarity với từng record, sắp xếp giảm dần và lấy `top_k`. `search_with_filter` lọc candidate theo metadata trước khi xếp hạng; `delete_document` tìm và xóa mọi record có `metadata["doc_id"]` tương ứng.

### `KnowledgeBaseAgent.answer`

Agent lấy top-k chunk, đánh số `[1]`, `[2]`, `[3]`, kèm nguồn rồi đưa vào prompt. Prompt yêu cầu chỉ dùng context, trích dẫn số nguồn và nói rõ khi không đủ thông tin. Nếu store không trả về kết quả, agent trả thông báo không tìm thấy và không gọi LLM không cần thiết.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết quả toàn bộ test hiện tại

```text
Lệnh: python -m pytest tests/ -q
Kết quả: 42 passed in 0.07s
Tổng số test: 42
```

Toàn bộ test cho chunking, cosine similarity, comparator, vector store, metadata filter, delete document và KnowledgeBaseAgent đều đã vượt qua.

### Kết quả riêng của `RecursiveChunker`

```text
Lệnh: python -m pytest tests/test_solution.py -k RecursiveChunker -v
Kết quả: 4 passed, 38 deselected
Tỷ lệ riêng RecursiveChunker: 4/4 = 100%
```

Runner trên 10 tài liệu Shopee với `chunk_size=500` cho kết quả:

- Tổng số chunk: **169**
- Độ dài trung bình: **399,2 ký tự**
- Chunk dài nhất: **499 ký tự**
- Chunk không vượt giới hạn: **169/169 (100%)**

Kết quả chi tiết được lưu trong `ket_qua_recursive_chunker.json`.

**Số lượng test toàn repo vượt qua:** **42 / 42**

**Số lượng test RecursiveChunker vượt qua:** **4 / 4**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|---|---|---|---|---|---|
| 1 | Người mua có 15 ngày để yêu cầu trả hàng. | Thời hạn hoàn trả sản phẩm là 15 ngày. | Cao | 0,2597 | Đúng |
| 2 | Người bán cần phản hồi đơn hoàn trong 2 ngày. | Shop phải trả lời yêu cầu hàng hoàn trong vòng 2 ngày. | Cao | 0,3756 | Đúng |
| 3 | Shopee hoàn tiền qua Ví ShopeePay. | Tiền được hoàn vào ví điện tử của người mua. | Cao | 0,3259 | Đúng |
| 4 | Sản phẩm bị hư hỏng khi vận chuyển. | Recursive chunking ưu tiên ranh giới đoạn văn. | Thấp | 0,1530 | Đúng |
| 5 | Người mua gửi yêu cầu trả hàng. | Hôm nay thời tiết có mưa lớn. | Thấp | 0,0792 | Đúng |

**Nhận xét:** Cặp 1 có điểm thấp hơn dự kiến dù cùng đề cập thời hạn 15 ngày. Nguyên nhân là TF-IDF word/bigram chỉ nhận diện từ vựng trùng nhau, chưa hiểu đầy đủ rằng “yêu cầu trả hàng” và “hoàn trả sản phẩm” gần nghĩa. Kết quả cho thấy lexical embedding tái lập tốt nhưng biểu diễn ngữ nghĩa kém hơn embedding từ mô hình học sâu.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

**Chiến lược sử dụng:** `RecursiveChunker(chunk_size=500)`

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được | Score | Relevant? | Câu trả lời của Agent |
|---|---|---|---|---|---|
| 1 | Người mua có tối đa bao lâu để gửi yêu cầu trả hàng/hoàn tiền đối với đơn hàng thông thường và thực phẩm tươi sống hoặc đông lạnh? | `chinh-sach-tra-hang-hoan-tien.md`, chunk 9 | 0,3405 | Có; gold document ở top-2 | Đơn thường: 15 ngày; thực phẩm tươi sống/đông lạnh: 24 giờ. |
| 2 | Trong những trường hợp nào người mua có thể yêu cầu trả hàng/hoàn tiền? Hãy liệt kê ít nhất bốn trường hợp. | `chinh-sach-tra-hang-hoan-tien.md`, chunk 6 | 0,2021 | Không đủ | Top-3 không chứa đủ bốn trường hợp vì danh sách bị trải qua nhiều chunk. |
| 3 | Shopee có hỗ trợ đổi sản phẩm trực tiếp không? Người mua nên làm gì nếu sản phẩm nhận được bị sai hoặc hư hỏng? | `chinh-sach-tra-hang-hoan-tien.md`, chunk 7 | 0,2112 | Không | Chunk chứa câu “chưa hỗ trợ yêu cầu đổi hàng” không vào top-3. |
| 4 | Sau khi Shopee chấp nhận hoàn tiền, người mua thanh toán khi nhận hàng có thể nhận tiền qua đâu và mất bao lâu? | `thoi-gian-nhan-tien-hoan.md`, chunk 1 | 0,2474 | Có; gold document ở top-1 | Ví ShopeePay trong 24 giờ; tài khoản ngân hàng liên kết trong 2 ngày làm việc. |
| 5 | Khi hệ thống ghi nhận đã trả hàng thành công nhưng Shop chưa nhận được hàng hoặc hàng hoàn gặp vấn đề, người bán phải phản hồi trong thời hạn bao lâu và thực hiện phản hồi ở đâu? | `quan-ly-don-tra-hang-nguoi-ban.md`, chunk 6 | 0,4703 | Có; gold document ở top-1 | Phản hồi trong 2 ngày tại Kênh Quản Lý Shop, mục Trả hàng/Hoàn tiền → Cần phản hồi. |

**Số câu có chunk liên quan trong top-3:** **3 / 5**

**Tổng điểm retrieval tự chấm:** **5/10**. Q1 đạt 1 điểm vì gold document ở top-2; Q4 và Q5 mỗi câu đạt 2 điểm vì gold document ở top-1 và context top-3 chứa đủ đáp án; Q2 và Q3 không có đủ gold answer trong top-3.

**Bài học rút ra:** Recursive chunking giữ được ranh giới tự nhiên, nhưng danh sách dài ở Q2 vẫn bị tách qua nhiều chunk. Metadata filter ở Q5 giúp chỉ tìm trong tài liệu dành cho người bán. Chất lượng retrieval còn phụ thuộc mạnh vào embedding: TF-IDF xử lý từ khóa tốt nhưng yếu với câu diễn đạt đồng nghĩa hoặc phủ định như Q3.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động | 5 / 5 |
| Hướng tiếp cận | 10 / 10 |
| Hoàn thiện code | 30 / 30 — 42/42 test pass |
| Dự đoán độ tương tự | 5 / 5 |
| Kết quả truy xuất | 5 / 10 |
| **Tổng phần cá nhân** | **55 / 60** |

## Việc cần hoàn thành trước khi nộp

- Điền họ tên, MSSV và tên nhóm.
- Nếu có thời gian, thử embedding ngữ nghĩa để cải thiện Q2 và Q3.
