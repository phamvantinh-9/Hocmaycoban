#  Dự đoán khả năng mua ô tô bằng thuật toán KNN

Dự án xây dựng hệ thống dự đoán khả năng mua ô tô của khách hàng bằng thuật toán **K-Nearest Neighbors (KNN)**.

Hệ thống sử dụng các thông tin của khách hàng như **tuổi, thu nhập, số lần tìm hiểu xe và số lần đến showroom** để dự đoán khách hàng có khả năng mua ô tô hay không.

---

##  Thông tin đề tài

- **Môn học:** Học máy cơ bản (Machine Learning)
- **Thuật toán:** K-Nearest Neighbors (KNN)
- **Bài toán:** Phân loại nhị phân
- **Đối tượng:** Khách hàng mua ô tô
- **Ngôn ngữ:** Python
- **Thư viện chính:** Scikit-learn
- **API:** FastAPI

---

##  Mục tiêu

Xây dựng mô hình KNN có khả năng dự đoán khách hàng có khả năng mua ô tô hay không dựa trên các đặc điểm của khách hàng.

Các thông tin được sử dụng gồm:

- Tuổi
- Thu nhập hàng tháng
- Số lần tìm hiểu xe
- Số lần đến showroom

Mô hình dựa trên những khách hàng đã có dữ liệu để tìm ra những khách hàng có đặc điểm gần giống nhất với khách hàng mới.

---

##  Giới thiệu thuật toán KNN

**K-Nearest Neighbors (KNN)** là một thuật toán học máy được sử dụng cho các bài toán **phân loại** và **hồi quy**.

Ý tưởng chính của KNN là:

> Một đối tượng mới thường có đặc điểm giống với những đối tượng ở gần nó trong không gian dữ liệu.

Khi có một khách hàng mới, mô hình sẽ tìm ra **K khách hàng gần nhất** trong dữ liệu đã biết và dựa vào nhóm của những khách hàng này để đưa ra dự đoán.

---

##  Cơ chế hoạt động

Quá trình dự đoán của KNN gồm các bước:

### Bước 1: Nhận dữ liệu khách hàng mới

Ví dụ:

```json
{
  "features": [30, 38, 4, 2]
}
```

Trong đó:

- `30`: Tuổi
- `38`: Thu nhập 38 triệu đồng/tháng
- `4`: Số lần tìm hiểu xe
- `2`: Số lần đến showroom

### Bước 2: Tính khoảng cách

Mô hình tính khoảng cách giữa khách hàng mới và từng khách hàng trong tập dữ liệu.

Khoảng cách Euclid được tính theo công thức:

```text
d = √[(x1-y1)² + (x2-y2)² + ... + (xn-yn)²]
```

Khoảng cách càng nhỏ thì hai khách hàng càng giống nhau.

### Bước 3: Tìm K khách hàng gần nhất

Ví dụ chọn:

```text
K = 5
```

Mô hình sẽ tìm ra 5 khách hàng có khoảng cách gần nhất với khách hàng mới.

### Bước 4: Bỏ phiếu

Giả sử 5 khách hàng gần nhất có:

```text
3 khách hàng → Có mua
2 khách hàng → Không mua
```

Kết quả cuối cùng:

```text
Có khả năng mua
```

---

##  Lựa chọn giá trị K

Giá trị **K** quyết định số lượng hàng xóm được sử dụng để đưa ra dự đoán.

Ví dụ:

```text
K = 1
```

Mô hình chỉ xem xét khách hàng gần nhất.

```text
K = 5
```

Mô hình xem xét 5 khách hàng gần nhất.

### K quá nhỏ

- Dễ bị ảnh hưởng bởi dữ liệu nhiễu.
- Mô hình có thể dự đoán không ổn định.

### K quá lớn

- Có thể làm mất đi đặc điểm của nhóm gần nhất.
- Kết quả có thể bị ảnh hưởng bởi quá nhiều dữ liệu xa.

Vì vậy cần lựa chọn K phù hợp với tập dữ liệu.

---

##  Dữ liệu đầu vào

Mô hình sử dụng 4 đặc trưng:

| Đặc trưng | Ý nghĩa | Đơn vị |
|---|---|---|
| Tuổi | Tuổi của khách hàng | Tuổi |
| Thu nhập | Thu nhập hàng tháng | Triệu đồng |
| Tìm hiểu xe | Số lần khách hàng tìm hiểu xe | Lần |
| Showroom | Số lần khách hàng đến showroom | Lần |

---

##  Ví dụ dữ liệu

| Khách hàng | Tuổi | Thu nhập | Tìm hiểu xe | Showroom | Kết quả |
|---|---:|---:|---:|---:|---|
| KH01 | 21 | 10 | 1 | 0 | Không mua |
| KH02 | 23 | 12 | 2 | 0 | Không mua |
| KH03 | 25 | 15 | 2 | 1 | Không mua |
| KH04 | 27 | 20 | 3 | 1 | Không mua |
| KH05 | 30 | 35 | 4 | 2 | Có mua |
| KH06 | 32 | 40 | 5 | 2 | Có mua |
| KH07 | 35 | 45 | 5 | 3 | Có mua |
| KH08 | 40 | 55 | 6 | 3 | Có mua |

---

##  Kết quả dự đoán

Mô hình sử dụng hai nhãn:

```text
0 → Không có khả năng mua
1 → Có khả năng mua
```

Ví dụ khách hàng mới:

```text
Tuổi: 30
Thu nhập: 38 triệu
Tìm hiểu xe: 4 lần
Đến showroom: 2 lần
```

Dữ liệu gửi đến API:

```json
{
  "features": [30, 38, 4, 2]
}
```

Kết quả:

```json
{
  "prediction": 1,
  "label": "Có khả năng mua"
}
```

---

##  Cấu trúc dự án

```text
KNN-Car-Purchase-Prediction/
│
├── app/
│   └── main.py
│
├── training/
│   └── train.py
│
├── models/
│   └── knn.joblib
│
├── data/
│   └── customers.csv
│
├── requirements.txt
│
├── Dockerfile
│
├── docker-compose.yml
│
└── README.md
```

---

##  Chạy dự án

### 1. Tạo môi trường ảo

Mở Terminal hoặc Command Prompt tại thư mục dự án:

```bash
python -m venv .venv
```

### 2. Kích hoạt môi trường

Đối với Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu sử dụng Command Prompt:

```cmd
.venv\Scripts\activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Huấn luyện mô hình

```bash
python training/train.py
```

Sau khi chạy thành công, mô hình được lưu tại:

```text
models/knn.joblib
```

### 5. Khởi động API

```bash
uvicorn app.main:app --reload --port 3000
```

API sẽ chạy tại:

```text
http://127.0.0.1:3000
```

---

##  Các điểm cuối API

### Kiểm tra trạng thái

```text
GET /health
```

Dùng để kiểm tra API và mô hình có đang hoạt động hay không.

Ví dụ:

```json
{
  "status": "ok",
  "model": "KNN"
}
```

---

### Dự đoán khả năng mua

```text
POST /api/v1/predict
```

Dữ liệu đầu vào:

```json
{
  "features": [30, 38, 4, 2]
}
```

Kết quả:

```json
{
  "prediction": 1,
  "label": "Có khả năng mua"
}
```

---

### Tài liệu API

FastAPI cung cấp giao diện tài liệu API tại:

```text
GET /docs
```

Sau khi chạy chương trình, truy cập:

```text
http://127.0.0.1:3000/docs
```

Tại đây có thể nhập trực tiếp dữ liệu và thực hiện dự đoán.

---

##  Kiểm thử

Có thể kiểm tra API bằng cách gửi dữ liệu:

```json
{
  "features": [30, 38, 4, 2]
}
```

Mô hình sẽ trả về kết quả dự đoán:

```text
1 - Có khả năng mua
```

Hoặc:

```text
0 - Không có khả năng mua
```

---

##  Khoảng cách Euclid

KNN sử dụng khoảng cách để xác định mức độ giống nhau giữa các đối tượng.

Với hai khách hàng:

```text
A = (x1, x2, x3, x4)

B = (y1, y2, y3, y4)
```

Khoảng cách Euclid:

```text
d(A,B) =
√[(x1-y1)² + (x2-y2)² + (x3-y3)² + (x4-y4)²]
```

Giá trị khoảng cách:

```text
Nhỏ → Hai khách hàng giống nhau hơn

Lớn → Hai khách hàng khác nhau hơn
```

---

##  Chuẩn hóa dữ liệu

Các đặc trưng có thể có đơn vị và phạm vi giá trị khác nhau.

Ví dụ:

```text
Tuổi:             18 - 60
Thu nhập:         5 - 100 triệu
Số lần tìm hiểu:  0 - 20
Showroom:         0 - 10
```

Thu nhập có giá trị lớn hơn nhiều so với số lần tìm hiểu xe.

Nếu không xử lý, đặc trưng có giá trị lớn có thể ảnh hưởng nhiều đến việc tính khoảng cách.

Vì vậy mô hình có thể sử dụng phương pháp **chuẩn hóa dữ liệu** trước khi đưa vào KNN.

---

##  Ưu điểm của KNN

- Dễ hiểu và dễ giải thích.
- Cách hoạt động trực quan.
- Không cần xây dựng công thức mô hình phức tạp.
- Có thể sử dụng cho bài toán phân loại và hồi quy.
- Phù hợp với dữ liệu có số lượng không quá lớn.

---

##  Nhược điểm của KNN

- Dự đoán có thể chậm khi dữ liệu lớn.
- Cần tính khoảng cách với nhiều điểm dữ liệu.
- Nhạy cảm với dữ liệu nhiễu.
- Kết quả phụ thuộc vào việc lựa chọn K.
- Các đặc trưng có thang đo khác nhau có thể ảnh hưởng đến khoảng cách.
- Cần chuẩn hóa dữ liệu trong nhiều trường hợp.

---

##  Ứng dụng thực tế

KNN có thể được sử dụng trong nhiều bài toán:

- Phân loại khách hàng.
- Dự đoán hành vi mua hàng.
- Nhận dạng hình ảnh.
- Phân loại văn bản.
- Hệ thống gợi ý.
- Nhận dạng mẫu dữ liệu.

Trong dự án này, KNN được sử dụng để **dự đoán khả năng mua ô tô của khách hàng**.

---

##  Công nghệ sử dụng

| Công nghệ | Mục đích |
|---|---|
| Python | Ngôn ngữ lập trình |
| Scikit-learn | Xây dựng mô hình KNN |
| Pandas | Xử lý dữ liệu |
| Joblib | Lưu và tải mô hình |
| FastAPI | Xây dựng API |
| Uvicorn | Chạy máy chủ API |
| Docker | Đóng gói và triển khai |

---
