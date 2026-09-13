# 🤖 Dự đoán khả năng mua ô tô bằng thuật toán KNN

Dự án xây dựng hệ thống dự đoán khả năng mua ô tô của khách hàng
bằng thuật toán K-Nearest Neighbors (KNN).

## 📌 Thông tin đề tài

- **Môn học:** Học máy cơ bản
- **Thuật toán:** K-Nearest Neighbors (KNN)
- **Bài toán:** Phân loại nhị phân
- **Đối tượng:** Khách hàng mua ô tô

## 🎯 Mục tiêu

Xây dựng mô hình KNN để dự đoán khách hàng có khả năng mua
ô tô hay không dựa trên các thông tin:

- Tuổi
- Thu nhập
- Số lần tìm hiểu xe
- Số lần đến showroom

## 🧠 Cơ chế hoạt động

KNN dự đoán khách hàng mới bằng cách:

1. Tính khoảng cách với các khách hàng trong dữ liệu.
2. Tìm K khách hàng gần nhất.
3. Xác định nhóm của các khách hàng gần nhất.
4. Bỏ phiếu để đưa ra kết quả dự đoán.

## 📊 Dữ liệu đầu vào

| Đặc trưng | Ý nghĩa |
|---|---|
| Tuổi | Tuổi khách hàng |
| Thu nhập | Thu nhập hàng tháng |
| Tìm hiểu xe | Số lần tìm hiểu xe |
| Showroom | Số lần đến showroom |

## 🔮 Kết quả dự đoán

- `0`: Không có khả năng mua
- `1`: Có khả năng mua

## 🚀 Chạy dự án

### Cài đặt thư viện

```bash
pip install -r requirements.txt
Huấn luyện mô hình
python training/train.py
Chạy API
uvicorn app.main:app --reload --port 3000
🌐 API
Kiểm tra trạng thái
GET /health
Dự đoán
POST /api/v1/predict
Tài liệu API
GET /docs
