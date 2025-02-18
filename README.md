# Phân biệt Chó Mèo - Cats vs Dogs Image Classifier 🐶🐱

## 🔎 Danh Mục

- [📝 Giới Thiệu](#📝-giới-thiệu)
- [⚙️ Chức Năng Chính](#⚙️-chức-năng-chính)
- [👩‍💻 Tổng quan hệ thống](#👩‍💻-tổng-quan-hệ-thống)
- [📊 Data Sử Dụng](#📊-data-sử-dụng)
- [🛠️ Hướng dẫn cài đặt](#🛠️-hướng-dẫn-cài-đặt)
- [🎯 Chạy Hệ Thống](#🎯-chạy-hệ-thống)
- [📊 Đánh Giá Mô Hình](#📊-đánh-giá-mô-hình)
- [⚠️ Các Vấn Đề](#⚠️-các-vấn-đề)

---

## 📝 Giới Thiệu

Cats vs Dogs Image Classifier là một hệ thống phân loại hình ảnh sử dụng mạng neural học sâu (CNN) để phân biệt hình ảnh giữa chó và mèo. Dự án sử dụng mô hình học sâu đã được huấn luyện trên bộ dữ liệu lớn về hình ảnh chó và mèo, giúp phân loại hình ảnh của hai loài này một cách chính xác.

---

## ⚙️ Chức năng chính

Dự án tập trung vào các chức năng chính như sau:

- ⬆️ Tải lên hình ảnh: Người dùng có thể tải lên hình ảnh của chó hoặc mèo để nhận kết quả phân loại.
- 🎯 Dự đoán loài: Mô hình sẽ dự đoán xem hình ảnh đó là của chó hay mèo.
- 💡 Hiển thị độ tin cậy: Kết quả dự đoán sẽ đi kèm với độ tin cậy (%) nếu độ chính xác đạt 90% trở lên.

---

## 👩‍💻 Tổng quan hệ thống

Hệ thống này bao gồm hai phần chính: Backend và Frontend.

### Backend

- FastAPI: Được sử dụng để xây dựng API cho hệ thống.
- TensorFlow: Được sử dụng để xây dựng và huấn luyện mô hình phân loại hình ảnh.
- H5 (Model file): Mô hình đã được huấn luyện và lưu trữ dưới định dạng .h5 (H5 file).

### Frontend

- Streamlit: Giao diện người dùng đơn giản giúp người dùng dễ dàng tải ảnh và nhận kết quả phân loại.

---

## 📊 Data sử dụng

- [Data](https://drive.google.com/file/d/1y0ce7a_nTuxLxsC7GCfZMYpAxeSejIiQ/view?usp=drive_link)

## 🛠️ Hướng dẫn cài đặt

### Yêu Cầu 📋

Để cài đặt và chạy được dự án, bạn cần cài đặt các công cụ sau:

- Python (Phiên bản 3.8 hoặc cao hơn)
- TensorFlow (Dùng cho việc xây dựng và huấn luyện mô hình)
- Streamlit (Dùng cho việc tạo giao diện người dùng)
- Uvicorn (Dùng để chạy FastAPI server)

### 🔨 Cài Đặt

1. Clone dự án về máy tính của bạn:
```
git clone https://github.com/AE-AI-HIT15/Cats-vs-Dogs-Image-Classification.git
cd Cats-vs-Dogs-Image-Classification
```
- Cấu trúc thư mục:

```
├── data/              
│   ├── train/
│   │   ├── cats/
│   │   │   ├── cat.1.jpg
│   │   │   ├── cat.2.jpg
│   │   │   └── ...
│   │   └── dogs/
│   │       ├── dog.1.jpg
│   │       └── ...
│   ├── validation/        # Dữ liệu kiểm thử
│   │   ├── cats/
│   │   │   ├── cat.0.jpg
│   │   │   ├── cat.8.jpg
│   │   │   └── ...
│   │   └── dogs/
│   │       ├── dog.01.jpg
│   │       └── ...
│   └── test/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...  
├── models/       # Mô hình đã huấn luyện      
│   ├── cat_dog_model_1.0.0.h5
│   ├── cat_dog_model_1.1.0.h5
├── model_train_data/      # Mã nguồn của dự án
│   ├── data_preprocessing.py # tiền xử lý dữ liệu
│   ├── model_building.py # xây dựng mô hình
│   ├── training.py # huấn luyện mô hình
│   ├── evaluation.py # đánh giá mô hình
│   ├── visualization.py # vẽ biểu đồ Accuracy và Loss
│   ├── train.py # hàm sử dụng các hàm đã trên <train_model>
├── organize_train_data.py # tiền xử lý ảnh
├── main_app.py    # Mã nguồn FastAPI
├── app.py             # Mã nguồn Streamlit
├── requirements.txt       # Các thư viện yêu cầu
├── .gitignore             # Các thư mục không up lên github
└── README.md              # Tệp README
```
2. Cài đặt các thư viện cần thiết:

- Sử dụng requirements.txt để cài đặt tất cả các thư viện cần thiết.
```
pip install -r requirements.txt
```
3. Tải mô hình đã được huấn luyện:

- Tải mô hình tại [Here](https://drive.google.com/drive/folders/1nD02S5aihGwY2PrykKn4SZDv_GRHGzxf).

## 🎯 Chạy hệ thống

Để chạy API với FastAPI, sử dụng lệnh sau:
```
uvicorn fastapi.py:app --reload
```
Hệ thống sẽ chạy tại địa chỉ: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Để chạy giao diện người dùng Streamlit, sử dụng lệnh sau:
```
streamlit run app.py
```
Hệ thống sẽ mở một cửa sổ trình duyệt tại [http://localhost:8501](http://localhost:8501).

### 🎉 Kết quả

Sau khi tải lên hình ảnh của một con chó hoặc mèo, mô hình sẽ trả về kết quả dự đoán, ví dụ:

- Prediction: Dog
- Confidence: 95.25%
- Nếu độ tin cậy dưới 90%, hệ thống sẽ 

## 📊 Đánh Giá Mô Hình

- Độ chính xác trên tập huấn luyện: 95%
- Độ chính xác trên tập kiểm thử: 93%

## ⚠️ Các Vấn Đề

### Vấn Đề 1: Hiệu suất mô hình chưa đạt yêu cầu

- Mô hình có độ chính xác chưa cao đối với một số hình ảnh có độ phân giải thấp.
- Giải pháp: Tinh chỉnh mô hình hoặc sử dụng các kỹ thuật tiền xử lý ảnh khác.

### Vấn Đề 2: Không hỗ trợ nhiều định dạng ảnh

- Dự án hiện chỉ hỗ trợ định dạng .jpg và .png.
- Giải pháp: Cập nhật mã nguồn để hỗ trợ thêm các định dạng ảnh khác như .bmp và .tiff.
