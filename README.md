# Cat vs Dog Image Classification Project

## Mô Tả
Dự án phân loại hình ảnh với mục tiêu phân loại hình ảnh giữa chó và mèo. Dự án sử dụng TensorFlow để huấn luyện mô hình học sâu và Streamlit để triển khai giao diện người dùng. Mô hình được huấn luyện trên bộ dữ liệu hình ảnh của chó và mèo, sau đó được triển khai trên giao diện web đơn giản để người dùng có thể tải lên hình ảnh và nhận kết quả phân loại.

## Công Nghệ và Công Cụ
- **Python 3.x**
- **TensorFlow 2.x**: Thư viện học sâu cho việc huấn luyện mô hình.
- **Keras**: Thư viện API để xây dựng mô hình học sâu.
- **Streamlit**: Dùng để triển khai giao diện người dùng web.
- **OpenCV**: Thư viện xử lý ảnh.
- **NumPy, Pandas**: Các thư viện xử lý dữ liệu.
- **Matplotlib, Seaborn**: Dùng để vẽ đồ thị và hình ảnh.
- **CUDA, cuDNN**: Cần thiết nếu bạn sử dụng GPU để tăng tốc quá trình huấn luyện.

## Cài Đặt
### Bước 1: Clone repository
Tải về mã nguồn dự án từ GitHub bằng lệnh:
```bash
git clone https://github.com/AE-AI-HIT15/Cats-vs-Dogs-Image-Classification.git
```
### Bước 2: Cài đặt môi trường ảo
Tạo môi trường ảo và kích hoạt nó:
```bash
python -m venv env
source env/bin/activate  # Trên macOS/Linux
.\env\Scripts\activate   # Trên Windows
```
### Bước 3: Cài đặt các thư viện yêu cầu
Cài đặt tất cả các thư viện cần thiết từ tệp requirements.txt:
```bash
pip install -r requirements.txt
```
### Bước 4: Chạy dự án
#### 1. Huấn luyện mô hình: Nếu bạn muốn huấn luyện mô hình, chạy tệp train_model.py.
```bash
python model_train_data/main.py
```
#### 2. Chạy ứng dụng Streamlit: Để triển khai giao diện web, sử dụng lệnh sau:
```bash
streamlit run app.py
```
#### 3. Kết quả phân loại: Tải lên hình ảnh (chó hoặc mèo) và nhấn "Submit" để mô hình phân loại và hiển thị kết quả.
#
# Cấu Trúc Thư Mục
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
│   ├──  cat_dog_model_1.0.0.h5
│   ├──  cat_dog_model_1.1.0.h5
├── model_train_data/      # Mã nguồn của dự án
│   ├── data_preprocessing.py
│   ├── model_building.py
│   ├── training.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├──main.py
├── organize_train_data.py # tiền xử lý ảnh
├── app.py/     # Thư mục FastAPI
│   ├──  main_app.py    # Mã nguồn FastAPI
├── streamlit_app/         # Thư mục Streamlit
│   └── app.py             # Mã nguồn Streamlit
├── requirements.txt       # Các thư viện yêu cầu
└── README.md              # Tệp README
```
# Các Phần Chính Của Dự Án
- **Huấn luyện mô hình:** Mô hình học sâu được xây dựng và huấn luyện bằng TensorFlow. Dữ liệu hình ảnh được tiền xử lý trước khi đưa vào mô hình.
- **Triển khai Streamlit:** Giao diện người dùng được tạo bằng Streamlit cho phép người dùng dễ dàng tải lên hình ảnh và nhận kết quả phân loại.
- **Tiền xử lý dữ liệu:** Dữ liệu được tiền xử lý bao gồm việc thay đổi kích thước ảnh và chuẩn hóa hình ảnh.

# Đánh Giá Mô Hình
- **Độ chính xác trên tập huấn luyện:** 95%
- **Độ chính xác trên tập kiểm thử:** 93%
# Các Vấn Đề
### **Vấn Đề 1: Hiệu suất mô hình chưa đạt yêu cầu**
- Mô hình có độ chính xác chưa cao đối với một số hình ảnh có độ phân giải thấp.
- **Giải pháp:** Tinh chỉnh mô hình hoặc sử dụng các kỹ thuật tiền xử lý ảnh khác.
### **Vấn Đề 2: Không hỗ trợ nhiều định dạng ảnh**
- Dự án hiện chỉ hỗ trợ định dạng .jpg và .png.
- Giải pháp: Cập nhật mã nguồn để hỗ trợ thêm các định dạng ảnh khác như .bmp và .tiff.
