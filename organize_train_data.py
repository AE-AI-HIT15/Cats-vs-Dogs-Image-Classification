import os
import shutil
from sklearn.model_selection import train_test_split

def organize_train_data(train_dir):
    classes = ['cats', 'dogs']
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Xác định tiền tố dựa trên tên lớp
        prefix = 'cat.' if cls == 'cats' else 'dog.'
        
        for filename in os.listdir(train_dir):
            if filename.startswith(prefix):
                src = os.path.join(train_dir, filename)
                dst = os.path.join(cls_dir, filename)
                shutil.move(src, dst)
                print(f"Đã di chuyển {filename} vào {cls_dir}")

def create_validation_set(train_dir, validation_dir, split_ratio=0.2):
    classes = ['cats', 'dogs']
    for cls in classes:
        cls_train_dir = os.path.join(train_dir, cls)
        cls_val_dir = os.path.join(validation_dir, cls)
        os.makedirs(cls_val_dir, exist_ok=True)
        
        images = os.listdir(cls_train_dir)
        train_images, val_images = train_test_split(images, test_size=split_ratio, random_state=42)
        
        for img in val_images:
            src = os.path.join(cls_train_dir, img)
            dst = os.path.join(cls_val_dir, img)
            shutil.move(src, dst)
            print(f"Đã di chuyển {img} từ {cls_train_dir} vào {cls_val_dir}")

if __name__ == "__main__":
    train_directory = 'data/train'
    validation_directory = 'data/validation'
    
    print("Bắt đầu phân loại dữ liệu trong thư mục train...")
    organize_train_data(train_directory)
    
    print("\nBắt đầu chia tập validation...")
    create_validation_set(train_directory, validation_directory, split_ratio=0.2)
    
    print("\nHoàn thành việc phân loại và chia tập dữ liệu.")