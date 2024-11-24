import streamlit as st
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
import io
from datetime import datetime
import cv2
from streamlit_option_menu import option_menu

# --- Cài đặt giao diện ---
st.set_page_config(page_title="Phân Loại Bệnh Lá Lúa", layout="wide", page_icon="🌾")

# --- Tải mô hình PyTorch ---
@st.cache_resource
def load_model(model_path):
    try:
        # Khởi tạo mô hình MobileNetV2
        model = models.mobilenet_v2(pretrained=False)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.last_channel, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 8),  # Số lớp phân loại
            torch.nn.Softmax(dim=1),
        )
        # Tải trọng số đã lưu
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()  # Chuyển mô hình sang chế độ đánh giá
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop()

model_path = 'model/mobilenetv2_trained_model.pth'  # Đường dẫn đến mô hình PyTorch
if os.path.exists(model_path):
    disease_model = load_model(model_path)
else:
    st.error("Không tìm thấy mô hình phân loại!")

# --- Các nhãn bệnh ---
disease_labels = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight"
]

# --- Hàm tiền xử lý ảnh ---
def preprocess_image(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    return image

# --- Hàm kiểm tra lá lúa ---
def is_rice_leaf(image):
    # Chuyển ảnh PIL sang OpenCV
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Chuyển ảnh sang HSV để phân tích màu
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_image, (36, 25, 25), (86, 255, 255))  # Màu xanh lá cây

    # Kiểm tra tỷ lệ màu xanh trong ảnh
    green_ratio = cv2.countNonZero(green_mask) / (image.shape[0] * image.shape[1])

    # Nếu trên 50% là xanh lá, có thể coi là lá lúa
    return green_ratio > 0.5

# --- Hàm lưu ảnh ---
def save_image(image_data, disease_name):
    disease_folder = os.path.join("images", disease_name)
    os.makedirs(disease_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(disease_folder, f"{timestamp}.jpg")
    
    # Mở ảnh từ dữ liệu bytes
    image = Image.open(io.BytesIO(image_data))
    image.save(image_path)
    return image_path

# --- Tiêu đề ứng dụng ---
st.markdown("<h1 style='text-align: center; color: green;'>🌾 Phân Loại Bệnh Trên Lá Lúa 🌾</h1>", unsafe_allow_html=True)

# --- Menu chính ---
with st.sidebar:
    menu_option = option_menu(
        menu_title="Menu Chính",  # Tiêu đề menu
        options=["Tải lên ảnh", "Chụp ảnh"],  # Các tùy chọn
        icons=["cloud-upload", "camera"],  # Biểu tượng
        menu_icon="list",  # Biểu tượng menu
        default_index=0,  # Tùy chọn mặc định
        styles={
            "container": {"padding": "5px", "background-color": "#2b2b2b"},  # Nền tối
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "color": "#ffffff",  # Màu chữ
                "background-color": "#333333",  # Nền liên kết
            },
            "nav-link-hover": {"background-color": "#444444"},  # Màu khi di chuột
            "nav-link-selected": {"background-color": "#565656", "color": "#ffffff"},  # Màu khi được chọn
        },
    )

# --- Trang tải lên ảnh ---
if menu_option == "Tải lên ảnh":
    uploaded_image = st.file_uploader("Chọn ảnh lá lúa để phân loại:", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        # Kiểm tra xem ảnh có phải là lá lúa không
        if is_rice_leaf(image):
            st.markdown("<h3 style='color: green;'>✅ Đây là lá lúa, đang phân loại bệnh...</h3>", unsafe_allow_html=True)
            
            # Tiền xử lý và dự đoán
            processed_image = preprocess_image(image, target_size=(224, 224))
            with torch.no_grad():
                prediction = disease_model(processed_image)[0]

            # Tìm xác suất và nhãn bệnh có xác suất cao nhất
            max_probability = torch.max(prediction).item()
            predicted_label = disease_labels[torch.argmax(prediction).item()]

            # Kiểm tra xác suất dự đoán
            if max_probability >= 0.5:
                st.markdown("<h3 style='color: green;'>🌟 Kết quả dự đoán:</h3>", unsafe_allow_html=True)
                st.success(f"{predicted_label}: {max_probability * 100:.2f}%")

                # Lưu ảnh vào thư mục tương ứng với bệnh
                save_image(uploaded_image.getvalue(), predicted_label)
            else:
                st.warning("Không thể xác định bệnh với độ chính xác cao.")
        else:
            st.markdown("<h3 style='color: red;'>⚠️ Đây không phải là lá lúa.</h3>", unsafe_allow_html=True)
