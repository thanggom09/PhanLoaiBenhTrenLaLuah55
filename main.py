import streamlit as st
import numpy as np
from PIL import Image
import os
import io
from datetime import datetime
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_option_menu import option_menu

# --- Cài đặt giao diện ---
st.set_page_config(page_title="Phân Loại Bệnh Lá Lúa", layout="wide", page_icon="🌾")

# --- Tải mô hình ---
@st.cache_resource
def load_keras_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop()

model_path = 'model/Trung_Model.h5'
if os.path.exists(model_path):
    disease_model = load_keras_model(model_path)
else:
    st.error("Không tìm thấy mô hình!")

# --- Các nhãn và biện pháp ---
disease_labels = [
    "Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf", "Leaf Blast", 
    "Leaf Scald", "Narrow Brown Leaf Spot", "Rice Hispa", "Sheath Blight"
]
disease_remedies = {
    "Bacterial Leaf Blight": "Sử dụng các giống kháng bệnh và phun thuốc gốc đồng.",
    "Brown Spot": "Phun Mancozeb hoặc Carbendazim.",
    "Healthy Rice Leaf": "Không cần xử lý.",
    "Leaf Blast": "Dùng Tricyclazole để phòng bệnh.",
    "Leaf Scald": "Giảm phân đạm, dùng thuốc bảo vệ thực vật.",
    "Narrow Brown Leaf Spot": "Phun Mancozeb hoặc Zineb.",
    "Rice Hispa": "Dùng thuốc trừ sâu chứa Chlorpyrifos.",
    "Sheath Blight": "Phun Validamycin hoặc Hexaconazole."
}

# --- Hàm tiền xử lý ảnh ---
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# --- Hàm kiểm tra lá lúa ---
def is_rice_leaf(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_image, (36, 25, 25), (86, 255, 255))
    green_ratio = cv2.countNonZero(green_mask) / (image.shape[0] * image.shape[1])
    return green_ratio > 0.5

# --- Hàm lưu ảnh ---
def save_image(image_data, save_folder="images", disease_name=None):
    folder_path = os.path.join(save_folder, disease_name if disease_name else "Uncategorized")
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(folder_path, f"{timestamp}.jpg")
    image = Image.open(io.BytesIO(image_data))
    image.save(image_path)
    return image_path

# --- Hàm lấy danh sách thư mục ---
def list_folders(base_folder):
    return [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

# --- Hàm lấy danh sách ảnh ---
def list_images_in_folder(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# --- Tiêu đề ---
st.markdown("<h1 style='text-align: center; color: green;'>🌾 Phân Loại Bệnh Trên Lá Lúa 🌾</h1>", unsafe_allow_html=True)

# --- Menu ---
with st.sidebar:
    menu_option = option_menu(
        menu_title="Menu Chính",
        options=["Tải lên ảnh", "Xem ảnh đã lưu"],
        icons=["cloud-upload", "folder-open"],
        menu_icon="list",
        default_index=0,
    )

# --- Tải ảnh ---
if menu_option == "Tải lên ảnh":
    uploaded_image = st.file_uploader("Chọn ảnh lá lúa để phân loại:", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        if is_rice_leaf(image):
            st.success("✅ Đây là lá lúa. Đang tiến hành phân loại...")
            processed_image = preprocess_image(image)
            prediction = disease_model.predict(processed_image)[0]
            max_probability = np.max(prediction)
            predicted_label = disease_labels[np.argmax(prediction)]

            if max_probability >= 0.5:
                st.success(f"🌟 Kết quả dự đoán: **{predicted_label}** ({max_probability * 100:.2f}%)")
                remedy = disease_remedies.get(predicted_label, "Không có thông tin.")
                st.info(f"💡 Biện pháp khắc phục: {remedy}")

                if st.button("Lưu ảnh"):
                    save_path = save_image(uploaded_image.getvalue(), disease_name=predicted_label)
                    st.success(f"Ảnh đã lưu vào: {save_path}")
            else:
                st.warning("Không thể xác định bệnh với độ chính xác cao.")
        else:
            st.error("⚠️ Ảnh này không phải là lá lúa.")

# --- Xem ảnh đã lưu ---
if menu_option == "Xem ảnh đã lưu":
    base_folder = "images"
    if not os.path.exists(base_folder):
        st.warning("Chưa có ảnh nào được lưu.")
    else:
        folders = list_folders(base_folder)
        selected_folder = st.selectbox("Chọn thư mục:", folders)
        if selected_folder:
            folder_path = os.path.join(base_folder, selected_folder)
            images = list_images_in_folder(folder_path)
            if images:
                st.markdown(f"### Ảnh trong thư mục **{selected_folder}**")
                cols = st.columns(3)
                for i, image_name in enumerate(images):
                    image_path = os.path.join(folder_path, image_name)
                    image = Image.open(image_path)
                    cols[i % 3].image(image, caption=image_name, use_column_width=True)
            else:
                st.info("Thư mục này không có ảnh.")
