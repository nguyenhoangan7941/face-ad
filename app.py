import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# =========================
# CẤU HÌNH APP
# =========================
st.set_page_config(
    page_title="FaceAD - Student Recognition",
    page_icon="🧑‍🎓",
    layout="wide"
)

# =========================
# ĐƯỜNG DẪN FILE
# =========================
MODEL_PATH = "my_model.h5"
LOGO_PATH = "logo.png"

# =========================
# CSS GIAO DIỆN
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #1e1b4b 100%);
        color: white;
    }

    header {
        background: transparent !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: white !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.12);
    }

    .logo-box {
        text-align: center;
        margin-bottom: 20px;
    }

    .hero-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 28px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.38);
        backdrop-filter: blur(18px);
        margin-bottom: 26px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.09);
        border-radius: 24px;
        padding: 28px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(14px);
        margin-bottom: 24px;
    }

    .main-title {
        font-size: 60px;
        font-weight: 950;
        line-height: 1.08;
        margin-bottom: 18px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .app-name {
        font-size: 30px;
        font-weight: 900;
        color: #bfdbfe !important;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 20px;
        color: #dbeafe !important;
        line-height: 1.75;
        margin-bottom: 20px;
    }

    .badge {
        display: inline-block;
        padding: 9px 18px;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.45);
        border-radius: 999px;
        color: #bae6fd !important;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .info-line {
        font-size: 17px;
        color: #e0e7ff !important;
        line-height: 1.65;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.085);
        border-radius: 22px;
        padding: 24px;
        min-height: 170px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24);
    }

    .feature-title {
        font-size: 22px;
        font-weight: 850;
        margin-bottom: 10px;
        color: #bfdbfe !important;
    }

    .feature-text {
        color: #e0e7ff !important;
        font-size: 16px;
        line-height: 1.55;
    }

    .section-title {
        font-size: 34px;
        font-weight: 900;
        margin-bottom: 12px;
        color: #f8fafc !important;
    }

    .result-box {
        padding: 32px;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(34,197,94,0.20), rgba(59,130,246,0.20), rgba(168,85,247,0.16));
        border: 1px solid rgba(125, 211, 252, 0.45);
        box-shadow: 0 20px 48px rgba(0,0,0,0.30);
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .result-label {
        font-size: 22px;
        font-weight: 800;
        color: #dbeafe !important;
    }

    .result-name {
        font-size: 46px;
        font-weight: 950;
        color: #86efac !important;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .confidence {
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc !important;
    }

    .warning-box {
        background: rgba(251, 191, 36, 0.12);
        border: 1px solid rgba(251, 191, 36, 0.35);
        padding: 20px;
        border-radius: 20px;
        color: #fef3c7 !important;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: #cbd5e1 !important;
        font-size: 15px;
        padding: 18px;
        margin-top: 20px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
        color: white !important;
        border-radius: 16px;
        padding: 0.9rem 1.4rem;
        font-weight: 900;
        border: none;
        width: 100%;
        font-size: 18px;
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.35);
        transition: 0.25s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 34px rgba(124, 58, 237, 0.45);
        color: white !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px;
        border: 1px dashed rgba(255,255,255,0.25);
    }

    div[role="radiogroup"] {
        background: rgba(255,255,255,0.08);
        padding: 14px;
        border-radius: 18px;
    }

    div[data-testid="stAlert"] {
        border-radius: 16px;
    }

    div[data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
    }

    img {
        border-radius: 20px;
        box-shadow: 0 16px 36px rgba(0,0,0,0.32);
    }

    @media screen and (max-width: 768px) {
        .main-title {
            font-size: 40px;
        }

        .subtitle {
            font-size: 17px;
        }

        .hero-card, .glass-card {
            padding: 22px;
        }

        .result-name {
            font-size: 34px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# DANH SÁCH CLASS
# =========================
class_labels = [
    "HoangKyAnh",
    "Lê Quang Dũng",
    "Lê Tuấn Thành",
    "Lương Ngọc Thuận",
    "Ngô Quốc Trung",
    "Nguyen Ngoc Bao",
    "Nguyễn Hoàng Quế Châu",
    "Nguyễn Phạm Hoàng An",
    "Nguyễn Thị Khánh Lê",
    "Nguyễn Thị Ngọc Tuyết",
    "Nguyễn Tiến Mạnh",
    "Nguyễn Việt Đức",
    "Nguyễn Đặng Vinh Phúc",
    "Phạm Gia Thành Duy",
    "Phạm Hứa Nhật Minh",
    "Phạm Nguyễn Bảo Châu",
    "Phạm Phú Hoà",
    "Trần Hải Yến",
    "Vũ Quang Thái",
    "Đinh Hữu Khánh Anh",
    "Đoàn Hùng",
    "Đỗ An Phúc"
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# =========================
# HÀM XỬ LÝ ẢNH
# =========================
IMG_SIZE = 200

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_face(model, image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index] * 100
    predicted_name = class_labels[predicted_index]

    return predicted_name, confidence, prediction

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)

    st.markdown("## FaceAD")
    st.markdown("**Student Face Recognition System**")
    st.markdown("---")

    page = st.radio(
        "📌 Điều hướng",
        ["🏠 Trang chủ", "🔍 Nhận dạng sinh viên", "ℹ️ Thông tin hệ thống"]
    )

    st.markdown("---")
    st.markdown("### 📚 Dự án")
    st.markdown("Ứng dụng nhận diện sinh viên lớp LogTech.")
    st.markdown("### 🧠 Model")
    st.markdown("CNN - TensorFlow/Keras")
    st.markdown("### 👥 Số sinh viên")
    st.markdown(f"{len(class_labels)} sinh viên")

# =========================
# TRANG CHỦ
# =========================
if page == "🏠 Trang chủ":
    col1, col2 = st.columns([1.25, 0.75])

    with col1:
        st.markdown(
            """
            <div class="hero-card">
                <div class="badge">AI Student Recognition</div>
                <div class="main-title">FaceAD</div>
                <div class="app-name">Hệ thống nhận diện sinh viên lớp LogTech </div>
                <div class="subtitle">
                    FaceAD là ứng dụng sử dụng trí tuệ nhân tạo để nhận diện khuôn mặt sinh viên
                    trong lớp LogTech. Ứng dụng hỗ trợ tải ảnh lên hoặc chụp trực tiếp
                    bằng camera, sau đó mô hình CNN sẽ phân tích và dự đoán tên sinh viên.
                </div>
                <p class="info-line">
                    Ứng dụng được xây dựng nhằm mô phỏng một hệ thống nhận diện khuôn mặt đơn giản,
                    trực quan và dễ sử dụng trong môi trường học tập.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown(
                """
                <div class="glass-card">
                    <h2>FaceAD</h2>
                    <p>Bạn hãy thêm file logo.png vào thư mục app để hiển thị logo.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### ✨ Tính năng chính")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">📁 Upload ảnh</div>
                <div class="feature-text">
                    Cho phép tải ảnh khuôn mặt sinh viên từ máy tính để hệ thống nhận diện.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">📷 Chụp camera</div>
                <div class="feature-text">
                    Hỗ trợ chụp ảnh trực tiếp bằng webcam hoặc camera thiết bị.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🧠 AI nhận diện</div>
                <div class="feature-text">
                    Mô hình CNN phân tích ảnh đầu vào và dự đoán sinh viên tương ứng.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    f4, f5, f6 = st.columns(3)

    with f4:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">📊 Xác suất dự đoán</div>
                <div class="feature-text">
                    Hiển thị độ tin cậy và xác suất dự đoán của từng sinh viên.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f5:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🎓 Dành cho lớp học</div>
                <div class="feature-text">
                    Phù hợp để minh họa bài toán nhận diện sinh viên trong lớp LogTech .
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f6:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">💻 Giao diện trực quan</div>
                <div class="feature-text">
                    Thiết kế bằng Streamlit, dễ chạy, dễ thao tác và dễ trình bày báo cáo.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="footer">
            FaceAD © Student Recognition App - LogTech 
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# TRANG NHẬN DẠNG
# =========================
elif page == "🔍 Nhận dạng sinh viên":
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">🔍 Nhận dạng sinh viên</div>
            <p>
                Tải ảnh hoặc chụp ảnh khuôn mặt sinh viên. Hệ thống sẽ xử lý ảnh và đưa ra
                kết quả dự đoán tên sinh viên thuộc lớp LogTech bán.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Không tìm thấy file my_model.h5")
        st.write("Các file hiện có trong thư mục:")
        st.write(os.listdir())
        st.stop()

    try:
        with st.spinner("⏳ Đang tải model FaceAD..."):
            model = load_my_model()

        output_classes = model.output_shape[-1]

        if len(class_labels) != output_classes:
            st.error("❌ Số lượng class_labels không khớp với model!")
            st.write(f"Model đang có số class: {output_classes}")
            st.write(f"Đang khai báo số class_labels: {len(class_labels)}")
            st.stop()

    except Exception as e:
        st.error("❌ Lỗi khi tải model!")
        st.code(str(e))
        st.stop()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown(
            """
            <div class="glass-card">
                <h2>📸 Chọn ảnh đầu vào</h2>
                <p>Ảnh nên rõ mặt, đủ sáng, không bị che khuất và chỉ nên có một khuôn mặt chính.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        option = st.radio(
            "Chọn cách đưa ảnh vào:",
            ["📁 Upload ảnh", "📷 Chụp bằng camera"]
        )

        image = None

        if option == "📁 Upload ảnh":
            uploaded_file = st.file_uploader(
                "Tải ảnh khuôn mặt sinh viên lên",
                type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)

        elif option == "📷 Chụp bằng camera":
            camera_file = st.camera_input("Chụp ảnh khuôn mặt sinh viên")

            if camera_file is not None:
                image = Image.open(camera_file)

        st.markdown(
            """
            <div class="warning-box">
                <b>Lưu ý:</b> Ứng dụng chỉ nhận diện tốt những sinh viên đã có trong dữ liệu huấn luyện.
                Nếu ảnh quá mờ, thiếu sáng hoặc khác nhiều so với dữ liệu train, kết quả có thể không chính xác.
            </div>
            """,
            unsafe_allow_html=True
        )

    with right_col:
        st.markdown(
            """
            <div class="glass-card">
                <h2>🖼️ Ảnh xem trước</h2>
                <p>Ảnh được chọn sẽ hiển thị tại đây trước khi nhận diện.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if image is not None:
            st.image(
                image,
                caption="Ảnh sinh viên đã chọn",
                use_container_width=True
            )
        else:
            st.info("📌 Chưa có ảnh. Hãy upload hoặc chụp ảnh để bắt đầu.")

    if image is not None:
        st.markdown("---")

        if st.button("🔍 Nhận diện sinh viên"):
            try:
                with st.spinner("🧠 FaceAD đang phân tích khuôn mặt..."):
                    predicted_name, confidence, prediction = predict_face(model, image)

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">✅ Sinh viên được nhận diện</div>
                        <div class="result-name">{predicted_name}</div>
                        <div class="confidence">Độ tin cậy: {confidence:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if confidence < 60:
                    st.warning(
                        "⚠️ Độ tin cậy chưa cao. Bạn nên thử ảnh rõ hơn, đủ sáng hơn hoặc chụp gần khuôn mặt hơn."
                    )

                st.markdown(
                    """
                    <div class="glass-card">
                        <h2>📊 Bảng xác suất dự đoán</h2>
                        <p>Danh sách bên dưới được sắp xếp từ xác suất cao xuống thấp.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                probs = prediction[0]
                sorted_indices = np.argsort(probs)[::-1]

                for index in sorted_indices:
                    percent = probs[index] * 100
                    st.write(f"**{class_labels[index]}:** {percent:.2f}%")
                    st.progress(float(probs[index]))

            except Exception as e:
                st.error("❌ Lỗi khi dự đoán ảnh!")
                st.code(str(e))

# =========================
# TRANG THÔNG TIN HỆ THỐNG
# =========================
elif page == "ℹ️ Thông tin hệ thống":
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">ℹ️ Thông tin hệ thống</div>
            <p>
                Trang này mô tả tổng quan về ứng dụng FaceAD và mô hình nhận diện khuôn mặt sinh viên.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <h2>📌 Thông tin ứng dụng</h2>
                <p><b>Tên ứng dụng:</b> FaceAD</p>
                <p><b>Chức năng:</b> Nhận diện sinh viên lớp LogTech </p>
                <p><b>Nền tảng:</b> Streamlit</p>
                <p><b>Ngôn ngữ:</b> Python</p>
                <p><b>Số sinh viên:</b> {len(class_labels)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="glass-card">
                <h2>🧠 Thông tin mô hình</h2>
                <p><b>File model:</b> my_model.h5</p>
                <p><b>Kiểu mô hình:</b> CNN</p>
                <p><b>Framework:</b> TensorFlow/Keras</p>
                <p><b>Kích thước ảnh đầu vào:</b> 200 x 200 x 3</p>
                <p><b>Đầu ra:</b> Xác suất thuộc từng sinh viên</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h2>🎯 Mục tiêu</h2>
                <p>
                    Ứng dụng FaceAD được xây dựng nhằm hỗ trợ nhận diện khuôn mặt sinh viên
                    trong lớp học thông qua ảnh đầu vào. Đây là sản phẩm minh họa cho việc ứng dụng
                    trí tuệ nhân tạo vào bài toán phân loại hình ảnh.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="glass-card">
                <h2>⚠️ Giới hạn</h2>
                <p>
                    Hệ thống chỉ nhận diện các sinh viên đã được đưa vào dữ liệu huấn luyện.
                    Độ chính xác phụ thuộc vào chất lượng ảnh, ánh sáng, góc chụp và số lượng ảnh train.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 👥 Danh sách sinh viên trong hệ thống")

    for i, name in enumerate(class_labels, start=1):
        st.write(f"{i}. {name}")