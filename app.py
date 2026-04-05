import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import xgboost as xgb

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Dự báo Food Delivery - Lê Tấn Toàn", layout="wide")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_data
def load_data():
    # Đọc file dữ liệu gốc
    df = pd.read_csv('food_ordering_behavior_dataset.csv')
    
    # Tiền xử lý mood cho EDA
    mood_map = {'Celebrating': 4, 'Happy': 3, 'Lazy': 2, 'Stressed': 1}
    df['mood_score'] = df['mood'].map(mood_map)
    
    # Tạo bảng gộp theo User ID
    user_df = df.groupby('user_id').agg({
        'order_value': 'sum',
        'rating_given': 'mean',
        'mood_score': 'mean',
        'order_id': 'count'
    }).reset_index()
    user_df.columns = ['user_id', 'total_spent', 'avg_rating', 'avg_mood', 'order_count']
    return df, user_df

try:
    df, user_df = load_data()
except Exception as e:
    st.error("⚠️ Lỗi: Không tìm thấy file 'food_ordering_behavior_dataset.csv'!")
    st.stop()

# --- 3. SIDEBAR ĐIỀU HƯỚNG ---
st.sidebar.title("📌 Menu Quản Lý")
page = st.sidebar.radio("Chọn trang:", ["Trang 1: Giới thiệu & EDA", "Trang 2: Hệ thống Dự báo AI"])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & EDA
# ---------------------------------------------------------
if page == "Trang 1: Giới thiệu & EDA":
    st.title("📊 Khám Phá Dữ Liệu Hành Vi Đặt Hàng")
    
    # Thông tin SV bắt buộc
    st.success(f"""
    **Tên đề tài:** Dự báo Rating và Tỉ lệ quay lại của khách hàng Food Delivery  
    **Sinh viên:** Lê Tấn Toàn  
    **MSSV:** 22T1020768
    """)

    st.markdown("""
    ### 🎯 Giá trị thực tiễn
    Ứng dụng giúp doanh nghiệp nhận diện trải nghiệm khách hàng ngay lập tức. Thông qua việc dự báo **Rating** và **Khả năng giữ chân**, 
    nhà quản lý có thể đưa ra các mã giảm giá hoặc ưu tiên giao hàng để cải thiện lòng trung thành của khách.
    """)

    st.divider()

    # Nội dung kỹ thuật
    st.subheader("1. Trích xuất dữ liệu thô")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("2. Phân tích trực quan")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Phân bổ mức độ hài lòng (Rating)**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='rating_given', data=df, palette='viridis', ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.write("**Ma trận tương quan giữa các đặc trưng**")
        fig2, ax2 = plt.subplots()
        sns.heatmap(user_df[['total_spent', 'avg_rating', 'avg_mood', 'order_count']].corr(), 
                    annot=True, cmap='RdBu', ax=ax2)
        st.pyplot(fig2)

    st.write("### 📝 Nhận xét dữ liệu")
    st.info("""
    - Dữ liệu cho thấy Rating tập trung nhiều ở mức 3-4 sao.
    - **Tâm trạng (Mood)** có ảnh hưởng lớn nhất đến kết quả đánh giá cuối cùng.
    - Các đặc trưng như Chi tiêu và Số lượng đơn hàng là tiền đề quan trọng để dự báo tỉ lệ quay lại.
    """)

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH (GỘP TỔNG HỢP)
# ---------------------------------------------------------
else:
    st.title("🤖 Hệ Thống Dự Báo Thông Minh")

    # Tải mô hình
    try:
        model_rating = joblib.load('model_xgb.pkl')
        model_repeat = joblib.load('model_repeat.pkl')
    except:
        st.warning("⚠️ Thiếu file model_xgb.pkl hoặc model_repeat.pkl!")
        st.stop()

    st.subheader("📝 Nhập thông tin đơn hàng mới")
    
    # Giao diện nhập liệu
    with st.form("main_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            u_id = st.text_input("User ID:", "USER_22T")
            order_val = st.number_input("Giá trị đơn (VNĐ):", 50, 5000, 500)
        with c2:
            mood_label = st.selectbox("Tâm trạng khách:", ['Stressed', 'Lazy', 'Happy', 'Celebrating'])
            time_order = st.number_input("Thời gian giao (phút):", 5, 120, 30)
        with c3:
            city = st.selectbox("Thành phố:", df['city'].unique())
            weather = st.radio("Thời tiết mưa?", ["No", "Yes"], horizontal=True)
        
        btn = st.form_submit_button("🚀 Thực hiện phân tích")

    if btn:
        # --- BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU ---
        mood_map = {'Stressed': 1.0, 'Lazy': 2.0, 'Happy': 3.0, 'Celebrating': 4.0}
        current_mood = float(mood_map[mood_label])
        current_val = float(order_val)

        # --- BƯỚC 2: DỰ ĐOÁN RATING ---
        # Chú ý: Tên cột phải khớp 100% với lúc train ở Colab
        input_r = pd.DataFrame(
            [[current_val, current_mood, 1.0]], 
            columns=['total_spent', 'avg_mood', 'order_count']
        )
        
        pred_r = float(model_rating.predict(input_r)[0])
        pred_r = max(1.0, min(5.0, pred_r))

        # --- BƯỚC 3: DỰ ĐOÁN % QUAY LẠI ---
        # Dùng kết quả pred_r vừa có để làm đầu vào
        input_rep = pd.DataFrame(
            [[current_val, pred_r, current_mood]], 
            columns=['total_spent', 'avg_rating', 'avg_mood']
        )
        
        # Lấy xác suất từ Classifier
        proba_raw = model_repeat.predict_proba(input_rep)[0][1]
        
        # Logic Baseline 50% + Buff nhạy cảm
        sensitivity = 1.1 # Tăng độ nhạy để con số thay đổi rõ rệt hơn
        proba_final = 50 + (float(proba_raw) - 0.5) * 100 * sensitivity
        proba_final = max(10.0, min(97.0, proba_final))

        # --- HIỂN THỊ KẾ QUẢ ---
        st.divider()
        st.subheader("📈 Kết quả phân tích từ AI")
        res1, res2 = st.columns(2)

        with res1:
            st.metric("⭐ Rating dự báo", f"{pred_r:.2f} / 5.0")
            p_val = float(max(0.0, min(pred_r/5, 1.0)))
            st.progress(p_val)
            if pred_r >= 4: st.success("Dự báo: Khách hàng rất hài lòng!")
            elif pred_r >= 3: st.warning("Dự báo: Trải nghiệm mức trung bình.")
            else: st.error("Dự báo: Nguy cơ đánh giá thấp!")

        with res2:
            st.metric("🔁 Xác suất quay lại", f"{proba_final:.1f}%")
            st.progress(float(proba_final/100))
            if proba_final > 50:
                st.info(f"Tỉ lệ giữ chân tốt (+{(proba_final-50):.1f}%)")
            else:
                st.warning(f"Cần thêm ưu đãi giữ chân (-{(50-proba_final):.1f}%)")
        
        st.info(f"**Lời khuyên:** Đơn hàng của khách {u_id} nên được giao đúng hạn {time_order} phút để duy trì chỉ số dự báo trên.")