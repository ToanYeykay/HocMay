import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import xgboost as xgb

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự báo Food Delivery - Lê Tấn Toàn", layout="wide")

# --- HÀM LOAD DỮ LIỆU ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('food_ordering_behavior_dataset.csv')
    mood_map = {'Celebrating': 4, 'Happy': 3, 'Lazy': 2, 'Stressed': 1}
    df['mood_score'] = df['mood'].map(mood_map)
    # Bảng user_df để lấy thống kê chung cho EDA
    user_df = df.groupby('user_id').agg({
        'order_value': 'sum',
        'rating_given': 'mean',
        'mood_score': 'mean',
        'order_id': 'count'
    }).reset_index()
    user_df.columns = ['user_id', 'total_spent', 'avg_rating', 'avg_mood', 'order_count']
    return df, user_df

try:
    df, user_df = load_and_prep_data()
except:
    st.error("Không tìm thấy file dữ liệu CSV!")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("Menu")
page = st.sidebar.radio("Chọn trang:", ["Giới thiệu & EDA", "Hệ thống Dự báo Tổng hợp"])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU
# ---------------------------------------------------------
if page == "Giới thiệu & EDA":
    st.title("Khám Phá Dữ Liệu Hành Vi Khách Hàng")
    st.info(f"**Sinh viên:** Lê Tấn Toàn | **MSSV:** 22T1020768")
    
    st.markdown("### Giá trị thực tiễn\nHệ thống giúp quản lý nhà hàng dự đoán độ hài lòng và tỉ lệ giữ chân khách hàng ngay khi đơn hàng được tạo.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Phân bổ Rating**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='rating_given', data=df, palette='viridis', ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.write("**Tương quan các chỉ số**")
        fig2, ax2 = plt.subplots()
        sns.heatmap(user_df[['total_spent', 'avg_rating', 'avg_mood', 'order_count']].corr(), 
                    annot=True, cmap='RdBu', ax=ax2)
        st.pyplot(fig2)

# ---------------------------------------------------------
# TRANG 2
# ---------------------------------------------------------
else:
    st.title("Dự Báo Toàn Diện Đơn Hàng")

    # Load 2 mô hình
    try:
        model_rating = joblib.load('model_xgb.pkl')
        model_repeat = joblib.load('model_repeat.pkl')
    except:
        st.warning("⚠️ Thiếu file .pkl (model_xgb.pkl hoặc model_repeat.pkl)")
        st.stop()

    st.subheader("📝 Nhập thông tin đơn hàng")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            u_id = st.text_input("User ID:", "1001")
            order_val = st.number_input("Giá trị đơn hàng:", 50, 5000, 500)
        with c2:
            mood_label = st.selectbox("Tâm trạng hiện tại:", ['Celebrating', 'Stressed', 'Lazy', 'Happy'])
            time_order = st.number_input("Thời gian giao dự kiến:", 5, 120, 30)
        with c3:
            city = st.selectbox("Thành phố:", df['city'].unique())
            weather = st.radio("Thời tiết mưa?", ["No", "Yes"], horizontal=True)

    st.divider()

    if st.button("Thực hiện phân tích tổng hợp"):
        # Chuyển đổi mood sang số
        mood_map = {'Stressed': 1, 'Lazy': 2, 'Happy': 3, 'Celebrating': 4}
        current_mood = mood_map[mood_label]

        # --- PHẦN 1: DỰ ĐOÁN RATING ---
        # Input: [total_spent, avg_mood, order_count] (Giả định khách đã từng đặt 1 đơn)
        input_r = pd.DataFrame([[order_val, current_mood, 1]], 
                              columns=['total_spent', 'avg_mood', 'order_count'])
        
        pred_rating = model_rating.predict(input_r)[0]
        pred_rating = max(1.0, min(5.0, pred_rating))

        # --- PHẦN 2: DỰ ĐOÁN % QUAY LẠI (Dựa trên Rating vừa dự đoán) ---
        # Input: [total_spent, avg_rating, avg_mood]
        input_rep = pd.DataFrame([[order_val, pred_rating, current_mood]], 
                                columns=['total_spent', 'avg_rating', 'avg_mood'])
        
        # Xác suất gốc từ XGBoost
        proba_raw = model_repeat.predict_proba(input_rep)[0][1]
        
        # Công thức Baseline 50% "Buff" cho Toàn
        sensitivity = 0.8
        proba_final = 50 + (proba_raw - 0.5) * 100 * sensitivity
        proba_final = max(5.0, min(98.5, proba_final))

        # --- HIỂN THỊ KẾ QUẢ SONG SONG ---
        st.subheader("Kết quả Phân tích từ AI")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric("Rating dự báo", f"{pred_rating:.1f} / 5.0")
            p_val_r = float(max(0.0, min(pred_rating/5, 1.0)))
            st.progress(p_val_r)
            if pred_rating >= 4: st.success("Dự báo: Khách hàng sẽ rất hài lòng!")
            elif pred_rating >= 3: st.warning("Dự báo: Trải nghiệm trung bình.")
            else: st.error("Dự báo: Rủi ro nhận đánh giá thấp!")

        with res_col2:
            st.metric("Tỉ lệ quay lại", f"{proba_final:.1f}%")
            st.progress(float(proba_final/100))
            if proba_final > 50:
                st.info(f"Tỉ lệ giữ chân tích cực (+{(proba_final-50):.1f}%)")
            else:
                st.warning(f"Cần thêm ưu đãi để giữ chân (-{(50-proba_final):.1f}%)")

        st.divider()
        st.write("🔍 **Phân tích chi tiết:** Với mức chi tiêu và tâm trạng hiện tại, hệ thống đánh giá đây là một đơn hàng có chỉ số vận hành ổn định. Đề xuất: Tiếp tục duy trì chất lượng giao hàng dưới 30 phút để bảo đảm Rating dự báo.")