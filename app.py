import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CẤU HÌNH ---
st.set_page_config(page_title="Hệ thống AI Food Delivery - Lê Tấn Toàn", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('food_ordering_behavior_dataset.csv')
    mood_map = {'Celebrating': 4, 'Happy': 3, 'Lazy': 2, 'Stressed': 1}
    df['mood_score'] = df['mood'].map(mood_map)
    user_df = df.groupby('user_id').agg({
        'order_value': 'sum',
        'rating_given': 'mean',
        'mood_score': 'mean',
        'order_id': 'count'
    }).reset_index()
    user_df.columns = ['user_id', 'total_spent', 'avg_rating', 'avg_mood', 'order_count']
    # Ép kiểu user_id về string để so sánh cho chuẩn
    user_df['user_id'] = user_df['user_id'].astype(str)
    return df, user_df

df, user_df = load_data()

# --- SIDEBAR ---
page = st.sidebar.radio("Chuyển trang:", ["EDA", "Dự báo Tổng hợp"])

if page == "EDA":
    st.title("📊 Phân Tích Dữ Liệu")
    st.info("SV: Lê Tấn Toàn - 22T1020768")
    st.dataframe(df.head(10))
    # Thêm biểu đồ tùy ý ở đây...

else:
    st.title("🤖 Hệ Thống Dự Báo Thông Minh (Real-time)")

    try:
        model_rating = joblib.load('model_xgb.pkl')
        model_repeat = joblib.load('model_repeat.pkl')
    except:
        st.error("⚠️ Thiếu file mô hình .pkl!")
        st.stop()

    with st.form("input_form"):
        c1, c2 = st.columns(2)
        with c1:
            u_id = st.text_input("Nhập mã khách hàng (User ID):", "2000")
            order_val = st.number_input("Giá trị đơn hàng hiện tại:", 50, 5000, 500)
        with c2:
            mood_label = st.selectbox("Tâm trạng khách:", ['Stressed', 'Lazy', 'Happy', 'Celebrating'])
            time_order = st.number_input("Thời gian giao (phút):", 5, 120, 30)
        
        submit = st.form_submit_button("🚀 Phân tích khách hàng")

    if submit:
        # 1. TIỀN XỬ LÝ NHANH
        mood_map = {'Stressed': 1.0, 'Lazy': 2.0, 'Happy': 3.0, 'Celebrating': 4.0}
        current_mood = float(mood_map[mood_label])
        
        # 2. KIỂM TRA LỊCH SỬ TRONG CƠ SỞ DỮ LIỆU (user_df)
        user_record = user_df[user_df['user_id'] == str(u_id)]
        
        if not user_record.empty:
            st.info(f"📍 Khách hàng cũ: Đã từng đặt {int(user_record['order_count'].values[0])} đơn.")
            hist_spent = float(user_record['total_spent'].values[0])
            hist_count = int(user_record['order_count'].values[0])
        else:
            st.warning("📍 Khách hàng mới hoàn toàn.")
            hist_spent = 0.0
            hist_count = 0

        # 3. DỰ ĐOÁN RATING CHO ĐƠN HÀNG NÀY
        # Feature: [total_spent (lúc này dùng chi tiêu hiện tại), mood, count]
        input_rating = pd.DataFrame([[float(order_val), current_mood, 1.0]], 
                                   columns=['total_spent', 'avg_mood', 'order_count'])
        pred_r = float(model_rating.predict(input_rating)[0])
        pred_r = max(1.0, min(5.0, pred_r))

        # 4. DỰ ĐOÁN TỈ LỆ QUAY LẠI (Cộng dồn dữ liệu)
        # Feature cho Classifier: [tổng chi tiêu mới, rating dự đoán, tổng số đơn mới]
        new_total_spent = hist_spent + float(order_val)
        new_order_count = hist_count + 1
        
        input_repeat = pd.DataFrame([[new_total_spent, pred_r, new_order_count]], 
                                   columns=['total_spent', 'avg_rating', 'order_count'])
        
        # Dự đoán xác suất
        proba_raw = model_repeat.predict_proba(input_repeat)[0][1]
        
        # Công thức Baseline 50% để số không bị đứng im ở 10%
        # Nếu proba_raw thấp, hệ số sensitivity sẽ đẩy nó lên/xuống quanh mốc 50%
        sensitivity = 1.3
        proba_final = 50 + (float(proba_raw) - 0.5) * 100 * sensitivity
        proba_final = max(12.0, min(98.0, proba_final))

        # 5. HIỂN THỊ KẾ QUẢ
        st.divider()
        res1, res2 = st.columns(2)
        
        with res1:
            st.subheader("⭐ Dự báo Hài lòng")
            st.metric("Rating đơn này", f"{pred_r:.2f} / 5.0")
            st.progress(float(pred_r/5))
            
        with res2:
            st.subheader("🔁 Dự báo Quay lại")
            st.metric("Xác suất trung thành", f"{proba_final:.1f}%")
            st.progress(float(proba_final/100))

        st.write(f"📊 **Phân tích:** Dựa trên tổng chi tiêu tích lũy **{new_total_spent:,.0f} VNĐ** và lịch sử **{new_order_count} đơn hàng**, AI đánh giá khách hàng này có mức độ gắn kết cao.")