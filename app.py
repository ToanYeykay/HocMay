import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Thống Phân Tích Foodie AI", layout="wide")

# --- 2. HÀM TẢI TÀI NGUYÊN (CÓ CACHE) ---
@st.cache_resource
def load_prediction_model():
    """Tải mô hình XGBoost và danh sách cột đã train"""
    try:
        model = xgb.XGBRegressor()
        model.load_model('food_rating_model.json')
        with open('model_columns.json', 'r') as f:
            model_cols = json.load(f)
        return model, model_cols
    except Exception as e:
        st.error(f"❌ Không thể tải mô hình: {e}")
        return None, None

@st.cache_data
def load_analysis_data():
    """Tải dữ liệu 50.000 dòng để phân tích"""
    file_path = 'food_ordering_behavior_dataset3.csv'
    if not os.path.exists(file_path):
        st.error(f"❌ Không tìm thấy file {file_path}. Vui lòng kiểm tra lại trên GitHub!")
        return None
    
    df = pd.read_csv(file_path)
    # Dọn dẹp khoảng trắng tên cột nếu có
    df.columns = df.columns.str.strip()
    return df

# --- 3. TRANG NHẬP LIỆU & DỰ ĐOÁN ---
def prediction_page():
    st.title("🚀 Dự Đoán Rating Khách Hàng")
    st.markdown("Nhập thông tin đơn hàng để AI dự đoán mức độ hài lòng của khách.")

    model, model_cols = load_prediction_model()
    if model is None: return

    with st.form("prediction_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            user_id = st.text_input("Mã khách hàng", "USR-999")
            age = st.number_input("Tuổi", 10, 100, 25)
            city = st.selectbox("Thành phố", ["Pune", "Mumbai", "Delhi", "Bangalore", "Hyderabad"])
            cuisine = st.selectbox("Loại món ăn", ["Chinese", "South Indian", "Biryani", "North Indian", "Fast Food", "Desserts"])
        
        with c2:
            order_val = st.number_input("Giá trị món (VNĐ)", 0, 1000000, 50000, step=5000)
            delivery = st.number_input("Phí vận chuyển (VNĐ)", 0, 100000, 15000, step=1000)
            time_period = st.selectbox("Buổi trong ngày", ["Morning", "Afternoon", "Evening", "Night"])
            discount = st.selectbox("Áp dụng giảm giá", [1, 0], format_func=lambda x: "Có" if x==1 else "Không")

        with c3:
            mood = st.selectbox("Tâm trạng khách", ["Happy", "Stressed", "Lazy", "Celebrating"])
            hunger = st.selectbox("Mức độ đói", ["Low", "Medium", "High"])
            rain = st.selectbox("Thời tiết mưa", [1, 0], format_func=lambda x: "Đang mưa" if x==1 else "Không mưa")
            rank = st.selectbox("Hạng khách hàng", ["Diamond", "Gold", "Silver", "Bronze"])

        submit_btn = st.form_submit_button("🔥 Phân tích & Dự đoán")

        if submit_btn:
            # Tạo DataFrame 1 dòng với toàn bộ cột của mô hình (tất cả là 0)
            input_data = pd.DataFrame(0, index=[0], columns=model_cols)
            
            # Điền các cột số
            input_data['age'] = age
            input_data['order_value'] = order_val
            input_data['delivery_fee'] = delivery
            input_data['discount_applied'] = discount
            input_data['rainy_weather'] = rain
            input_data['net_value'] = order_val + delivery
            
            # Điền các cột One-Hot (Gán giá trị 1)
            categorical_features = {
                'city': city, 'order_time': time_period, 'cuisine': cuisine,
                'mood': mood, 'hunger_level': hunger, 'rank': rank
            }
            
            for feat, val in categorical_features.items():
                col_name = f"{feat}_{val}"
                if col_name in input_data.columns:
                    input_data[col_name] = 1

            # Thực hiện dự đoán
            pred_rating = model.predict(input_data)[0]
            
            # Hiển thị kết quả
            st.divider()
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Rating dự báo", f"{pred_rating:.2f} / 5.0 ⭐")
            
            if pred_rating >= 4.0:
                st.balloons()
                st.success("Khách hàng này có khả năng cao sẽ rất hài lòng!")
            elif pred_rating <= 2.5:
                st.warning("Cảnh báo: Đơn hàng này có nguy cơ nhận đánh giá thấp.")

# --- 4. TRANG PHÂN TÍCH DASHBOARD ---
def analysis_page():
    st.title("📊 Dashboard Phân Tích Hệ Sinh Thái")
    df = load_analysis_data()
    if df is None: return

    # Các chỉ số Metric chính
    st.markdown("### 📈 Chỉ số tổng quan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng đơn hàng", f"{len(df):,}")
    m2.metric("Rating trung bình", f"{df['rating_given'].mean():.2f} ⭐")
    m3.metric("Khách hàng duy nhất", f"{df['user_id'].nunique():,}")
    m4.metric("Doanh thu TB/Đơn", f"{int(df['order_value'].mean()):,}đ")

    st.divider()

    # Biểu đồ phân tích
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🏆 Cơ cấu hạng thành viên")
        if 'rank' in df.columns:
            rank_data = df['rank'].value_counts()
            st.bar_chart(rank_data)
        else:
            st.warning("Không tìm thấy dữ liệu 'rank' để vẽ biểu đồ.")

    with col_right:
        st.subheader("🚚 Phí Ship vs Mức Độ Hài Lòng")
        fig, ax = plt.subplots(figsize=(8, 5))
        # Lấy mẫu 1000 dòng để vẽ nhanh hơn
        sns.regplot(data=df.sample(1000), x='delivery_fee', y='rating_given', 
                    scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax)
        ax.set_title("Tương quan Phí vận chuyển & Rating")
        st.pyplot(fig)

    st.divider()
    
    # Bảng dữ liệu chi tiết
    with st.expander("🔍 Xem chi tiết 100 dòng dữ liệu mới nhất"):
        st.dataframe(df.tail(100), use_container_width=True)

# --- 5. ĐIỀU HƯỚNG CHÍNH ---
def main():
    st.sidebar.title("🛠️ Điều hướng")
    page = st.sidebar.radio("Chọn chức năng:", ["Dự đoán Rating", "Phân tích Dữ liệu AI"])

    if page == "Dự đoán Rating":
        prediction_page()
    else:
        analysis_page()

if __name__ == "__main__":
    main()