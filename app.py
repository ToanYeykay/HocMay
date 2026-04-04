import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Food Ordering Behavior Dashboard", layout="wide")

st.title("Phân Tích Hành Vi Đặt Đồ Ăn 🍔")

# Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("food_ordering_behavior_dataset.csv")
    return df

df = load_data()

# Sidebar lọc dữ liệu
city = st.sidebar.multiselect("Chọn thành phố:", options=df["city"].unique(), default=df["city"].unique())
df_selection = df[df["city"].isin(city)]

# Hiển thị chỉ số tổng quan
col1, col2, col3 = st.columns(3)
col1.metric("Tổng đơn hàng", len(df_selection))
col2.metric("Giá trị TB", f"{df_selection['order_value'].mean():.2f}")
col3.metric("Độ tuổi TB", int(df_selection['age'].mean()))

# Biểu đồ
st.subheader("Số lượng đơn hàng theo loại ẩm thực")
fig, ax = plt.subplots()
sns.countplot(data=df_selection, x='cuisine', palette='viridis', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

def main():
    st.set_page_config(page_title="Quản lý khách hàng", layout="centered")
    
    st.title("📝 Nhập thông tin khách hàng")
    st.info("Vui lòng điền đầy đủ thông tin bên dưới để bắt đầu.")

    with st.form("customer_info"):
        name = st.text_input("Họ và tên")
        phone = st.text_input("Số điện thoại")
        # Thêm các trường khác ở đây...
        
        submit = st.form_submit_button("Tiếp theo")
        
        if submit:
            if name and phone:
                # Lưu vào session_state để dùng cho các trang sau
                st.session_state['customer_name'] = name
                st.success(f"Chào mừng {name}! Bạn có thể chuyển sang trang phân tích.")
            else:
                st.error("Vui lòng nhập đầy đủ Họ tên và Số điện thoại.")

if __name__ == "__main__":
    main()