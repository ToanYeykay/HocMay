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