from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายข้อมูลโรคหัวใจวายด้วยเทคนิค K-Nearest Neighbor')

col1, col2 = st.columns(2)

with col1:
    st.header("")
    st.image("https://i.ibb.co/6b04LzJ/heart1.jpg") # แก้ไขตรงนี้

with col2:
    st.header("")
    st.image("https://i.ibb.co/3sXQ3cQ/heart2.jpg") # แก้ไขตรงนี้

html_7 = """
<div style="background-color:#33beff;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>ข้อมูลโรคหัวใจสำหรับทำนาย</h4></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

# Load the new dataset
try:
    dt = pd.read_csv("./data/Medicaldataset_converted.csv")
    st.write("ข้อมูลส่วนแรก 10 แถว")
    st.dataframe(dt.head(10))
    st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
    st.dataframe(dt.tail(10))
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'Medicaldataset_converted.csv' กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ data/ และชื่อไฟล์ถูกต้อง")

# Basic statistics
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# Feature selection for graph visualization
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# Boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกตามชนิดของโรคหัวใจ")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='Result', y=feature, ax=ax)
st.pyplot(fig)

# Pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='Result')
    st.pyplot(fig2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

# Dictionary for mapping English column names to Thai labels
feature_labels = {
    'Age': 'อายุ',
    'Gender': 'เพศ',
    'Heart rate': 'อัตราการเต้นของหัวใจ',
    'Systolic blood pressure': 'ความดันโลหิตตัวบน (ซิสโตลิก)',
    'Diastolic blood pressure': 'ความดันโลหิตตัวล่าง (ไดแอสโตลิก)',
    'Blood sugar': 'น้ำตาลในเลือด',
    'CK-MB': 'ครีเอตีนไคเนส เอ็มบี (เอนไซม์บ่งบอกกล้ามเนื้อหัวใจ)',
    'Troponin': 'โทรโปนิน (โปรตีนบ่งบอกความเสียหายของกล้ามเนื้อหัวใจ)'
}

# Create input fields dynamically based on the dataset features
user_input = {}
for feature in dt.columns[:-1]:
    label_text = feature_labels.get(feature, feature)
    if feature == 'Gender':
        gender_options = {'ชาย': 0, 'หญิง': 1}
        selected_gender = st.selectbox(
            f'กรุณาเลือกข้อมูล: {label_text}',
            options=list(gender_options.keys())
        )
        user_input[feature] = gender_options[selected_gender]
    else:
        user_input[feature] = st.number_input(f'กรุณาเลือกข้อมูล: {label_text}', value=0.0)

if st.button("ทำนายผล"):
    X = dt.drop('Result', axis=1)
    y = dt.Result

    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)
    
    x_input = np.array([[user_input[feature] for feature in X.columns]])

    prediction = Knn_model.predict(x_input)
    st.write("### ผลการทำนาย:")
    
    if prediction[0] == 1:
        st.success('ผลการทำนาย: คุณมีความเสี่ยงเป็นโรคหัวใจ')
        st.image("https://i.ibb.co/6b04LzJ/heart1.jpg") # แก้ไขตรงนี้
    else:
        st.success('ผลการทำนาย: คุณไม่มีความเสี่ยงเป็นโรคหัวใจ')
        st.image("https://i.ibb.co/3sXQ3cQ/heart2.jpg") # แก้ไขตรงนี้
    
else:
    st.write("ไม่ทำนาย")