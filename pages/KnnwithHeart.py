from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="การทำนายโรคหัวใจด้วย KNN",
    page_icon="❤️",
    layout="wide"
)

# --- Header Section ---
st.title("❤️ การทำนายโรคหัวใจด้วยเทคนิค K-Nearest Neighbor")
st.markdown("""
แอปพลิเคชันนี้ใช้โมเดล **K-Nearest Neighbor (KNN)** เพื่อทำนายความเสี่ยงโรคหัวใจจากข้อมูลสุขภาพของคุณ
""")
st.markdown("---")

# --- Image Section ---

# --- Data Information Section ---
st.subheader("📊 ข้อมูลที่ใช้ในการฝึกโมเดล")
st.info("โปรดตรวจสอบข้อมูลเพื่อทำความเข้าใจลักษณะของชุดข้อมูล")
try:
    dt = pd.read_csv("./data/Medicaldataset_converted.csv")
    st.write("แสดงข้อมูล 10 แถวแรก:")
    st.dataframe(dt.head(10))
except FileNotFoundError:
    st.error("❌ **ไม่พบไฟล์ 'Medicaldataset_converted.csv'** กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ `data/` และชื่อไฟล์ถูกต้อง")
    st.stop() # Stop the app if file is not found

# --- Visualization Section ---
st.markdown("---")
st.subheader("📈 การสำรวจข้อมูล (Data Exploration)")
st.info("ใช้กราฟด้านล่างเพื่อดูการกระจายของข้อมูลแต่ละฟีเจอร์")

feature = st.selectbox(
    "เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล", 
    dt.columns[:-1],
    help="เลือกฟีเจอร์เพื่อดูว่าค่าของฟีเจอร์นั้นๆ มีการกระจายตัวอย่างไรเมื่อเทียบกับผลลัพธ์ (Result)"
)

# Boxplot
st.write(f"#### 🎯 Boxplot: แสดงการกระจายของฟีเจอร์ '{feature}'")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=dt, x='Result', y=feature, ax=ax)
ax.set_title(f'Boxplot ของ {feature} เทียบกับผลลัพธ์', fontsize=16)
ax.set_xlabel('Result (0: ไม่มี, 1: มี)', fontsize=12)
ax.set_ylabel(feature, fontsize=12)
st.pyplot(fig)

# Pairplot (Optional)
st.markdown("---")
if st.checkbox("แสดง Pairplot เพื่อดูความสัมพันธ์ของทุกฟีเจอร์ (อาจใช้เวลาโหลดสักครู่)"):
    st.write("#### 🌺 Pairplot: ความสัมพันธ์ระหว่างฟีเจอร์ทั้งหมด")
    st.warning("การแสดงกราฟนี้อาจใช้เวลาประมวลผลนาน หากชุดข้อมูลมีขนาดใหญ่")
    with st.spinner('กำลังสร้าง Pairplot...'):
        fig2 = sns.pairplot(dt, hue='Result')
        st.pyplot(fig2)

# --- Prediction Section ---
st.markdown("---")
st.header("🔮 ทำนายผลความเสี่ยงโรคหัวใจ")
st.info("กรุณาป้อนข้อมูลของคุณในช่องด้านล่างเพื่อรับการทำนาย")

# Dictionary for mapping English column names to Thai labels
feature_labels = {
    'Age': 'อายุ', 'Gender': 'เพศ', 'Heart rate': 'อัตราการเต้นของหัวใจ (ครั้ง/นาที)',
    'Systolic blood pressure': 'ความดันโลหิตตัวบน (mmHg)', 'Diastolic blood pressure': 'ความดันโลหิตตัวล่าง (mmHg)',
    'Blood sugar': 'น้ำตาลในเลือด (mg/dL)', 'CK-MB': 'CK-MB (ng/mL)', 'Troponin': 'โทรโปนิน (ng/mL)'
}

# Create input fields using columns for better layout
user_input = {}
cols = st.columns(2)

for i, feature in enumerate(dt.columns[:-1]):
    with cols[i % 2]:
        label_text = feature_labels.get(feature, feature)
        if feature == 'Gender':
            gender_options = {'ชาย': 0, 'หญิง': 1}
            selected_gender = st.selectbox(
                f'กรุณาเลือกข้อมูล: {label_text}',
                options=list(gender_options.keys()),
                key=f"input_{feature}"
            )
            user_input[feature] = gender_options[selected_gender]
        else:
            user_input[feature] = st.number_input(
                f'กรุณาป้อนค่าสำหรับ: **{label_text}**', 
                value=0.0,
                key=f"input_{feature}"
            )

# Prediction button and result display
st.markdown("---")
if st.button("🌟 ทำนายผล", type="primary"):
    X = dt.drop('Result', axis=1)
    y = dt.Result

    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)
    
    x_input = np.array([[user_input[feature] for feature in X.columns]])

    prediction = Knn_model.predict(x_input)
    st.subheader("✅ ผลการทำนาย:")
    
    if prediction[0] == 1:
        st.error('⚠️ **คุณมีความเสี่ยงที่จะเป็นโรคหัวใจ**')
        st.markdown("ขอแนะนำให้ปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อยืนยันผลและรับคำแนะนำที่ถูกต้อง")
    else:
        st.success('🟢 **คุณไม่มีความเสี่ยงที่จะเป็นโรคหัวใจ**')
        st.markdown("อย่างไรก็ตาม การดูแลสุขภาพอย่างสม่ำเสมอเป็นสิ่งสำคัญ")