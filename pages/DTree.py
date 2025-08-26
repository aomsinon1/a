import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.header("Decision Tree for classification")

# โหลดข้อมูลจากไฟล์ที่ผู้ใช้ส่งมา
# ใช้ try-except เพื่อจัดการกรณีไม่พบไฟล์
try:
    # แก้ไขพาธเพื่อให้โค้ดสามารถเข้าถึงไฟล์ในโฟลเดอร์ data/ ได้
    df = pd.read_csv('data/Medicaldataset_converted.csv')
    st.write("ข้อมูล 10 แถวแรกของชุดข้อมูล:")
    st.dataframe(df.head(10))

    # --- การเตรียมข้อมูลสำหรับโมเดล ---
    # *สำคัญ*: คุณต้องเปลี่ยนชื่อคอลัมน์เป้าหมาย ('target_column')
    # ให้ตรงกับคอลัมน์ที่คุณต้องการทำนายในไฟล์ CSV ของคุณ
    # ตัวอย่าง: สมมติว่าคอลัมน์ที่ต้องการทำนายชื่อว่า 'Result'
    target_column = 'Result'
    
    # ดึงรายชื่อคอลัมน์ที่เป็น feature ทั้งหมดโดยอัตโนมัติ
    features = [col for col in df.columns if col != target_column]

    # กำหนด X (features) และ y (target)
    X = df[features]
    y = df[target_column]

    # แบ่งข้อมูลเป็นชุดฝึก (training set) และชุดทดสอบ (testing set)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

    # สร้างและฝึกโมเดล Decision Tree
    ModelDtree = DecisionTreeClassifier()
    dtree = ModelDtree.fit(x_train, y_train)

    # --- ส่วนรับข้อมูลจากผู้ใช้เพื่อพยากรณ์ ---
    st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์")

    # สร้าง dictionary สำหรับแปลงชื่อคอลัมน์เป็นภาษาไทย
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

    # สร้างช่องรับข้อมูล (number_input) สำหรับแต่ละ feature โดยอัตโนมัติ
    user_input = {}
    for feature in features:
        label_text = feature_labels.get(feature, feature)
        if feature == 'Gender':
            # สำหรับเพศ ให้ใช้ selectbox แทน
            gender_options = {'ชาย': 0, 'หญิง': 1}
            selected_gender = st.selectbox(
                f'ป้อนค่าสำหรับ: {label_text}',
                options=list(gender_options.keys())
            )
            user_input[feature] = gender_options[selected_gender]
        else:
            # สำหรับ feature อื่นๆ ที่เป็นตัวเลข ยังคงใช้ number_input
            user_input[feature] = st.number_input(f'ป้อนค่าสำหรับ: {label_text}', value=0.0)

    if st.button("พยากรณ์"):
        # สร้างรายการค่าที่ผู้ใช้ป้อน เพื่อส่งให้โมเดลทำนาย
        x_input = [[user_input[feature] for feature in features]]
        y_predict2 = dtree.predict(x_input)
        
        st.write("### ผลการพยากรณ์:")
        # แสดงผลการทำนาย
        st.write(y_predict2)
        st.button("ไม่พยากรณ์")
    else:
        st.button("ไม่พยากรณ์")

    # --- ส่วนแสดงความแม่นยำและการพล็อตต้นไม้ ---
    # ทำนายผลจากชุดข้อมูลทดสอบ
    y_predict = dtree.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    st.write(f'ความแม่นยำของโมเดลในการพยากรณ์: **{(score * 100):.2f} %**')

    # พล็อต (แสดงภาพ) Decision Tree
    fig, ax = plt.subplots(figsize=(12, 8))
    tree.plot_tree(dtree, feature_names=features, ax=ax, filled=True, rounded=True)
    st.pyplot(fig)

except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'Medicaldataset_converted.csv' กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ data/ และชื่อไฟล์ถูกต้อง")
except KeyError as e:
    st.error(f"เกิดข้อผิดพลาด: ไม่พบคอลัมน์ '{e}' ในไฟล์ CSV ของคุณ กรุณาตรวจสอบว่าชื่อคอลัมน์ในโค้ดตรงกับในไฟล์ข้อมูล")
except Exception as e:
    st.error(f"เกิดข้อผิดพลาด: {e}")