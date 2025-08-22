import streamlit as st


st.title("ยินดีต้อนรับxxx")
st.markdown("โปรดเลือกหน้าจากเมนูด้านซ้ายเพื่อเริ่มใช้งาน")

# นี่คือส่วนที่สร้างลิงก์ไปที่หน้า "การทำนายโรคหัวใจวาย"
# โค้ดนี้จะใช้ได้ก็ต่อเมื่อมีไฟล์ DTree.py อยู่ในโฟลเดอร์ pages
st.page_link("pages/DTree.py", label="การทำนายโรคหัวใจวายDtree", icon="1️⃣")
st.page_link("pages/KnnwithHeart.py", label="การทำนายโรคหัวใจวายknn", icon="1️⃣")
st.page_link("pages/NaiveBaye.py", label="การทำนายโรคหัวใจวายNaiveBaye", icon="1️⃣")

# สามารถเพิ่มลิงก์ไปหน้าอื่น ๆ ได้ที่นี่
# st.page_link("pages/another_page.py", label="หน้าอื่น ๆ", icon="📄")