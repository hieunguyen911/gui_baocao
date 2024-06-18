import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_star_rating import st_star_rating
import warnings
warnings.filterwarnings('ignore', message="Error kìa")
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
from wordcloud import WordCloud
import re
# 1. Read data
data_res =pd.read_csv("df_res_merge.csv", encoding='utf-8')
data_res[["ID","ReviewCount"]]=data_res[["ID","ReviewCount"]].astype("Int64")
data_res[["Rating"]]=data_res[["Rating"]].astype(float)
data_res[["Address","District","Restaurant"]]=data_res[["Address","District","Restaurant"]].astype(str)
top_rating=data_res.sort_values(by=['Rating'], ascending=False).head(3)
r1=top_rating.iloc[0].tolist()
r2=top_rating.iloc[1].tolist()
r3=top_rating.iloc[2].tolist()
top_comment=data_res.sort_values(by=['ReviewCount'], ascending=False).head(3)
c1=top_comment.iloc[0].tolist()
c2=top_comment.iloc[1].tolist()
c3=top_comment.iloc[2].tolist()
id_list=data_res['ID'].unique()

# Set up the sidebar with three options
st.set_page_config(layout="wide")
st.sidebar.title("Menu")
selection = st.sidebar.radio("Go to", ["Main Page", "Project","About us"])

# Create a function for the login page
def find_res(words_list, dataframe, column_name='Restaurant',ratingCount=20):
    pattern = '|'.join(words_list)
    filtered_df = dataframe[dataframe[column_name].str.contains(pattern, case=True, na=False)]
    filtered_df = filtered_df[filtered_df['ReviewCount'] >= ratingCount]
    filtered_df = filtered_df.sort_values(by='Rating', ascending=False)
    filtered_df=filtered_df[['Restaurant','Address','Time','Price','District','Rating']]
    return filtered_df    

def find_res_are(words_list, dataframe, column_name='District',ratingCount=20):
    filtered_df = dataframe[dataframe[column_name].isin(words_list)]
    filtered_df = filtered_df[filtered_df['ReviewCount'] >= ratingCount]
    filtered_df = filtered_df.sort_values(by='Rating', ascending=False)
    filtered_df=filtered_df[['Restaurant','Address','Time','Price','District','Rating']]
    return filtered_df    
def res_item(name,address,rating,count,key_in):
    st.write("#### "+name)
    star=st_star_rating(label="Rating",maxValue=10, defaultValue=rating,size=20,key=key_in)
    st.write(star)
    st.write("###### Số lượt rating : "+str(count))
    st.write("###### Địa chỉ: "+ address)

def login_page():
    with st.container():
        left_co, cent_co,last_co = st.columns([3,15,3])   
        with cent_co:
            st.markdown(
            '<p style="color: orange; font-size: 64px;"> SENTIMENT ANALYSIS PROJECT</p>',unsafe_allow_html=True)
            st.image("https://statics.vinpearl.com/traditional-vietnamese-food-1_1689495611.jpg") 
            id_restaurant =st.text_input(label="Nhap vao ID",value="Enter your ID_Restaurant")
            if st.button("Đăng nhập"):
                try:
                    id_restaurant = int(id_restaurant)
                    if id_restaurant in id_list:
                        st.session_state["idRes"]=id_restaurant
                        st.session_state['data_res']=data_res
                        st.switch_page("pages/report_page.py")
                    else:
                        st.error("Incorrect ID. Please try again.")
                except ValueError:
                    st.write("Please enter a valid ID") 
            st.markdown(
            '<p style="color: orange; font-size: 64px;"> THỐNG KÊ CỬA HÀNG </p>',unsafe_allow_html=True)     
    with st.container():
        col1, col2 =st.columns([2,6])
        with col1:
            st.write("""## Chọn món yêu thích:""") 
            options = st.multiselect(
            "Chọn món",
            ["All","Hủ Tiếu", "Bánh Mì", "Cơm Tấm", "Bún bò", "Thịt nướng","Bánh Cuốn", "Cơm gà"],
            [ "Hủ Tiếu", "Bánh Mì"])
        with col2:
            num=len(options)
            st.write("""## Kết quả tìm kiếm: TOP CỬA HÀNG THEO MÓN ĂN""")
            st.dataframe(find_res(options,data_res).head(num+5), hide_index=True)
    with st.container():
        col1, col2 =st.columns([2,6])
        with col1:
            st.write("""## Chọn theo vị trí:""") 
            options2 = st.multiselect(
            "Vị trí",
            ["Quận 1","Quận 2","Quận 3","Quận 4","Quận 5","Quận 6","Quận 7","Quận 8","Quận 9","Quận 10","Quận 11","Quận 12"],
            [ "Quận 1","Quận 2"])
        with col2:
            num=len(options2)
            st.write("""## Kết quả tìm kiếm: TOP CỬA HÀNG THEO QUẬN""")
            st.dataframe(find_res_are(options2,data_res,column_name='District').head(num+7), hide_index=True)    
    with st.container():
       
        col1, col2 =st.columns(2)
        with col1:
            st.write("""## Top nhà hàng có rating cao nhất:""")
            res_item(name=r1[1],address=r1[2],count=r1[9],rating=r1[8],key_in=1)   
            res_item(name=r2[1],address=r2[2],count=r2[9],rating=r2[8],key_in=2)  
            res_item(name=r3[1],address=r3[2],count=r3[9],rating=r3[8],key_in=3)   
        with col2:
            st.write("""## Top nhà hàng có lượng review cao nhất:""")
            res_item(name=c1[1],address=c1[2],count=c1[9],rating=c1[8],key_in=4)
            res_item(name=c2[1],address=c2[2],count=c2[9],rating=c2[8],key_in=5)
            res_item(name=c3[1],address=c3[2],count=c3[9],rating=c3[8],key_in=6) 
            
# Create a function for the about us page
def about_us_page():
    st.title("About Us")
    st.write("""
    Welcome to our project! This is our team.
    """)
    col1, col2 =st.columns(2)
    with col1:
        st.image('anh_Hieu.png', caption="Nguyễn Minh Hiếu")
        st.write("##### Email: alex.machinedesigner@gmail.com")
        st.write("""##### MAIN TASK: GUI DESIGN, GUI CODING""")
    with col2:
        st.image('anh_Thien.png', caption="Lương Đức Thiện")
        st.write("""##### Email: ducthien.steven@gmail.com""")
        st.write("""##### MAIN TASK: MODEL IMPLEMENT, EDA, VIETNAMESE PROCESSING""")

# Create a function for the project page
def project_page():
    st.title("Project")
    st.write("""
    ### YÊU CẦU ĐỀ BÀI         
    #### Sentiment Analysis
    Yêu cầu xử lý dữ liệu Tiếng Việt các comment trên hệ thống Shopee Food.
    Phân tích Sentiment Analysis và xây dựng model trả về dữ liệu report cho chủ cửa hàng.
    """)
    st.write("""### EDA """)
    st.image('Project 3/EDA/RCBD.png')
    st.write("""Nhận xét:  
             Các quán ăn, nhà hàng có mặt trên ShopeeFood chủ yếu tập trung nhiều tại các quận trung tâm 1, 2, 3, 4, 5. """)
    st.image('Project 3/EDA/RaCBD.png')
    st.write("""
    Nhận xét:  
    Lượng reviews tập trung cho các quán ăn, nhà hàng các Quận 1, 3, 4, 5, xuất phát từ vị trí trung tâm thuận lợi, đông khách.  
    Quận 2 có số lượng food shop nhiềus Top5 (như chart trên) nhưng lượng reviews lại thuộc Top3 thấp nhất,  
    cho thấy vị trí địa lý và khu vực dân cư (đông/thưa) là vô cùng quan trọng, quyết định số lượng reviews.""")
    st.image('Project 3/EDA/RaCoM.png')
    st.write("""Nhận xét:  
    2011-2018: số lượng reviews có tốc độ gia tăng rất lớn, tăng liên tục qua mỗi tháng trong năm, và qua mỗi năm.  
    Điều này có thể đến từ với việc gia nhập ngành nhanh chóng, mở rộng mạng lưới và đội ngũ giao hàng của ShopeeFood, cùng với việc các nhà hàng, quán ăn lớn bé quan tâm và mở rộng kinh doanh hơn với kênh delivery (thay vì chỉ kinh doanh truyền thống đón khách dine-in).  
    Kéo theo đó là làn sóng food reviews tăng mạnh, đến từ những đánh giá của khách hàng thực, lẫn cả các chiến lược câu/mua reviews của các nhà hàng, quán ăn.  
    2019: đánh dấu sự chững lại số lượng food reviews.  
    2020-2022: đại dịch Covid-19 lan rộng, cách ly xã hội, lượt reviews giảm sút.  
    2023-2024: việc kinh doanh diễn ra trở lại bình thường. Lượng reviews thậm chí tăng vọt vào tháng 12/2023.""")
    st.image('Project 3/EDA/RaGbD.png')
    st.write("""Nhận xét:   
    Dữ liệu rating bị mất cân bằng với lượng rating rất nhiều cho nhóm điểm >=7.  
    Lượng rating >=7 tập trung cho các food shops tại các Quận 1, 3, 4, 5.""")
    st.image('Project 3/EDA/RaGbY.png')
    st.write("""Nhận xét:  
    2011-2017: khách hàng khá "hào phóng" khi cho điểm rất cao (rating >=7).  
    2018-2022: lượng rating <=2 bắt đầu gia tăng, có thể do khách hàng đã có nhiều trải nghiệm, bắt đầu trở nên khắt khe, khó tính khi cho điểm, HOẶC có thể đến từ sự cạnh tranh không lành mạnh trong kinh doanh giữa các food shops.  
    Đặc biệt riêng trong 2023-nay, lượng rating 5-6 điểm tăng vọt một cách khó hiểu! """)
    st.image('Project 3/EDA/PwC.png', caption="POSITIVE WORD CLOUD")
    st.image('Project 3/EDA/NwC.png', caption="NEGATIVE WORD CLOUD")
    st.write("""### XỬ LÝ NGÔN NGỮ TIẾNG VIỆT""")
    st.image("Project 3/vietProcess.jpg", caption="QUY TRÌNH XỬ LÝ NGÔN NGỮ TIẾNG VIỆT")
    st.write("""### XÂY DỰNG TẬP LUẬT""")
    st.image("Project 3/tapLuat.jpg", caption="TẬP LUẬT")
    st.write("""### XÂY DỰNG MODEL""")
    st.write("""#### Kết quả fiting model 3 classes""")
    col1, col2 =st.columns(2)
    with col1:
        st.image('Project 3/Model/kq1.3.1.PNG')
    with col2:
        st.image('Project 3/Model/kq1.3.2.PNG')
    st.write("""
    Nhận xét: Nếu chia làm 3 nhóm:  
    Nhóm 1 (Negative): Các models hoạt động ở mức trung bình, nhưng vẫn còn khả năng cải thiện, đặc biệt là recall (số lượng lớn các kết quả false negatives).  
    Nhóm 2 (Neutral): Các models gặp khó khăn đáng kể với nhóm này, có độ accuracy và recall thấp.  
    Nhóm 3 (Positive): Các models hoạt động khá tốt, với accuracy và recall cao, cho thấy xác định và dự đoán đúng hầu hết các trường hợp của nhóm 3.  
    KẾT LUẬN: CHỈ PHÂN LOẠI THEO 2 NHÓM: POSITIVE và NEGATIVE.""")
    st.write("""#### Kết quả fiting model 2 classes""")
    col1, col2 =st.columns(2)
    with col1:
        st.image('Project 3/Model/kq2.2.1.PNG')
    with col2:
        st.image('Project 3/Model/kq2.2.2.PNG')
    st.write("""Nhận xét:  
    MultinomialNB và SVM là 02 models cho kết quả accuracy cao nhất.  
    Tuy nhiên MultinomialNB mất rất ít thời gian xử lý, đồng thời recall (dự đoán) cho Nhóm 1 (Negative) có chút nhỉnh hơn SVM.""")
    st.write("""#### Kết quả fiting model 2 classes Spark""")
    st.image('Project 3/Model/kq3.1.PNG')
    st.write("""#### KẾT LUẬN:  Chọn thuật toán MultinomialNB trong machine learning truyền thống làm mô hình Sentiment analysis, vì có accuracy cao và thời gian xử lý nhanh.""")
    
    
# Display the selected page
if selection == "Main Page":
    login_page()
elif selection == "About us":
    about_us_page()
elif selection == "Project":
    project_page()
