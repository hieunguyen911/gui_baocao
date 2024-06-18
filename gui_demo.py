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
#VIETNAMESE PROCESSING

##LOAD EMOJICON
file = open('Project 3/Model/files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('Project 3/Model/files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('Project 3/Model/files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('Project 3/Model/files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('Project 3/Model/files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()
#function1:
def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = str(text)
    try:
        document = text.lower()
    except AttributeError:
        # Skip text.lower() for non-string inputs
        pass

    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        # ...
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document


#function2:
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)


#function3:
def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i = 0
    special_words = {'không', 'chẳng'}  # không chọn 'chả' vì có 'bún chả', không chọn 'hông' vì có 'bên hông chợ'

    if special_words.intersection(text_lst):
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            if word in special_words:
                next_idx = i + 1
                if next_idx <= len(text_lst) - 1:
                    word = word + '_' + text_lst[next_idx]
                i = next_idx + 1
            else:
                i = i + 1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()


#function4:
import re
# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "ngonnnn" thành "ngon", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)

# Áp dụng hàm chuẩn hóa cho văn bản
# print(normalize_repeated_characters(example))


#function5:
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        # lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        lst_word_type = ['A','AB','V','VB','VY','R']  # chỉ quan tâm các POS mô tả, bỏ các chủ thể
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document


#function6:
def remove_stopword(text, stopwords, additional_stopwords=None):
    if additional_stopwords:
        # Combine stopwords and additional stopwords into a set for efficient lookups
        all_stopwords = set(stopwords) | set(additional_stopwords)
    else:
        all_stopwords = set(stopwords)
    ###### REMOVE stop words
    document = ' '.join('' if word in all_stopwords else word for word in text.split())
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document


#function7:
positive_words = [
    "thích", "tốt", "xuất_sắc", "tuyệt_vời", "tuyệt_hảo", "đẹp", "ổn", "ngon",
    "hài_lòng", "ưng_ý", "hoàn_hảo", "chất_lượng", "thú_vị", "nhanh",
    "tiện_lợi", "dễ_sử_dụng", "hiệu_quả", "ấn_tượng",
    "nổi_bật", "tận_hưởng", "tốn_ít_thời_gian", "thân_thiện", "hấp_dẫn",
    "gợi_cảm", "tươi_mới", "lạ_mắt", "cao_cấp", "độc_đáo",
    "hợp_khẩu_vị", "rất_tốt", "rất_thích", "đáng_tin_cậy", "đẳng_cấp",
    "hấp_dẫn", "an_tâm", "không_thể_cưỡng_lại", "thỏa_mãn", "thúc_đẩy",
    "cảm_động", "phục_vụ_tốt", "làm_hài_lòng", "gây_ấn_tượng", "nổi_trội",
    "sáng_tạo", "quý_báu", "phù_hợp", "tận_tâm",
    "hiếm_có", "cải_thiện", "hoà_nhã", "chăm_chỉ", "cẩn_thận",
    "vui_vẻ", "sáng_sủa", "hào_hứng", "đam_mê", "vừa_vặn", "đáng_tiền",

    # new-added
    "tuyệt", "yêu", "mê", "hợp_khẩu", "đáng_yêu", "dễ_thương", "thơm", "rẻ", "bổ",
    "nhiệt_tình", "hài_lòng", "yêu_thương", "ok", "okay", "okie", "oke", "okela", "lịch sự", "thỏa_thích",
    "lễ_phép", "sạch", "sạch_sẽ", "ủng_hộ", "hợp_lý", "thích_hợp", "miễn_chê", "ngon_nhất", "phải_chăng"
]

negative_words = [
    "kém", "tệ", "đau", "xấu", "dở", "ức",
    "buồn", "rối", "thô", "lâu", "chán"
    "tối", "chán", "ít", "mờ", "mỏng",
    "lỏng_lẻo", "khó", "cùi", "yếu",
    "kém_chất_lượng", "không_thích", "không_thú_vị", "không_ổn",
    "không_hợp", "không_đáng_tin_cậy", "không_chuyên_nghiệp",
    "không_phản_hồi", "không_an_toàn", "không_phù_hợp", "không_thân_thiện", "không_linh_hoạt", "không_đáng_giá",
    "không_ấn_tượng", "không_tốt", "chậm", "khó_khăn", "phức_tạp",
    "khó_hiểu", "khó_chịu", "gây_khó_dễ", "rườm_rà", "khó_truy_cập",
    "thất_bại", "tồi_tệ", "khó_xử", "không_thể_chấp_nhận", "không_rõ_ràng",
    "không_chắc_chắn", "rối_rắm", "không_tiện_lợi", "không_đáng_tiền", "chưa_đẹp", "không_đẹp",

    # new-added
    "thất_vọng", "dơ", "ngán", "khóc", "nuốt_không_nổi", "ghét", "mặn", "nhạt", "đắt", "lạt", "tệ"
    "hôi", "tanh", "tức_giận", "nhăn_nhó", "sống", "không_thèm", "nguội", "dị_ứng", "chê", "chật", "khủng_khiếp",
    "không_sạch", "không_sạch_sẽ", "bất_tiện", "không_vệ_sinh"
]

def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []

    for word in list_of_words:
        if word in document_lower:
            word_count += document_lower.count(word)
            word_list.append(word)

    return word_count, word_list

pkl_filename = "Project 3/Model/Sentiment_MNB_model.pkl"
pkl_tfidf = "Project 3/Model/tfidf_vectorizer.pkl"

#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    sentiment_model = pickle.load(file)
# doc model count len
with open(pkl_tfidf, 'rb') as file:  
    tfidf_model = pickle.load(file)
    
#function8: Classify_comment
def classify_comment(comment):
    # Preprocess the comment
    processed_comment = process_text(comment, emoji_dict, teen_dict, wrong_lst)
    processed_comment = covert_unicode(processed_comment)
    processed_comment = process_special_word(processed_comment)
    processed_comment = normalize_repeated_characters(processed_comment)
    processed_comment = process_postag_thesea(processed_comment)
    processed_comment = remove_stopword(processed_comment, stopwords_lst)

    # Transform the comment to TF-IDF features
    X_tfidf = tfidf_model.transform([processed_comment])

    # Predict the class of the comment
    y_pred = sentiment_model.predict(X_tfidf)[0]
    comment_sentiment = 'Tiêu cực' if y_pred == 1 else 'Tích cực'

    # Find positive and negative words in the comment
    pwc,positive_words_found = find_words(processed_comment, positive_words)
    nwc,negative_words_found = find_words(processed_comment, negative_words)

    # Ensure the elements are strings
    positive_words_found_str = [str(word) for word in positive_words_found]
    negative_words_found_str = [str(word) for word in negative_words_found]

    # Generate word clouds based on found words
    col1, col2=st.columns(2)
    with col1:
        if pwc>0:
            positive_wordcloud = WordCloud(background_color="white", max_font_size=50, scale=3).generate(' '.join(positive_words_found_str))
            fig, ax = plt.subplots(figsize = (12, 8))
            ax.imshow(positive_wordcloud, interpolation = 'bilinear')
            plt.axis('off')
            plt.title("TÍCH CỰC")
            st.pyplot(fig)  
               
    with col2:
        if nwc>0:
            negative_wordcloud = WordCloud(background_color="white", max_font_size=50, scale=3).generate(' '.join(negative_words_found_str))
            fig, ax = plt.subplots(figsize = (12, 8))
            ax.imshow(negative_wordcloud, interpolation = 'bilinear')
            plt.axis("off")
            plt.title("TIÊU CỰC")
            st.pyplot(fig)
            
    st.write(f"""#### Phân nhóm: {comment_sentiment}""")
    st.write(f"""##### Mô tả tích cực: {positive_words_found}""")
    st.write(f"""##### Mô tả tiêu cực: {negative_words_found}""")
def csv_report(df_testcsv):
        df_testcsv["comment"]= df_testcsv["comment"].apply(lambda x: process_text(x, emoji_dict, teen_dict, wrong_lst))
        df_testcsv["comment"] = df_testcsv["comment"].apply(lambda x: covert_unicode(x))
        df_testcsv["comment"] = df_testcsv["comment"].apply(lambda x: process_special_word(x))
        df_testcsv["comment"] = df_testcsv["comment"].apply(lambda x: normalize_repeated_characters(x))
        df_testcsv["comment"] = df_testcsv["comment"].apply(lambda x: process_postag_thesea(x)) 
        df_testcsv["comment"] = df_testcsv["comment"].apply(lambda x: remove_stopword(x, stopwords_lst))
        df_testcsv['P_list'] = df_testcsv['comment'].apply(lambda x: find_words(x, positive_words)[1])
        df_testcsv['N_list'] = df_testcsv['comment'].apply(lambda x: find_words(x, negative_words)[1])
        # Transform the comment to TF-IDF features
        df_testcsv['comment'].fillna(" ", inplace=True)
        X = df_testcsv['comment']
        X_tfidf = tfidf_model.transform(X)

        # Predict the class of the comment
        y_pred = sentiment_model.predict(X_tfidf)
        df_testcsv.loc[:, 'predict'] = y_pred
        df_testcsv.loc[:, 'predict'] = df_testcsv['predict'].map({1: 'Negative', 2: 'Positive'})
        # Map predictions to labels
        positive_comments_count = df_testcsv['predict'].value_counts().get('Positive', 0)
        negative_comments_count = df_testcsv['predict'].value_counts().get('Negative', 0)
        # Print result
        col1, col2  =st.columns(2)
        with col1:
            # Print number of positive and negative comments
            st.write(f"Số lượng bình luận tích cực: {positive_comments_count}")
            st.write(f"Số lượng bình luận tiêu cực: {negative_comments_count}")
        with col2: 
             # Plot bar chart of comment statistics
            fig, ax = plt.subplots(figsize = (5, 3))
            df_testcsv.groupby('predict')['predict'].count().plot(kind='bar', title='Thống kê bình luận')
            plt.yticks(range(0, 50, 10))
            plt.title("Thống kê bình luận")
            st.pyplot(fig)
        st.table(df_testcsv)
        col1, col2  =st.columns(2)
        with col1:
        # Generate word clouds for positive and negative comment
            st.write("""### Word Cloud bình luận tích cực""")
            positive_comments = df_testcsv[df_testcsv['predict'] == 'Positive']['P_list']
            if not positive_comments.empty:
                fig1, ax1= plt.subplots(figsize = (12, 8))
                wordcloudpos = WordCloud(background_color="white", max_font_size=50, scale=3).generate(' '.join(positive_comments.astype(str).values))
                ax1.imshow(wordcloudpos, interpolation = 'bilinear')
                plt.axis('off')
                plt.title("TÍCH CỰC")
                st.pyplot(fig1)
        with col2:
            st.write("""### Word Cloud bình luận tiêu cực""")
            negative_comments = df_testcsv[df_testcsv['predict'] == 'Negative']['N_list']
            if not negative_comments.empty:
                fig2, ax2= plt.subplots(figsize = (12, 8))
                wordcloudneg = WordCloud(background_color="white", max_font_size=50, scale=3).generate(' '.join(negative_comments.astype(str).values))
                ax2.imshow(wordcloudneg, interpolation = 'bilinear')
                plt.axis('off')
                plt.title("TIÊU CỰC")
                st.pyplot(fig2)   
    
# Set up the sidebar with three options
st.set_page_config(layout="wide")
st.sidebar.title("Menu")
selection = st.sidebar.radio("Go to", ["Main Page", "Prediction Tool", "Project","About us"])

# Create a function for the login page
def find_res(words_list, dataframe, column_name='Restaurant',ratingCount=20):
    pattern = '|'.join(words_list)
    filtered_df = dataframe[dataframe[column_name].str.contains(pattern, case=False, na=False)]
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
        st.title("Main Page")
        id_restaurant =st.text_input("ID Restaurant", value="Enter your ID")
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
            
    with st.container():
       
        col1, col2 =st.columns([1,3])
        with col1:
            st.write("""## Chọn món yêu thích:""") 
            options = st.multiselect(
            "Chọn món",
            ["All","Hủ Tiếu", "Bánh Mì", "Cơm Tấm", "Bún bò", "Thịt nướng","Bánh Cuốn", "Cơm gà"],
            [ "Hủ Tiếu", "Bánh Mì"])
        with col2:
            num=len(options)
            st.write("""## Kết quả tìm kiếm:""")
            st.dataframe(find_res(options,data_res).head(num+5), hide_index=True)
    with st.container():
       
        col1, col2, col3 =st.columns(3)
        with col1:
            pass
        with col2:
            st.write("""## Top nhà hàng có rating cao nhất:""")
            res_item(name=r1[1],address=r1[2],count=r1[9],rating=r1[8],key_in=1)   
            res_item(name=r2[1],address=r2[2],count=r2[9],rating=r2[8],key_in=2)  
            res_item(name=r3[1],address=r3[2],count=r3[9],rating=r3[8],key_in=3)     
        with col3:
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
    
    
def prediction_demo():
    st.title("Prediction Tool")
    st.write("""
    Sử dụng file hoặc input comment tại đây để xử lý dữ liệu
    """)
    st.subheader("Select data")
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            lines.columns = ['comment']
            st.dataframe(lines)
            csv_report(lines)    
    if type=="Input":        
        comment = st.text_area(label="Input your comment:")
        if comment!="":
            lines = comment
            classify_comment(lines) 
 
# Display the selected page
if selection == "Main Page":
    login_page()
elif selection == "About us":
    about_us_page()
elif selection == "Project":
    project_page()
elif selection == "Prediction Tool":
    prediction_demo()
