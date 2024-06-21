from streamlit_echarts import st_echarts
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from streamlit_star_rating import st_star_rating
import warnings
warnings.filterwarnings('ignore', message="Error kìa")
import regex
from wordcloud import WordCloud
import re
from underthesea import word_tokenize, pos_tag, sent_tokenize
import pickle

st.set_page_config(layout="wide")
#1 read data
data=pd.read_csv("data_new.csv", encoding='utf-8')
data_res=st.session_state.data_res
id_res=st.session_state.idRes

#2 process data
df_comment= data[data["IDRestaurant"]==id_res]
df_comment['Time']=df_comment['Time'].apply(pd.to_datetime)
df_comment['Time_Y'] = df_comment['Time'].apply(lambda x: str(x.year))
score= df_comment['Total_Score_2'].value_counts()
try:
    #', score[1])
    if (score[1]!=0) and (score[1]!=0):
        p_n=round(score[2]/(score[1]+score[2])*100,2)
    else:
        p_n=0
except:
    p_n=0
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
    "thất_vọng", "dơ", "ngán", "khóc", "nuốt_không_nổi", "ghét", "mặn", "nhạt", "đắt", "lạt",
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
def word_cloud(text):
    wordcloud = WordCloud(background_color="white", max_font_size=50, scale=3).generate(' '.join(text))
    return wordcloud
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
def sentiment_report():    
        # Transform the comment to TF-IDF features
        df_comment['Comment_processed'].fillna(" ", inplace=True)
        X = df_comment['Comment_processed']
        X_tfidf = tfidf_model.transform(X)

        # Predict the class of the comment
        y_pred = sentiment_model.predict(X_tfidf)
        df_comment.loc[:, 'predict'] = y_pred
        df_comment.loc[:, 'predict'] = df_comment['predict'].map({1: 'Negative', 2: 'Positive'})
        # Map predictions to labels
        positive_comments_count = df_comment['predict'].value_counts().get('Positive', 0)
        negative_comments_count = df_comment['predict'].value_counts().get('Negative', 0)
        # Print result
        col1, col2  =st.columns(2)
        with col1:
            # Print number of positive and negative comments
            st.write("""#### Trung bình Rating: """, df_comment['Rating'].mean())
            st.write("""#### Số lượng review: """, df_comment['Comment'].count())
            st.write(f"""#### Số lượng bình luận tích cực: {positive_comments_count}""")
            st.write(f"""#### Số lượng bình luận tiêu cực: {negative_comments_count}""")
        with col2: 
             # Plot bar chart of comment statistics
            fig, ax = plt.subplots(figsize = (5, 3))
            df_comment.groupby('predict')['predict'].count().plot(kind='bar', title='Thống kê bình luận')
            plt.yticks(range(0, 50, 10))
            plt.title("Thống kê bình luận")
            st.pyplot(fig)
            
        # Print the 10 most recent comments
        print("\n10 bình luận mới nhất")
        report = df_comment.sort_values(by='Time', ascending=False).head(10)
        report = report[["IDRestaurant", "User", "Time", "Rating", 'Comment', 'predict']]  # Modify to match your actual columns
        st.table(report)
        col1, col2  =st.columns(2)
        with col1:
        # Generate word clouds for positive and negative comment
            st.write("""### Word Cloud bình luận tích cực""")
            positive_comments = df_comment[df_comment['predict'] == 'Positive']['Comment_processed']
            if not positive_comments.empty:
                fig1, ax1= plt.subplots(figsize = (12, 8))
                wordcloudpos = WordCloud(background_color="white", max_font_size=50, scale=3).generate(' '.join(positive_comments))
                ax1.imshow(wordcloudpos, interpolation = 'bilinear')
                plt.axis('off')
                plt.title("TÍCH CỰC")
                st.pyplot(fig1)
        with col2:
            st.write("""### Word Cloud bình luận tiêu cực""")
            negative_comments = df_comment[df_comment['predict'] == 'Negative']['Comment_processed']
            if not negative_comments.empty:
                fig2, ax2= plt.subplots(figsize = (12, 8))
                wordcloudneg = WordCloud(background_color="white", max_font_size=50, scale=3).generate(' '.join(negative_comments))
                ax2.imshow(wordcloudneg, interpolation = 'bilinear')
                plt.axis('off')
                plt.title("TIÊU CỰC")
                st.pyplot(fig2)
                
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
        st.table(df_testcsv[['predict','comment','P_list','N_list']])
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
def res_item(name,address,rating,count):
    st.write("#### "+name)
    star=st_star_rating(label="Rating",maxValue=10, defaultValue=rating,size=20)
    st.write(star)
    st.write("###### Số lượt rating : "+str(count))
    st.write("###### Địa chỉ: "+ address) 
r=data_res[data_res["ID"]==id_res].iloc[0].to_list()



#GUI
st.sidebar.title("Report Page")
selection = st.sidebar.radio("Go to", ["Restaurant Report", "Prediction Tool"])
def res_report():
    if st.button("Quay lại Trang chủ"):
        st.switch_page("gui_demo.py")
    with st.container():
        col1, col2  =st.columns(2)
        with col1:
            res_item(name=r[1],address=r[2],count=r[9],rating=r[8])
        with col2:
            option = {
                "tooltip": {
                    "formatter": '{a} <br/>{b} : {c}%'
                },
                "series": [{
                    "name": 'Progress',
                    "type": 'gauge',
                    "startAngle": 180,
                    "endAngle": 0,
                    "progress": {
                        "show": "true"
                    },
                    "radius":'100%', 

                    "itemStyle": {
                        "color": '#58D9F9',
                        "shadowColor": 'rgba(0,138,255,0.45)',
                        "shadowBlur": 10,
                        "shadowOffsetX": 2,
                        "shadowOffsetY": 2,
                        "radius": '55%',
                    },
                    "progress": {
                        "show": "true",
                        "roundCap": "true",
                        "width": 15
                    },
                    "pointer": {
                        "length": '60%',
                        "width": 8,
                        "offsetCenter": [0, '5%']
                    },
                    "detail": {
                        "valueAnimation": "true",
                        "formatter": '{value}%',
                        "backgroundColor": '#58D9F9',
                        "borderColor": '#999',
                        "borderWidth": 4,
                        "width": '60%',
                        "lineHeight": 20,
                        "height": 20,
                        "borderRadius": 188,
                        "offsetCenter": [0, '45%'],
                        "valueAnimation": "true",
                    },
                    "data": [{
                        "value": p_n,
                        "name": 'Mức độ hài lòng'
                    }]
                }]
            };
            st.write(st_echarts(options=option))
    with st.container():
        col1, col2  =st.columns(2)
        with col1:
            st.write("""#### Số lượt comment theo thời gian""") 
            fig, ax = plt.subplots(figsize = (10, 3))
            ax.hist(df_comment['Time_Y'].sort_values(), bins=100)
            plt.yticks(range(0, 50, 10))
            plt.xlabel('Year')
            st.pyplot(fig)
        with col2:
            st.write("""#### Số lượt Rating""") 
            bins = np.arange(0, 11, 1)  # Creates bins [0, 1, 2, ..., 10]
            labels = [f'{i} to {i+1}' for i in range(10)]
            df_comment['RatingGroup'] = pd.cut(df_comment['Rating'], bins=bins, labels=labels, right=False)
            df_comment.groupby('RatingGroup')['RatingGroup'].count().plot(kind='bar')
            st.pyplot(fig)  
        sentiment_report()
def prediction_tool():
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
if selection == "Restaurant Report":
    res_report()
elif selection == "Prediction Tool":
    prediction_tool()
