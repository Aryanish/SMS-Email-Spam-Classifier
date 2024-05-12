import streamlit as st
import pickle
import nltk
import string
import sklearn

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    ps = PorterStemmer()

    y = []

    for i in text:
        if i.isalnum():
            y.append(ps.stem(i))

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.markdown(
    "\n"
    "    <style>\n"
    "    .reportview-container {\n"
    "        background: #;\n"
    "    }\n"
    "    </style>\n"
    "    ",
    unsafe_allow_html=True
)
from PIL import Image
image = Image.open('C:\\Users\\Aryan\\Downloads\\niet 3.png')
col1, col2, col3 = st.columns(3)
with col1:
    st.image(image)
st.title("Email/SMS Spam Classifier")
st.header("Created by Aryan Yadav From CSE (DATA SCIENCE)")

input_sms = st.text_input("Enter the message")
if st.button('predict'):

    # 1. preprocess
    transform_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
