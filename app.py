import streamlit as st 

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
import string

import pandas as pd
import joblib 

import plotly.express as px


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)



from src.bertpluslayers import BertSentimentClassifier

model = BertSentimentClassifier(3, 128)
model.load_state_dict(torch.load('models/bestmodel.model', map_location=torch.device('cpu')))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


max_len = 82

def predict(sentence):
    model.eval()
    encoding = tokenizer.encode_plus(
      sentence,
      max_length=max_len,
      add_special_tokens=True, 
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',  
    )

    mapper = {1:'neutral', 0:'negative', 2:'positive'}
    
    output = model(encoding['input_ids'], encoding['attention_mask'])
    
    return mapper[output.detach().numpy().argmax()]


punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

parser = spacy.load("en_core_web_sm")

def spacy_processor(sentence):
    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens


lr_model = joblib.load("models/pipe_random_forest_regressor.sav")

def main():
    
    st.title("Natural Language Processing For Finance")

    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h1 style="color:white;text-align:center;">Streamlit NLP App </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    activity = ['Sentiment Analysis','NER', 'WordCloud']
    choice = st.sidebar.selectbox("Select Activity",activity)

    docx_file = st.file_uploader("Upload Article",type=['txt'])
    if docx_file is not None:
        text = str(docx_file.read(),"utf-8")

    if choice == 'Sentiment Analysis':
        st.info("Bert Classification")
        

        if (st.button('classify')):

            doc = parser(text) 
            sentences = [sent.text.strip() for sent in doc.sents]
            classification = [predict(sent) for sent in sentences]

            regression = lr_model.predict(pd.Series(sentences))
            arsentement = pd.DataFrame({"sentences": sentences, "reg": regression, "class": classification})
            arsentement["absreg"] = abs(arsentement["reg"])
            
            fig = px.scatter(arsentement, 
                 y="class", 
                 x="reg", 
                 color="reg", 
                 size="absreg", 
                 hover_data=["class", "reg", "sentences"],
                 range_x=[-2.5, 2.5])

            st.plotly_chart(fig)

    if choice == 'WordCloud':
        st.info("Words Cloud")

        if st.button("Generate"):
            wordcloud = WordCloud(background_color="white", max_words=50, stopwords=set(STOPWORDS)).generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot()

            
    if choice == 'NER':
        st.info("Named Entity Recognition")

        if st.button("Extract Named Entity"):
            HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>""" 
            html = displacy.render(parser(text), style="ent")
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True) 


    st.sidebar.subheader("About")
    st.sidebar.write("This streamlit application aims to help reading news article, by extracting critical information buried in long documents") 
    st.sidebar.markdown("""
    \n\n Technologies:
    * `spacy`
    * `torch`
    * `LLMs`                    
    * `sckit learn`
    """)
    st.sidebar.markdown("""<h1 style='text-align: center;color:  #0e76a8;'><a style='text-align: center;color:  #0e76a8;' href="https://www.linkedin.com/in//youssefamdouni/fr" target="_blank">Linkedin Profile</a></h1>""", unsafe_allow_html=True)
    st.sidebar.markdown("""<h1 style='text-align: center;color: black;' ><a style='text-align: center;color: black;'href="https://www.github.com//YoussefAmdouni" target="_blank">Github</a></h1>""", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
