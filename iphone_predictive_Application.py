from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import os
nltk.download('stopwords')



app=Flask(__name__)
Swagger(app)


@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_Customer_Voice():
    
    """Let's understand the voice of customer 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: comment
        in: query
        type: string
        required: true

    responses:
        200:
            description: The output values
        
    """
    
    
    model=tf.keras.models.load_model('iphone_lstm_model.h5')
    model1=tf.keras.models.load_model('iphone_Sentiment_BI_lstm_model.h5')
    voc_size=5000
    sent_length=20
    comment=request.args.get("comment")

    print(comment)
    ps=PorterStemmer()
    corpus=[]

    review = re.sub('[^a-zA-Z]', ' ', comment)
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    comment=' '.join([str(elem)for elem in corpus])
    
    
    onehot_repr=[one_hot(comment,voc_size)] 
   

    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

    pred=model.predict(embedded_docs) 
    
    b=([np.argmax(pred)])
    
    reps={0:'Battery',1:'Affordability',2:'Build_Quality',3:'Storage',4:'Camera',5:'Innovation',6:'Others'}

    c=[reps.get(x,x) for x in b]
    
    print(c)
    
    pred_sent=(model1.predict(embedded_docs)).tolist()
    
    pred1_sent=float(str(pred_sent).replace('[','').replace(']',''))
    
    if pred1_sent >0.5:
        val='Negative'
    else : val='Positive/Neutral'
    print(val)
    
    return "Customer Talking About "+str(c) + " ; " + " Customer Sentiment " + "  - " + str(val)
    


@app.route('/predict_file',methods=["POST"])
def predict_Customer_Voice_file():
    """Let's understand the voice of customer 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    
    model=tf.keras.models.load_model('iphone_lstm_model.h5')
    model1=tf.keras.models.load_model('iphone_Sentiment_BI_lstm_model.h5')
    print(model.summary())
    
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test)
    
    voc_size=5000
    sent_length=20

    ps=PorterStemmer()
    corpus1=[]
    
    for i in range(0,len(df_test)):
        print(i)

        review = re.sub('[^a-zA-Z]', ' ', df_test['review'][i])
        review = review.lower()
        review = review.split()
    
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus1.append(review)
    
    
    onehot_repr1=[one_hot(words,voc_size)for words in corpus1] 

    embedded_docs1=pad_sequences(onehot_repr1,padding='pre',maxlen=sent_length)

    print(embedded_docs1)
    
    prediction=model.predict_classes(embedded_docs1) 
    
    d=list(prediction)
    
    reps={0:'Battery',1:'Affordability',2:'Build_Quality',3:'Storage',4:'Camera',5:'Innovation',6:'Others'}

    e=[reps.get(x,x) for x in d]
    
    print(e)
    
    pred_sent1=(model1.predict(embedded_docs1)).tolist()
    
    df1=pd.DataFrame(pred_sent1)
    
    df1.rename({0:'prob'},axis=1,inplace=True)
    
    df1['sentiment']=np.where(df1.prob>0.5,'Negative','Positive/Neutral')

    return {'Customer Sentiment' : str(list(df1['sentiment'])), 'Customer Talking About' : str(list(e))} 
   


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
    


