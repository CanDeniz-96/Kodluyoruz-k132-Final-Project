"""this .py file can be used to test with new data.
To test, after clicking 'app.py', please go to the link below:
http://localhost:8080/docs
or you can write ip address as:
http://127.0.0.1:8080/docs
Next, select "rfc_predict" or "dtc_predict".
Click to "Try it out"
Write your commend you want to try and click to "Execute" button.
The last step is waiting while the page is loading

NOTE: To stop this process, pres 'ctrl + c'"""

#import required libraries
from fastapi import FastAPI
from fastapi import Form
import uvicorn
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer


#Initializing a FastApi app
app = FastAPI()





def vectorize(text:str):
    """A string text will be converted a 512 dimensions vector"""
    vectorizer = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    return vectorizer.encode(text)


# Setting get and post requests.
@app.get('/')
@app.post('/rfc_predict')
@app.post('/dtc_predict')

def root():
    return """Well Come. To start test, go to http://127.0.0.1:8080/docs"""


async def rfc_predict(comment: str = Form(...)):
    """comment: str: is a comment text."""
    print(comment)
    #Loading the rfc_model from a pickle file.
    model = pickle.load(open("YemekSepeti_rfc_model.sav", "rb"))

    #Vectorizing the comment text to be processed
    comment_vector = vectorize(comment)

    #Prediction
    prediction = model.predict([comment_vector])

    #Due to the FastApi types, numpy array or numpy object should be converted str or dict.
    return {"Category" : str(prediction[0])}



async def dtc_predict(comment: str =Form(...)):
    """comment: str: is a comment text."""
    print(comment)

    #Loading the DecissionTree model
    dtc_model = pickle.load(open('YemekSepeti_dtc_model.sav', 'rb'))
    #Vectorizing the comment text to be processed
    comment_vector = vectorize(comment)

    #Prediction
    prediction = str(dtc_model.predict([comment_vector])[0])
    return {'Category for Decission Tree Classifier': prediction}

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8080)



