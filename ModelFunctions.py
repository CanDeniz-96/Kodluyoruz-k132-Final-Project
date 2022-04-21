
#importing required libraries
import pandas as pd
import numpy as np
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pickle





# Reading to comments from a csv file and preprocessing
data = pd.read_csv("comments.csv", index_col=False)
data.dropna(axis=0, inplace=True)

li = list(data['comment'])

#Stopwords are not important words for meaningful sentence without "az" for this task so we'll remove these words.
wpt = WordPunctTokenizer()
stop_word_list = [i for i in stopwords.words("turkish") if not i == "az"]


#Remove punctuations, stopwords, numbers; lowering letters with "I" and "İ"
for i in range(len(li)):
    if isinstance(li[i], str):
        for j in li[i]:
            if (j in string.punctuation) or (j.isdigit()):
                li[i] = li[i].replace(j, "")
        li[i] = li[i].replace("İ", "i")
        li[i] = li[i].replace("I", "ı")
        li[i] = li[i].lower()
        li[i] = " ".join([k for k in li[i].split() if not k in stop_word_list])
    

#Assign a new list to DataFrame object.
data.drop(columns="comment", axis=1, inplace=True)
data['Comments'] = li



#Creating sentence vector by using sentence_transformers.SentenceTransformer.
#At the same time, vectorizer_fit is also x values for train.

vectorizer = SentenceTransformer('distiluse-base-multilingual-cased-v1')
vectorizer_fit = vectorizer.encode(li)

y = data['means']

x_train, x_test, y_train, y_test = train_test_split(vectorizer_fit, y, test_size=0.2, random_state=42)

#Saving function to model as a .sav file

def save_model(model_name, model_file_name: str):
    """Model name is the type of the model your use, equals to model.fit variable
    Model file name should be given without .sav"""
    pickle.dump(model_name, open(model_file_name+".sav", 'wb'))

#Logistic Regression function:
def Logistic_Regression():
    lr = LogisticRegression()
    lr_model = lr.fit(x_train, y_train)
    lr_pred = lr.predict(x_test)
    print(accuracy_score(lr_pred, y_test), confusion_matrix(y_test, lr_pred), sep="\n")
    save_model(lr_model, "YemekSepeti_LogisticRegression")

#Random Forest model Function:
def RandomForest():
    rfc = RandomForestClassifier(n_estimators=50)
    rfc_model = rfc.fit(x_train, y_train)
    rfc_pred = rfc.predict(x_test)
    print(accuracy_score(y_test, rfc_pred), confusion_matrix(rfc_pred, y_test), sep="\n")
    save_model(rfc_model, "YemekSepeti_rfc_model")

#DecissionTree function:
def decission_tree():
    dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dtc_model = dtc.fit(x_train, y_train)
    dtc_pred = dtc.predict(x_test,)
    print(accuracy_score(dtc_pred, y_test))
    print(confusion_matrix(y_test, dtc_pred))
    save_model(dtc_model, "YemekSepeti_dtc_model")


    
RandomForest()
Logistic_Regression()
decission_tree()
