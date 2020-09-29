# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 02:20:31 2020

@author: Nithin
"""

# -*- coding: utf-8 -*-
"""




@author: Nithin
"""



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

from flask import Flask,request
import numpy as np
import pickle
import streamlit as st
import pandas as pd
from flasgger import Swagger
from Model_Pickle import *
import base64
import io

from PIL import Image

app=Flask(__name__)
Swagger(app)

pickle_in = open("model_classifier.pkl","rb")
classifier=pickle.load(pickle_in)

threshold=50
defaultBalance=1000

category_pickle = open("category_pickle.pkl","rb")
category_data=pickle.load(category_pickle)
print(category_data)
Category=dict.fromkeys(category_data,0)

#Category={'Grocery':0,'Health':0,'Food and Drinks':0,'Transport':0}
details = {'HB000010': {'amount':defaultBalance,'Category':Category}, 'HB000011': {'amount':defaultBalance,'Category':Category}, 'HB000012': {'amount':defaultBalance,'Category':Category}, 'HB000013': {'amount':defaultBalance,'Category':Category}}

pickle_in = open("model_details.pkl","rb")
details=pickle.load(pickle_in)

def reset_data():
    category_pickle = open("category_pickle.pkl","rb")
    category_data=pickle.load(category_pickle)
    print(category_data)
    Category=dict.fromkeys(category_data,0)
    details = {'HB000010': {'amount':defaultBalance,'Category':Category}, 'HB000011': {'amount':defaultBalance,'Category':Category}, 'HB000012': {'amount':defaultBalance,'Category':Category}, 'HB000013': {'amount':defaultBalance,'Category':Category}}
    pickle_out = open("model_details.pkl","wb")
    pickle.dump(details, pickle_out)
    pickle_out.close()
    
def save_data():
    pickle_out = open("model_details.pkl","wb")
    pickle.dump(details, pickle_out)
    pickle_out.close()
    
    

def getRewardCategory(UserID):
    max=0
    maxKey=''
    for key in details[UserID]['Category']:
        if(details[UserID]['Category'][key]>=max):
            max=details[UserID]['Category'][key]
            maxKey=key
    print("getRewardCategory"+maxKey)
    return maxKey

def giveReward(category_reward):
    str=''
    if(category_reward=='Grocery'):
        str="Woolworths 30$"
    elif(category_reward=='Health'):
        str="Priceline 30$"
    elif(category_reward=='Food and Drinks'):
        str="Mcdonalds 30$"
    elif(category_reward=='Transport'):
        str="Fuel Caltex 30$"
    else:
        str=category_reward+" Reward 30$"
    print("Reward id"+str+category_reward)
    return str
        

def updateTransaction(UserID,prediction,typeTransaction,amount):
    print("Prediction"+str(details))
    reward=''
    rewardEligible=False
    if(typeTransaction=='Debit'):      
        details[UserID]['amount']=details[UserID]['amount']-amount
        #Add amount to category
        details[UserID]['Category'][prediction]=details[UserID]['Category'][prediction]+amount
        #details[UserID]['Category']['Total']=details[UserID]['Category']['Total']+amount
        if(threshold>details[UserID]['amount']):
            #Give Reward
            category_reward=getRewardCategory(UserID)
            reward=giveReward(category_reward)
            rewardEligible=True

    else:
        details[UserID]['amount']=details[UserID]['amount']+amount
    return rewardEligible,reward
    
    
@app.route('/')
def welcome():
    return "Welcome All"

def retrain_my_model(file):
    return retrain_model(file)


@app.route('/predict',methods=["Get"])
def predict_category(UserID,transaction,typeTransaction,amount):

    Result="No Result"
    amount=int(amount)
    prediction=classifier.predict([transaction])
    rewardEligible,reward=updateTransaction(UserID,prediction[0],typeTransaction,amount)
    if(rewardEligible==True):
        Result=("Predicted Category for user " + UserID +" is " + str(prediction[0]))
        st.success(Result)
        st.warning("You have a new Reward:"+reward)
        st.info(("Balance Amount:"+str(details[UserID]['amount'])))
        print(details[UserID]['Category'].items())
        st.bar_chart(pd.DataFrame(details[UserID]['Category'], index=[0]).transpose())
        
    else:
        Result=("Predicted Category for user " + UserID +" is " + str(prediction[0]))
        st.success(Result)
        st.info("Balance Amount:"+str(details[UserID]['amount']))

    save_data()
    return str(Result)
import os
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def get_table_download_link(file):
    df= pd.read_csv(file)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="TrainingData_Default.csv">Download Sample Training File </a>'




def main():
    menu= ["Predict","Re-train"]
    st.title("Bank Transaction Simulator")
    html_retrain = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Category Training Dashboard </h2>
        </div>
        """
    html_predict = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Predict </h2>
        </div>
        """
    
    choice= st.sidebar.selectbox("Menu",menu)
    if choice == "Re-train":
        st.markdown(html_retrain,unsafe_allow_html=True)
        
        st.markdown(get_table_download_link('TrainingData_Default.csv'), unsafe_allow_html=True)
        
                
        st.set_option('deprecation.showfileUploaderEncoding', False)
        csv_file_buffer = st.file_uploader("Upload training file", type=["csv"]) 
        try:
            if st.button("Train using custom dataset"):
                retrain_accuracy=retrain_my_model(pd.read_csv(csv_file_buffer))
                reset_data()
                st.success(retrain_accuracy)
        except:
            st.error("Please upload valid CSV file")
            
        if st.button("Train using built-in dataset"):
                retrain_accuracy=retrain_my_model(pd.read_csv('TrainingData_Default.csv'))
                reset_data()
                st.success(retrain_accuracy)
            
        
                
        
            

            
    else:
        st.markdown(html_predict,unsafe_allow_html=True)
        UserID=st.selectbox('UserID',('HB000010', 'HB000011', 'HB000012','HB000013'))
        transaction = st.text_input("Transaction","Type Here")
        typeTransaction=st.selectbox('Transaction Type',('Debit', 'Credit'))
        amount = st.number_input("Amount",1)
    
      
        result=""
        if st.button("Simulate"):
            result=predict_category(UserID,transaction,typeTransaction,amount)
        #st.success('The output is {}'.format(result))
        
        if st.button("Reset Data"):
            st.text("Data Reset Done")
            reset_data()
            st.balloons()

            
      

    
     

if __name__=='__main__':
    main()
    

    
