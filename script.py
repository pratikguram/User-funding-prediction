#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# prediction function 
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 5) 
    loaded_model = pickle.load(open("model.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0]
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='User will fund'
        else: 
            prediction ='User will not fund'            
        return render_template("result.html", prediction = prediction) 

