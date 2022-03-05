#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
app = Flask (__name__)
from flask import request, render_template
import joblib

@app.route("/", methods=["GET", "POST"])

def index():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        income2 = float(income)
        age2 = float(age)
        loan2 = float(loan) 
        print(income2, age2, loan2)
        
        model1 = joblib.load("CCD_DT")
        pred1 = model1.predict([[income2, age2, loan2]])
        s1 = "The credit card default based on Decision Tree is : " + str(pred1)
        
        model2 = joblib.load("CCD_REG")
        pred2 = model2.predict([[income2, age2, loan2]])
        s2 = "The credit card default based on Linear Regression Model is : " + str(pred2)
        
        model3 = joblib.load("CCD_NN")
        pred3 = model3.predict([[income2, age2, loan2]])
        s3 = "The credit card default based on Neural Network Model is : " + str(pred3)
        
        model4 = joblib.load("CCD_RF")
        pred4 = model4.predict([[income2, age2, loan2]])
        s4 = "The credit card default based on Random Forest Model is : " + str(pred4)
        
        model5 = joblib.load("CCD_GB")
        pred5 = model5.predict([[income2, age2, loan2]])
        s5 = "The credit card default based on XGBoost Model is : " + str(pred5) 
        
        return(render_template("index.html", result1=s1, result2=s2, result3=s3, result4=s4, result5=s5 ))        
    else:
        return(render_template("index.html", result1="2", result2="2", result3="2", result4="2", result5="2" ))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




