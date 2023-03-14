from flask import Flask , render_template , request
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np



app = Flask(__name__)

scaler = StandardScaler()
model = pickle.load(open('LogisticsFireForest.pkl','rb'))

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/getvalues' , methods=['POST','GET'])
def getvalues():
    data = [int(i) for i in request.form.values()]
    print(data)
    print(type(data[0]))
    newdata = np.array(data).reshape(1,-1)
    # final_data = scaler.fit_transform(newdata)
    prediction = model.predict(newdata)
    probab = model.predict_proba(newdata)

    return render_template('pass.html',pred = prediction,p = probab)


if __name__ == '__main__' :
    app.run(debug=True)
#


# from flask import Flask,request, url_for, redirect, render_template
# import pickle
# import numpy as np
# from sklearn.preprocessing import StandardScaler
#
# app = Flask(__name__)
#
# model=pickle.load(open('LogisticsFireForest.pkl','rb'))
# scaler = StandardScaler()
#
#
# @app.route('/')
# def hello_world():
#     return render_template("index.html")
#
#
# @app.route('/getvalues',methods=['POST','GET'])
# def getvalues():
#     int_features=[int(x) for x in request.form.values()]
#     final=[np.array(int_features)]
#     print(int_features)
#     print(final)
#     # final = scaler.fit_transform(final)
#     prediction=model.predict_proba(final)
#     # final_pred = model.predict(final)
#     output='{0:.{1}f}'.format(prediction[0][1], 2)
#
#     if output>str(0.5):
#         return render_template('pass.html' ,pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
#     else:
#         return render_template('pass.html' ,pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
