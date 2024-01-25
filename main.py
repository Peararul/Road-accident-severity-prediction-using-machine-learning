from flask import Flask, render_template, request
import joblib
import os
import  numpy as np
import pickle

app= Flask(__name__)
model = joblib.load(open('decisiontree.pkl', 'rb'))
picfolder=os.path.join('static','pics')
app.config['UPLOAD_FOLDER']=picfolder

@app.route("/")
def home():
    return render_template("home.html")
@app.route("/Prediction")
def index():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result",methods=['POST','GET'])
def result():


    age_of_driver = int(request.form['age_of_driver'])
    vehicle_type = int(request.form['vehicle_type'])
    age_of_vehicle = int(request.form['age_of_vehicle'])
    engine_cc = int(request.form['engine_cc'])
    day = int(request.form['day'])
    weather = int(request.form['weather'])
    light = int(request.form['light'])
    roadsc = int(request.form['roadsc'])
    gender = int(request.form['gender'])
    speedl = int(request.form['speedl'])
    x = np.array([age_of_driver, vehicle_type, age_of_vehicle, engine_cc, day, weather, roadsc, light, gender, speedl]).reshape(1, -1)

    scaler_path = os.path.join('C:/Users/Dell/Road accident severity prediction final 1', 'models/scaler.pkl')
    scaler = None
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    x = scaler.transform(x)
    #model_path = os.path.join('C:/Users/Dell/Road accident severity prediction 11', 'models/lr.sav')
    #model = joblib.load(model_path)
    #x=np.array([Did_Police_Officer_Attend, age_of_driver, vehicle_type, age_of_vehicle, engine_cc, day, weather, roadsc,light, gender, speedl]).reshape(1,-1)
    result = model.predict(x)
    #result = model.predict(x)

    #return render_template('index.html', pred=str(result))
    # for No Stroke Risk
    if result==1:
        return render_template('one.html', pred=str(result))
    elif result==2:
        return render_template('second.html', pred=str(result))
    else:
        return render_template('three.html', pred=str(result))
@app.route("/Exploration")
def Exploration():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'],'image1.jpg')
    imageList = os.listdir('static/pics')
    imageList = ['pics/' + image for image in imageList]
    return render_template("visualization.html", imageList=imageList)



if __name__=="__main__":
    app.run(debug=True,port=7384)