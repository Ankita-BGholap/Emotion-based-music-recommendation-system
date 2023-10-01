from flask import Flask, render_template, Response ,request
import cv2
import numpy as np
import statistics as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


app = Flask(__name__)

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    
    model=load_model('C:/Users/Ankita Gholap/Desktop/MoodTune/model_file_30epochs.h5')

    video=cv2.VideoCapture(0)

    faceDetect=cv2.CascadeClassifier('C:/Users/Ankita Gholap/Desktop/MoodTune/haarcascade_frontalface_default.xml')

    labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

    output=[]
    i=0
    while (i<=30):
        ret,frame=video.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces= faceDetect.detectMultiScale(gray, 1.3, 3)


        for x,y,w,h in faces:
            sub_face_img=gray[y:y+h, x:x+w]
            resized=cv2.resize(sub_face_img,(48,48))
            normalize=resized/255.0
            reshaped=np.reshape(normalize, (1, 48, 48, 1))
            result=model.predict(reshaped)

            label=np.argmax(result, axis=1)[0]

            predicted_emotion = labels_dict[label]
            output.append(predicted_emotion)

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        i = i+1

        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    print(output)        
    video.release()
    cv2.destroyAllWindows()
    final_ot1 = st.mode(output)
    print (final_ot1)
    return render_template("index5.html",final_output1=final_ot1)


if __name__ == '__main__':
    app.debug = True
    app.run() 