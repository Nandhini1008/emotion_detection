import os
import cv2
import numpy as np
from keras.models import model_from_json


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

path = os.path.join("e_d", "Model", "emotion_model.json")
json_file = open(path, 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

path1 = os.path.join("e_d", "Model", "emotion_model.weights.h5")
emotion_model.load_weights(path1)
print("Loaded model from disk")


cap = cv2.VideoCapture(0)


while True:
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    path2 = os.path.join("e_d", "Haarcascades", "haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(path2)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    num_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

  
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
