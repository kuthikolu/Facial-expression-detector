import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  





import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('fer2013_mini_XCEPTION.119-0.65.hdf5', compile=False)

print("Model Loaded")


emotion_labels = ['Angry', 'Discust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        
        
        predictions = model.predict(roi_gray)
        emotion_index = np.argmax(predictions)
        emotion_text = emotion_labels[emotion_index]

        
        
        
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
