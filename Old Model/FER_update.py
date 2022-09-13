#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy


# In[2]:


img_array =cv2.imread("Desktop/jaffe10test/jaffedbase/train/ANGRY/KA.AN1.39.tiff")


# In[3]:


img_array.shape 


# In[4]:


plt.imshow(img_array) #BGR


# In[5]:


Datadirectory ="Desktop/jaffe10test/jaffedbase/train/"  #training dataset


# In[6]:


Classes =["ANGRY","disgust","FEAR","HAPPY","NEUTRAL","SAD","SURPRISE"]  #List of classes 


# In[7]:


for category in Classes:
    path=os.path.join(Datadirectory,category)
    for img in os.listdir(path):
        img_array =cv2.imread(os.path.join(path,img))
        #backtorgb=cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
        plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break    


# In[8]:


img_size=224   #ImageNet==>224,224
new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
plt.show()


# In[9]:


new_array.shape #why we convert becasue of transfer learning , TF takes 224*224 as input


# ### Read all images & convert them into array

# In[10]:


training_Data =[]  #data array

def create_training_Data():
    for category in Classes:
        path=os.path.join(Datadirectory,category)
        class_num=Classes.index(category)   ##Label
        for img in os.listdir(path):
            try:
                img_array =cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass


# In[11]:


create_training_Data()


# In[12]:


print(len(training_Data))


# In[13]:


import random

random.shuffle(training_Data)


# In[14]:


import numpy as np
X=[] ##data/feature
y=[]  ## label

for features,label in training_Data:
    X.append(features)
    y.append(label)
    
    
X =np.array(X).reshape(-1,img_size,img_size,3)  ## converting into 4 dimension    


# In[15]:


X.shape


# In[16]:


## Normalize the data
X=X/255.0;   #we are normalizing it


# In[17]:


y[0]  #these are labels ,showing 2 because it has random number


# In[18]:


y[0]


# In[19]:


Y= np.array(y)


# In[20]:


## Deep Learning model for training -Transfer Learning


# In[21]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[22]:


model=tf.keras.applications.MobileNetV2() # Using mobilenet bcz its light weight


# In[23]:


model.summary() #4.2M trained parameters


# ## Transfer Learning- Tuning, weights will start from last check point

# In[24]:


base_input=model.layers[0].input


# In[25]:


base_output=model.layers[-2].output #last 2 layers are neglected


# In[26]:


final_output =layers.Dense(128)(base_output)  #adding new layer , after output of Global pooling layer
final_output =layers.Activation('relu')(final_output) #activation function
final_output =layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation ='softmax')(final_output) #my classes my 7


# In[27]:


final_output


# In[28]:


new_model = keras.Model(inputs =base_input, outputs = final_output)


# In[29]:


new_model.summary()


# In[30]:


new_model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[55]:


new_model.fit(X,Y, epochs=50)


# In[56]:


new_model.save('Final_model_001.h5')


# In[57]:


new_model =tf.keras.models.load_model('Final_model_001.h5')


# In[116]:


frame=cv2.imread("happyboy2.jpg")


# In[117]:


frame = numpy.array(frame)


# In[118]:


frame.shape


# In[119]:


plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


# In[120]:


## We need face detection algorithm


# In[121]:


faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[122]:


gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# In[123]:


gray.shape


# In[124]:


faces =faceCascade.detectMultiScale(gray,1.1,4)  ##open cv code, find all possible faces in a photo, and can also detect multiple faces
for x,y,w,h in faces:
    roi_gray =gray[y:y+h,x:x+w]
    roi_color =frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x,y), (x+w,y+h), (225,0,0), 2)
    facess=faceCascade.detectMultiScale(roi_gray)
    if len(faces) ==0:
        print("Face not detected")
    else:
        for (ex,ey,ew,eh) in facess:
            face_roi =roi_color[ey: ey+eh, ex:ex +ew]


# In[125]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[126]:


plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))


# In[127]:


import numpy as np
final_image=cv2.resize(face_roi,(224,224))  ##face_roi is rgb
final_image =np.expand_dims(final_image,axis=0)  #need 4th dimensions
final_image=final_image/255.0  ##normalising it


# In[128]:


Predictions=new_model.predict(final_image)


# In[129]:


Predictions[0]


# In[130]:


np.argmax(Predictions)


# ## Real time video Demo

# In[132]:


import cv2 

path="Desktop/jaffe10test/jaffedbase/train/"
font_scale =1.5
font=cv2.FONT_HERSHEY_PLAIN

#SET the rectangle background to white
rectangle_bgr =(225,225,225)

#make a black image
img=np.zeros((500,500))

#set same text
text="Some text in a box!"

#get the width and height of the text box
(text_width, text_height)=cv2.getTextSize(text,font,fontScale=font_scale,thickness=1)[0]

#set the text start position
text_offset_x= 10
text_offset_y=img.shape[0] - 25

#make the cords of the box with a small padding of two pixels
box_coords=((text_offset_x,text_offset_y), (text_offset_x+ text_width + 2, text_offset_y - text_height -2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x,text_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=1)

cap=cv2.VideoCapture(1)

#check if the webcam is opened correctly

if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    
while True:
    ret,frame=cap.read()
    #eye_Cascade =cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faces =faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray =gray[y:y+h, x:x+w]
        roi_color =frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (225,0,0), 2)
        facess=faceCascade.detectMultiScale(roi_gray)
        if len(facess)==0:
            print("Face not detected")
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey:ey+eh, ex:ex+ew]   #croping the face
     
    
    final_image =cv2.resize(face_roi, (224,224))
    final_image =np.expand_dims(final_image, axis=0) #need fourth dimension
    final_image=final_image/255.0
    
    font =cv2.FONT_HERSHEY_SIMPLEX
    
    Predictions =new_model.predict(final_image)
    
    font_scale =1.5
    font =cv2.FONT_HERSHEY_PLAIN
    
    if (np.argmax(Predictions)==0):
        status ="Angry"
        
        x1,y1,w1,h1 = 0,0,175,75
        
        #Draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,225), 2)
        cv2.putText(frame, status,(100,150), font, 3, (0,0,225),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
        
    elif (np.argmax(Predictions)==1):
        status ="Disgust"
        
        x1,y1,w1,h1 = 0,0,175,75
        
        #Draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,225), 2)
        cv2.putText(frame, status,(100,150), font, 3, (0,0,225),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
        
    elif (np.argmax(Predictions)==2):
        status ="Fear"
        
        x1,y1,w1,h1 = 0,0,175,75
        
        #Draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,225), 2)
        cv2.putText(frame, status,(100,150), font, 3, (0,0,225),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))    
            
    elif (np.argmax(Predictions)==3):
        status ="Happy"
        
        x1,y1,w1,h1 = 0,0,175,75
        
        #Draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,225), 2)
        cv2.putText(frame, status,(100,150), font, 3, (0,0,225),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
        
    elif (np.argmax(Predictions)==4):
        status ="Neutral"
        
        x1,y1,w1,h1 = 0,0,175,75
        
        #Draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,225), 2)
        cv2.putText(frame, status,(100,150), font, 3, (0,0,225),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
        
    elif (np.argmax(Predictions)==5):
        status ="Sad"
        
        x1,y1,w1,h1 = 0,0,175,75
        
        #Draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,225), 2)
        cv2.putText(frame, status,(100,150), font, 3, (0,0,225),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
        
    elif (np.argmax(Predictions)==6):
        status ="Surprise"
        
        x1,y1,w1,h1 = 0,0,175,75
        
        #Draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,225), 2)
        cv2.putText(frame, status,(100,150), font, 3, (0,0,225),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
        
        
        
    else:
        status ="No emotions"
        
        x1,y1,w1,h1 = 0,0,175,75
        
        #Draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,225,0), 2)
        cv2.putText(frame, status,(100,150), font, 3, (0,225,0),2,cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,225,0))
        
        
    #gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    #faces = faceCascade.detectMultiScale(gray, 1.1,4)
    
    #draw a rectangle around the faces
    #for(x,y,w,h) in faces:
    #    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    
    
    
    #Use putText() method for
    #Inserting text on video
    
    cv2.imshow('Face Emotion Recognition', frame)
    
    if cv2.waitKey(2) & 0xFF ==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()        


# In[ ]:




