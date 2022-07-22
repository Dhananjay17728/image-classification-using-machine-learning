import streamlit as st
import numpy as np
from skimage.transform import resize
import pickle
from PIL import Image
st.title("Image Classifier")
st.text("upload the image")
model=pickle.load(open('img_model.pkl','rb'))
upload_file=st.file_uploader('choose an image ...', type='jpg')
if upload_file is not None:
    img=Image.open(upload_file)
    st.image(img,caption='Upload Image')
    if st.button('PREDICT'):
        flat_data=[]
        Categories=['pretty sunflower','rugby ball leather','ice cream cone']
        st.write('Result...')
        img=np.array(img)
        img_resized=resize(img,(150,150,3))
        flat_data.append(img_resized.flatten())
        flat_data=np.array(flat_data)
        print(img.shape)
        y_out=model.predict(flat_data)
        y_out=Categories[y_out[0]]
        st.title(f'PREDICTED OUTPUT:{y_out}')
        q=model.predict_proba(flat_data)
        for index,item in enumerate(Categories):
            st.write(f'{item}:{q[0][index]*100}%')