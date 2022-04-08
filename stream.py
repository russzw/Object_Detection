import streamlit as st
import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import cv2
from keras.models import load_model

model= load_model('inception.h5', compile=(False))


def predict(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    p = decode_predictions(preds, top=1)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    # print('Predicted:', decode_predictions(preds, top=1)[0])
    for (i, (imagenetID, label, prob)) in enumerate(p[0]):
        label= ("{}: {:.2f}%".format(label, prob*100))
    st.markdown(label)






def main():

    st.title('Object Detection 1.0')

#upload video
    vid_file = st.file_uploader("Upload Video",type=['mp4','mkv','avi'])

    if vid_file is not None:
        path = vid_file.name
        with open(path, mode="wb") as f:
            f.write(vid_file.read())
            st.success("File Saved")
    #capture video
        cap = cv2.VideoCapture(path)
        i = 0

        if st.button("Detect"):

            while(cap.isOpened()):
                ret, frame = cap.read()

                if ret == False:
                    break

                path2 ='./frames/fr'+ str(i)+ ' .jpg'
                cv2.imwrite(path2,frame)

                img_path = path2
                img = image.load_img(img_path, target_size=(299, 299))
                predict(img, model)

                i+=1

    

        cap.release()
        #output.release()
        cv2.destroyAllWindows()




if __name__=='__main__':
    main()