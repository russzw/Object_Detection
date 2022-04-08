import streamlit as st
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import cv2

# from keras.models import load_model
# load model
model = InceptionV3()


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
        label = ("{}: {:.2f}%".format(label, prob * 100))
    st.markdown(label)


def secpred(frame, model):
    # pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # predict using the Inceptionv3 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        pred_class = label

    return pred_class


def obj_det(search, frame, model):
    label = secpred(frame, model)
    label = label.lower()
    if label.find(search) > -1:
        st.image(frame, caption=label)
        #return sys.exit()
    else:
        pass
        # st.text('Not Found')
        # return sys.exit()


def main():
    st.title('Object Detection 1.0')

    # upload video
    vid_file = st.file_uploader("Upload Video", type=['mp4', 'mkv', 'avi'])

    if vid_file is not None:
        path = vid_file.name
        with open(path, mode="wb") as f:
            f.write(vid_file.read())
            st.success("File Saved")
        # capture video
        cap = cv2.VideoCapture(path)
        i = 0

        if st.button("Detect"):

            while (cap.isOpened()):
                ret, frame = cap.read()

                if ret == False:
                    break

                path2 = './frames/fr' + str(i) + ' .jpg'
                cv2.imwrite(path2, frame)

                img_path = path2
                img = image.load_img(img_path, target_size=(299, 299))
                predict(img, model)

                i += 1

            cap.release()
            # output.release()
            cv2.destroyAllWindows()

        key = st.text_input('Search key')
        key = key.lower()

        if key is not None:

            if st.button("Search for an object"):

                # Start the video prediction loop
                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret:
                        break

                    # Perform object detection
                    obj_det(key, frame, model)

                cap.release()
                #output.release()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

st.header("created by Russell and Brenda")