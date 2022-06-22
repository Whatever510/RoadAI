import numpy as np
from tensorflow.keras.models import load_model
import cv2

COLORS = [
    [178, 178, 178],  # 0 = background
    [0, 255, 0],      # 1 = drivable
    [255, 0, 0],      # 2 = non-drivable
    [0, 0, 255],      # 3 = hood
    [255, 255, 0],    # 4 = dashed centerline
    [0, 255, 255],    # 5 = lanemarking
    [127, 0, 255],    # 6 = parking
    [255, 0, 255],    # 7 = double solid
    [127, 127, 255],  # 8 = stopline
    [255, 0, 127],    # 9 = parking lane
    [255, 127, 255],  # 10 = obstacle
]

#############################################################################
# # SET IMAGE RESOLUTION AND VIDEO NAME

WIDTH, HEIGHT = 224, 224
VIDEO_NAME = ""
MODEL_NAME = 'unet_vgg16.h5'

#############################################################################

def main():
    model = load_model(MODEL_NAME)
    # open video file
    cap = cv2.VideoCapture(VIDEO_NAME)


    while cap.isOpened():
        ret, frame =  cap.read()
        if ret:
            # resize frame
            frame = cv2.resize(frame, (224, 224))
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            test_image = np.expand_dims(frame, axis=0)
            pred = model.predict(test_image)

            pred = np.argmax(pred, axis=-1)
            pred = pred[0]
            pred = pred.astype(np.uint8)
            pred = cv2.resize(pred, (WIDTH, HEIGHT))

            # create color image based on the prediction
            color_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            for i in range(len(COLORS)):
                color_image[pred == i] = COLORS[i]

            # overlay the prediction on the original frame
            overlay = cv2.addWeighted(frame, 0.7, color_image, 0.3, 0)

            # resize to original size
            overlay = cv2.resize(overlay, (640, 480))

            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            # display the original and overlayed frame
            cv2.imshow('frame', overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    

    cap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()