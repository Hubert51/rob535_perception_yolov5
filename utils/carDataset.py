import rob535
import cv2
YOLO = 1

if __name__ == '__main__':
    # read data from numpy file and for training
    # rob535.data.preprocess_dataset(cuda=True, )

    # pre-proprocess data for YOLO
    # YOLO is 1
    rob535.data.generate_data(path="/home/ruijie/Desktop/rob535_perception_yolov5/data", framework=YOLO)



    # print(1)