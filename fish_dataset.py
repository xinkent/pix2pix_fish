import numpy as np
from PIL import Image

from io import BytesIO


def load_dataset(dataDir='./dataset/train_data/', data_range=range(0,300)):
        print("load dataset start")
        print("     from: %s"%dataDir)
        imgDataset = []
        labelDataset = []

        imgStart = 12313
        labelStart = 1
        for i in data_range:
            imgNum = imgStart + int(i*(29/10))
            labelNum = i + 1
            img = Image.open(dataDir + "GP029343_%06d.png"%imgNum)
            label = Image.open(dataDir + "2017-03-02_105804_%d_width_773_height_1190.png"%labelNum)
            label = label.convert("L")
            w,h = img.size
            r =  300/min(w,h)
            img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
            label = label.resize((int(r*w), int(r*h)),Image.BILINEAR)


            img = np.asarray(img).astype("f")/128.0-1.0

            label = np.asarray(label)/128.0-1.0
            label = label[:,:,np.newaxis]

            img_h,img_w,_ = img.shape
            # label_h, label_w, _ = label.shape
            xl = np.random.randint(0,img_w-256)
            yl = np.random.randint(0,img_h-512)
            # label_xl = np.random.randint(0,label_w-256)
            # label_yl = np.random.randint(0,label_h-256)
            img = img[yl:yl+512, xl:xl+256,:]
            # label = label[label_yl:label_yl+256, label_xl:label_xl+256]
            label = label[yl:yl+512, xl:xl+256,:]
            imgDataset.append(img)
            labelDataset.append(label)

        print("load dataset done")
        return np.array(imgDataset),np.array(labelDataset)
