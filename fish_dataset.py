import numpy as np
from PIL import Image

from io import BytesIO


def load_dataset2(dataDir='/data1/train_data1/', data_range=range(0,300),skip=True):
        print("load dataset start")
        print("     from: %s"%dataDir)
        imgDataset = []
        clabelDataset = []

        imgStart = 1
        clabelStart = 1

        for i in data_range:
            if skip:
                if i%3 != 0:
                    continue
            imgNum = imgStart + i
            clabelNum = clabelStart + i
            img = Image.open(dataDir + "up/up%05d.png"%imgNum)
            label_color = Image.open(dataDir + "up_25night/night_up%05d.png"%clabelNum)

            w,h = img.size
            r = 300/min(w,h)
            img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
            label_color = label_color.resize((int(r*w), int(r*h)),Image.BILINEAR)

            img = np.asarray(img)/128.0-1.0
            label_color = np.asarray(label_color)/128.0-1.0

            h,w,_ = img.shape
            xl = np.random.randint(0,w-256)
            yl = np.random.randint(0,h-512)
            img = img[yl:yl+512, xl:xl+256, :]
            label_color = label_color[yl:yl+512, xl:xl+256,:]

            imgDataset.append(img)
            clabelDataset.append(label_color)


        print("load dataset done")
        return np.array(imgDataset),np.array(clabelDataset)


def load_dataset2(dataDir='/data1/train_data1/', data_range=range(0,300),skip=True):
        print("load dataset start")
        print("     from: %s"%dataDir)
        imgDataset = []
        clabelDataset = []
        slabelDataset = []

        imgStart = 1
        slabelStart = 1
        clabelStart = 1

        for i in data_range:
            if skip:
                if i%3 != 0:
                    continue
            imgNum = imgStart + i
            slabelNum = slabelStart + i
            clabelNum = clabelStart + i
            img = Image.open(dataDir + "up/up%05d.png"%imgNum)
            label_color = Image.open(dataDir + "up_25night/night_up%05d.png"%clabelNum)
            label_sonar = Image.open(dataDir + "sonar/" + "2017-03-02_105804_%d_width_773_height_1190.png"%slabelNum)
            label_sonar = label_sonar.convert("L")

            w,h = img.size
            r = 300/min(w,h)
            img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
            label_color = label_color.resize((int(r*w), int(r*h)),Image.BILINEAR)
            label_sonar = label_sonar.resize((int(r*w), int(r*h)),Image.BILINEAR)

            img = np.asarray(img)/128.0-1.0
            label_sonar = np.asarray(label_sonar)/128.0-1.0
            label_sonar = label_sonar[:,:,np.newaxis]
            label_color = np.asarray(label_color)/128.0-1.0

            h,w,_ = img.shape
            xl = np.random.randint(0,w-256)
            yl = np.random.randint(0,h-512)
            img = img[yl:yl+512, xl:xl+256, :]
            label_sonar = label_sonar[yl:yl+512, xl:xl+256,:]
            label_color = label_color[yl:yl+512, xl:xl+256,:]

            imgDataset.append(img)
            slabelDataset.append(label_sonar)
            clabelDataset.append(label_color)


        print("load dataset done")
        return np.array(imgDataset),np.array(slabelDataset),np.array(clabelDataset)
