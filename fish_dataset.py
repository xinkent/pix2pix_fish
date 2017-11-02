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
            img = Image.open(dataDir + "up/train_up%05d.png"%imgNum)
            label_color = Image.open(dataDir + "up_night/night_up%05d.png"%clabelNum)
            label_sonar = Image.open(dataDir + "sonar/" + "2017-03-02_105804_%d_width_773_height_1190.png"%slabelNum)


            label_sonar = label_sonar.convert("L")
            img = img.resize((512,256), Image.BILINEAR)
            img= img.transpose(Image.ROTATE_90)
            label_sonar = label_sonar.resize((256, 512),Image.BILINEAR)
            label_color = label_color.resize((256, 512),Image.BILINEAR)

            img = np.asarray(img)/128.0-1.0

            label_sonar = np.asarray(label_sonar)/128.0-1.0
            label_sonar = label_sonar[:,:,np.newaxis]

            label_color = np.asarray(label_color)/128.0-1.0

            imgDataset.append(img)
            slabelDataset.append(label_sonar)
            clabelDataset.append(label_color)


        print("load dataset done")
        return np.array(imgDataset),np.array(slabelDataset),np.array(clabelDataset)
