import numpy as np
from PIL import Image

from io import BytesIO

def load_dataset(dataDir='/data1/train_data/', data_range=range(0,300),skip=True, night=10):
        print("load dataset start")
        print("     from: %s"%dataDir)
        imgDataset = []
        clabelDataset = []
        excludes = np.concatenate([np.arange(226,253), np.arange(445,455), np.arange(796, 803), np.arange(2100,2117),
        np.arange(2267, 2317), np.arange(2764, 2835), np.arange(3009, 3029), np.arange(3176, 3230),
        np.arange(3467, 3490), np.arange(3665, 3735), np.arange(3927, 4001), np.arange(4306,4308),
        np.arange(4416, 4476), np.arange(4737, 4741), np.arange(4846, 4906), np.arange(5406, 5464),
        np.arange(5807, 5841), np.arange(6101, 6145)]) # training対象外

        mask = [d not in excludes for d in data_range]
        data_range = data_range[mask]
        imgStart = 1
        clabelStart = 1

        for i in data_range:
            if skip:
                if i%3 != 0:
                    continue
            imgNum = imgStart + i
            clabelNum = clabelStart + i
            img = Image.open(dataDir + "up/up%05d.png"%imgNum)
            label_color = Image.open(dataDir+ "night_100/" + "up_" + str(night).replace('.','') + "night/night_up%05d.png"%clabelNum)

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


def load_dataset2(dataDir='/data1/train_data/', data_range=range(0,300),skip=True, night=10):
        print("load dataset start")
        print("     from: %s"%dataDir)
        imgDataset    = []
        nightlDataset = []
        sonarDataset  = []

        # trainingに使えないデータ(エイがカメラの前を通った場面など)を除去
        excludes = np.concatenate([np.arange(226,253), np.arange(445,455), np.arange(796, 803), np.arange(2100,2117),
                                   np.arange(2267, 2317), np.arange(2764, 2835), np.arange(3009, 3029), np.arange(3176, 3230),
                                   np.arange(3467, 3490), np.arange(3665, 3735), np.arange(3927, 4001), np.arange(4306,4308),
                                   np.arange(4416, 4476), np.arange(4737, 4741), np.arange(4846, 4906), np.arange(5406, 5464),
                                   np.arange(5807, 5841), np.arange(6101, 6145)]) # training対象外
        mask = [d not in excludes for d in data_range]
        data_range = data_range[mask]

        imgStart   = 0
        sonarStart = 0
        nightStart = 0
        for i in data_range:
            # 似たようなデータが多いの1/3に間引く
            if skip:
                if i%3 != 0:
                    continue
            imgNum   = imgStart + i
            sonarNum = sonarStart + i
            nightNum = nightStart + i
            img 　= Image.open(dataDir + "up/up%05d.png"%imgNum)
            night = Image.open(dataDir + "night_100/" +"up_" + str(night).replace('.','') + "night/night_up%05d.png"%nightNum)
            sonar = Image.open(dataDir + "sonar/sonar%05d.png"%sonarNum)
            sonar = sonar.convert("L")

　　　　　　　# 短い辺が300pixになるようにresizeし、rgbを(-1,1)に正規化
            w,h = img.size
            r = 300/min(w,h)
            img   = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
            night = night.resize((int(r*w), int(r*h)),Image.BILINEAR)
            sonar = sonar.resize((int(r*w), int(r*h)),Image.BILINEAR)
            img   = np.asarray(img)/128.0-1.0
            sonar = (np.asarray(label_sonar)/128.0-1.0)[:,:,np.newaxis]
            night = np.asarray(label_color)/128.0-1.0
            # 512 * 256にランダムクリップ
            h,w,_ = img.shape
            xl = np.random.randint(0,w-256)
            yl = np.random.randint(0,h-512)
            img = img[yl:yl+512, xl:xl+256, :]
            sonar = sonar[yl:yl+512, xl:xl+256,:]
            night = night[yl:yl+512, xl:xl+256,:]

            imgDataset.append(img)
            sonarDataset.append(sonar)
            nightDataset.append(night)


        print("load dataset done")
        return np.array(imgDataset),np.array(sonarDataset),np.array(nightDataset)
