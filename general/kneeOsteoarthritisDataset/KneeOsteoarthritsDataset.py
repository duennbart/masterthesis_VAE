import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class KneeOsteoarthritsDataset():


    def __init__(self, data_path,image_size=(256,256)):
        self.data_path = data_path
        self.image_size = image_size
        # get image path and label
        # train data
        #path = os.path.join(self.data_path,'train')
        self.dict_url_class = self.read_img_pathes(self.data_path)

        # test data
        #path = os.path.join(self.data_path, 'test')
        #self.dict_url_class.update(self.read_img_pathes(path))

        # validation data
        #path = os.path.join(self.data_path, 'val')
        #self.dict_url_class.update(self.read_img_pathes(path))

    def read_img_pathes(self,path):
        '''
        return pathes to images and the corresponding labels as a dictonary
        :path:
        :return:
        '''

        dict_url_class = {}
        # read class 0

        for i in range(0,5):
            KL_class = i
            cL_path = os.path.join(path, str(KL_class))
            entries = os.listdir(cL_path)
            entries.sort()
            for entry in entries:

                if not entry.startswith('.')  and os.path.splitext(entry)[-1].lower() == '.png':
                    dict_url_class[os.path.join(cL_path,entry)] = KL_class

        return dict_url_class

    def load_imgs(self):
        self.imgs = []
        self.labels = []
        for item in self.dict_url_class.items():
            img, label =  self.load_img(path=item[0],label=item[1])
            self.imgs.append(img)
            self.labels.append(label)

        assert len(self.imgs) == len(self.labels)
        return np.asarray(self.imgs), np.asarray(self.labels)

    def load_img(self,path,label=None):
        img = cv2.imread(path, 0)
        img = cv2.resize(img,self.image_size)
        img = img.astype(np.float32, copy=False) / 255.0
        return img,label

    def plot_img(self,img,label):
        plt.gray()
        # visualize 5 images from dataset
        f, ax = plt.subplots(1, 1)
        ax.imshow(img)

        f.suptitle('Category: ' + str(label))
        plt.show()



