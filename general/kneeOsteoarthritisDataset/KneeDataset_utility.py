import numpy as np
from kneeOsteoarthritisDataset.KneeOsteoarthritsDataset import KneeOsteoarthritsDataset

data_path = '/home/biomech/Documents/OsteoData/KneeXrayData/ClsKLData/kneeKL299'

kneeosteo = KneeOsteoarthritsDataset(data_path = data_path)

imgs, labels = kneeosteo.load_imgs()

rand_idx = np.random.randint(low=0,high=len(labels))

img = imgs[rand_idx]
label = labels[rand_idx]
kneeosteo.plot_img(img,label)
counter = 0
for item in labels:
    if item == 2:
        counter += 1
print(counter)