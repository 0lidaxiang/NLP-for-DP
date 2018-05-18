import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = ""
        path = glob('./datasets/%s/*' % (self.dataset_name))
        
        # batch_images = np.random.choice(path, size=batch_size)
        batch_images =  sorted(path)

        imgs_A = []
#         imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            _w = int(w)
            img_A  = img[:, :, :] 

            img_A = scipy.misc.imresize(img_A, self.img_res)
#             img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
#             if not is_testing and np.random.random() < 0.5:
#                 img_A = np.fliplr(img_A)
#                 img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
#             imgs_B.append(img_B)

#         imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_A = np.array(imgs_A)
#         imgs_B = np.array(imgs_B)

#         return imgs_A, imgs_B
        return imgs_A

    def load_batch(self, batch_size=1, is_testing=False):
#         data_type = "train" if not is_testing else "val"
        path = glob('./datasets/%s/*' % (self.dataset_name))

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A  = []
            for img in batch:
                img = self.imread(img)
                h, w, _ = img.shape
                half_w = int(w/2)
                img_A = img[:, : , :]
#                 img_B = img[:, half_w:, :]

                img_A = scipy.misc.imresize(img_A, self.img_res)
#                 img_B = scipy.misc.imresize(img_B, self.img_res)

#                 if not is_testing and np.random.random() > 0.5:
#                         img_A = np.fliplr(img_A)
#                         img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
#                 imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
#             imgs_B = np.array(imgs_B)
            yield imgs_A


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
