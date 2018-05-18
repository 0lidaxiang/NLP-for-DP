import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1'
import glob
import numpy as np
import pandas as pd
from scipy.misc.pilutil import imread,imresize,imsave
from keras.models import load_model

def generator_test_Img(list_dir,resize):
    output_training_img=[]
    for i in list_dir:
        img = imread(i,mode='L')
        img = imresize(img,resize)
        output_training_img.append(img)
    output_training_img = np.array(output_training_img)/127.5-1
    output_training_img = np.expand_dims(output_training_img,axis=1) # (batch,img_row,img_col) ==> (batch,1,img_row,img_cok)
    return output_training_img

def numpy_to_csv(input_image,image_number=10,save_csv_name='predict.csv'):
    save_image=np.zeros([int(input_image.size/image_number),image_number],dtype=np.float32)

    for image_index in range(image_number):
        save_image[:,image_index]=input_image[image_index,:,:].flatten()

    base_word='id'
    df = pd.DataFrame(save_image)
    index_col=[]
    for i in range(n):
        col_word=base_word+str(i)
        index_col.append(col_word)
    df.index.name='index'
    df.columns=index_col
    df.to_csv(save_csv_name)
    print("Okay! numpy_to_csv")


test_data_dir='.\NLP_data\Test\White\*' 
test_data_dir_list=glob.glob(test_data_dir)
test_data_list=[]
test_data_list.extend(test_data_dir_list)

n=10
output_img_col = output_img_row=128
white_img = generator_test_Img(list_dir=test_data_list,resize=(output_img_col,output_img_row))
#這邊的G就是Example Code的G拉
#只是怕放在最下面會搞亂大家 所以分別寫出來
image_array = G.predict(white_img).squeeze(1)     #Predict出來是 (n,1,img_row,img_col) 只是做Reshape為(n,img_row,img_col)
image_array = (image_array+1)/2                   #這邊很重要 因為答案是0~1的灰階值 
                                                  #然而如果你Generator輸出是tanh 會介於-1~1之間，須把他變成0~1 ==> ((-1~1)+1)/2=(0~1)
                                                  #當然如果你是Sigmoid輸出，就不用作Rescale的動作


print(image_array.shape)                          #預期是(n,img_row,img_col) 如果不是 可能會轉換不出來，或是跟答案沒對上

numpy_to_csv(input_image=image_array,image_number=n,save_csv_name='Predict.csv')
