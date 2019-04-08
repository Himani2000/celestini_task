import cv2
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import os   
from matplotlib import pyplot as plt


grayscale_max = 255

dirsave="./Results/"
dirGT0="./GT0"
dirGT1="./GT1"


dirIP0="./IM0"
dirIP1="./IM1"


def load_image_IP0():

    files = os.listdir(dirIP0)
    listimgl=[]
    #listpathl=[]
    filepaths_IP = [os.path.join(dirIP0,i) for i in files]
    for i in filepaths_IP:
        img_l = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        listimgl.append(img_l)
        #listpathl.append(i)    

        #return img_l, i
    return listimgl



def load_image_IP1():

    files = os.listdir(dirIP1)
    listimgr=[]
    #listpathr=[]
    filepaths_IP = [os.path.join(dirIP1,i) for i in files]
    #print("The value of i is ", i)
    for i in filepaths_IP:
        img_r = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        listimgr.append(img_r)
        #listpathr.append(i)
        #return img_r, i
    return listimgr




def load_image_GT0():

    files = os.listdir(dirGT0)
    listGT0=[]
    filepaths_GT = [os.path.join(dirGT0,i) for i in files]
    for i in filepaths_GT:
        gt_0 = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        listGT0.append(gt_0)

    return listGT0



def load_image_GT1():

    files = os.listdir(dirGT1)
    listGT1=[]
    filepaths_GT = [os.path.join(dirGT1,i) for i in files]
    for i in filepaths_GT:

        gt_1 = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        #return gt_1
        listGT1.append(gt_1)

    return listGT1    


    
def show_image(title, image):
    max_val = image.max()
    # image = np.absolute(image)
    image = np.divide(image, max_val)
    # cv2.imshow(title, image)
    #cv2.imwrite(title+str(random.randint(1, 100))+'.png', image*grayscale_max)
    #cv2.imwrite(dirsave + title+ path, image*grayscale_max)
    cv2.imwrite(dirsave+title+str(random.randint(1, 100))+'.png', image*grayscale_max)


def add_padding(input, padding):
    rows = input.shape[0]
    print("Rows = ", rows)
    columns = input.shape[1]
    output = np.zeros((rows + padding * 2, columns + padding * 2), dtype=float)
    output[ padding : rows + padding, padding : columns + padding] = input
    return output


def search_bounds(column, block_size, width, rshift):
    disparity_range = 75
    padding = block_size // 2
    right_bound = column
    if rshift:
        left_bound = column - disparity_range
        if left_bound < padding:
            left_bound = padding
        step = 1
    else:
        left_bound = column + disparity_range
        if left_bound >= (width - 2*padding):
            left_bound = width - 2*padding - 2
        step = -1
    return left_bound, right_bound, step


# max disparity 30
def disparity_map(left, right, block_size, rshift):

    padding = block_size // 2
    left_img = add_padding(left, padding)
    right_img = add_padding(right, padding)

    height, width = left_img.shape

    # d_map = np.zeros((height - padding*2, width - padding*2), dtype=float)
    d_map = np.zeros(left.shape , dtype=float)

    for row in range(height - block_size + 1):
        for col in range(width - block_size + 1):

            bestdist = float('inf')
            shift = 0
            left_pixel = left_img[row:row + block_size, col:col + block_size]
            l_bound, r_bound, step = search_bounds(col, block_size, width, rshift)

            # for i in range(l_bound, r_bound - padding*2):
            for i in range(l_bound, r_bound, step):
                right_pixel = right_img[row:row + block_size, i:i + block_size]

                # if euclid_dist(left_pixel, right_pixel) < bestdist :
                ssd = np.sum((left_pixel - right_pixel) ** 2)
                # print('row:',row,' col:',col,' i:',i,' bestdist:',bestdist,' shift:',shift,' ssd:',ssd)
                if ssd < bestdist:
                    bestdist = ssd
                    shift = i

            if rshift:
                d_map[row, col] = col - shift
            else:
                d_map[row, col] = shift - col
            print('Calculated Disparity at ('+str(row)+','+str(col)+') :', d_map[row,col])
    #print(d_map.shape)
    return d_map


def squared_mean_square_error(disparity_map, ground_truth):
    # ssd = np.sum((disparity_map - ground_truth)**2)
    # mse = ssd/(ground_truth.shape[0]*ground_truth.shape[1])
    mse = np.sum((disparity_map - ground_truth)**2)
    return mse

def absolute_mean_square_error(disparity_map, ground_truth):
    # ssd = np.sum((disparity_map - ground_truth)**2)
    # mse = ssd/(ground_truth.shape[0]*ground_truth.shape[1])
    mse = np.sum((disparity_map - ground_truth))
    return mse



def correlation_coefficient(disparity_map, ground_truth):
    product = np.mean((disparity_map - disparity_map.mean()) * (ground_truth - ground_truth.mean()))
    standard_dev = disparity_map.std() * ground_truth.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product




global d_map_lr_5
global d_map_lr_7


def main():
    l= load_image_IP0()
    r= load_image_IP1()
    #cv2.imshow('l', l)
    #print(type(l),type(r))
    #print(len(l))
    #l='im0.png'
    #r='im1.png'

    #d_map_lr_5 = disparity_map(l,r,3, True)
    #show_image('D_Map_lr_block5_', d_map_lr_5)


    ground_truth_1=load_image_GT0()
    ground_truth_2 = load_image_GT1()
    

    for i,j in zip(l,r):
         # For window size of 5
        d_map_lr_5 = disparity_map(i,j,5, True)
        show_image('D_Map_lr_5_', d_map_lr_5)

        # For window size of 7
        d_map_lr_7 = disparity_map(i,j,7, True)
        show_image('D_Map_lr_7_', d_map_lr_7)
    


    # Mean Squared Error



    for i,j in zip(ground_truth_1,ground_truth_2):
         # For window size of 5
        print(i.shape)
        print(d_map_lr_5.shape)
        loss_sm_5_lr = squared_mean_square_error(d_map_lr_5, i)
        print("Loss for window size 5  for GT0 is" , loss_sm_5_lr)

        
        loss_sm_7_lr = squared_mean_square_error(d_map_lr_7, i)
        print("Loss for window size 5  for GT0 is" , loss_sm_7_lr)

        
        loss_am_5_lr = absolute_mean_square_error(d_map_lr_5, i)
        print("Loss for window size 5  for GT0 is" , loss_am_5_lr)

        
        loss_am_7_lr = absolute_mean_square_error(d_map_lr_7, i)
        print("Loss for window size 5  for GT0 is" , loss_am_7_lr)

        
        loss_cc_5_lr = correlation_coefficient(d_map_lr_5, i)
        print("MSE for window size 5  for GT0 is" , loss_cc_5_lr)

        loss_cc_5_lr = correlation_coefficient(d_map_lr_5, j)
        print("MSE for window size 5  for GT1 is" , loss_cc_5_lr)

        loss_cc_7_lr = correlation_coefficient(d_map_lr_7, i)
        print("MSE for window size 5  for GT0 is" , loss_cc_7_lr)

        loss_cc_5_lr = correlation_coefficient(d_map_lr_7, j)
        print("MSE for window size 5  for GT1 is" , loss_cc_7_lr)

   
    return


main()

