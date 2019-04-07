import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
import sys
#directory = "/celestini_task/"
#files = os.listdir(directory)
GT_Save="/media/dheeraj/9A26F0CB26F0AA01/WORK/CELESTINI/celestini_task/Ground_Truth"
Input_Save="/media/dheeraj/9A26F0CB26F0AA01/WORK/CELESTINI/celestini_task/Input"
#Categories=["Dog", "Cat"]



'''images=os.path.join(directory,files)
print("\n\n\nimages=\n", images)'''
#print( " files = \n\n\n\n\n", files)
#print("\n\n\n\n")

Directory_GT0= "/media/dheeraj/9A26F0CB26F0AA01/WORK/CELESTINI/celestini_task/MiddEval3-GT0-Q/trainingQ/"
files = os.listdir(Directory_GT0)
print(files)

'''
for category in Categories:
    
     # here we need to join the path of the directories of dogs and cats together
    path=os.path.join(data, category)
    
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE ) #converting the photo to greyscale
        plt.imshow(img_array)
        plt.show()
        break
    break    

'''

images=[]
for file in files:
	path=os.path.join(Directory_GT0, file)
	for img in os.listdir(path):
		#images.append(img)
		#print("path= " , "\n\n\n\n", img)
		print("path = " , path )
		image=img.split('.')
		print(image)
		#if (img[1]=='png')):

		if (image[1] != 'png'):
			continue

		img_array=cv2.imread(os.path.join(path, img))
		#print(file)
		print("img = ", img)
		cv2.imwrite(GT_Save + "/" +"GT0_" + str(file) + str(img)  , img_array)		

Directory_GT1= "/media/dheeraj/9A26F0CB26F0AA01/WORK/CELESTINI/celestini_task/MiddEval3-GT1-Q/trainingQ/"
files = os.listdir(Directory_GT1)
print(files)

for file in files:
	path=os.path.join(Directory_GT1, file)
	for img in os.listdir(path):
		#images.append(img)
		#print("path= " , "\n\n\n\n", img)
		print("path = " , path )
		image=img.split('.')
		print(image)
		#if (img[1]=='png')):

		if (image[1] != 'png'):
			continue

		img_array=cv2.imread(os.path.join(path, img))
		#print(file)
		print("img = ", img)
		cv2.imwrite(GT_Save + "/" +  "GT1_"+ str(file) + str(img)  , img_array)		


Directory_input= "/media/dheeraj/9A26F0CB26F0AA01/WORK/CELESTINI/celestini_task/MiddEval3-data-Q/trainingQ/"
files = os.listdir(Directory_input)
print(files)

for file in files:
	path=os.path.join(Directory_input, file)
	for img in os.listdir(path):
		#images.append(img)
		#print("path= " , "\n\n\n\n", img)
		print("path = " , path )
		image=img.split('.')
		print(image)
		#if (img[1]=='png')):

		if (image[1] != 'png'):
			continue

		img_array=cv2.imread(os.path.join(path, img))
		#print(file)
		print("img = ", img)
		cv2.imwrite(Input_Save + "/" + str(file) + str(img)  , img_array)		




	
