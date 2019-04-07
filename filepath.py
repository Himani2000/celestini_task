import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
import sys
GT_Save="/media/dheeraj/9A26F0CB26F0AA01/WORK/CELESTINI/celestini_task/Ground_Truth"
Input_Save="/media/dheeraj/9A26F0CB26F0AA01/WORK/CELESTINI/celestini_task/Input"




Directory_GT0= "/media/dheeraj/9A26F0CB26F0AA01/WORK/CELESTINI/celestini_task/MiddEval3-GT0-Q/trainingQ/"
files = os.listdir(Directory_GT0)
print(files)


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
		cv2.imwrite(GT_Save + "/"  + str(file)+"_GT0_" + str(img)  , img_array)		

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
		cv2.imwrite(GT_Save + "/" + str(file) +  "_GT1_" +str(img)  , img_array)		


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




	
