import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
import sys
#directory = "/celestini_task/"
#files = os.listdir(directory)
GT_Save="./Ground_Truth/"
Input_Save="./Input/"
#Categories=["Dog", "Cat"]



'''images=os.path.join(directory,files)
print("\n\n\nimages=\n", images)'''
#print( " files = \n\n\n\n\n", files)
#print("\n\n\n\n")

Directory= "/media/dheeraj/9A26F0CB26F0AA01/WORK/CELESTINI/celestini_task/MiddEval3-GT0-Q/trainingQ/"
files = os.listdir(Directory)

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
	path=os.path.join(Directory, file)
	for img in os.listdir(path):
		#images.append(img)
		#print("path= " , "\n\n\n\n", img)
		img_array=cv2.imread(os.path.join(path, img))
		cv2.imshow('image', img_array)
		#print(img_array)
		#plt.imshow(img_array)
		#plt.show()

		#print("printing ith image: ", "\n\n")
		#print(i[5])

		
		#if img[5]=='G' and img[6]=='T':
		#cv2.imwrite(GT_Save + "/" +str(img) , img_array )
			
		#elif img[5]=='N' and img[6]=='O':
		cv2.imwrite(Input_Save + "/"  + "_" +str(img) , img_array)		
	
#print(images)

'''
for i in images:
	print("printing ith image: ", "\n\n")
	print(i[5])
	if i[5]=='G' and i[6]=='T':
		cv2.imwrite(GT_Save + "/" +str(i) , )
			
	elif i[5]=='N' and i[6]=='O':
		cv2.imwrite(Noisy_Save + "/" +str(i) , i)		
			
'''
'''
for file in files:
	path=os.path.join(directory, files)
    for img in os.listdir(path):
    	img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE ) #converting the photo to greyscale
    	plt.imshow(img_array)
    	plt.show()
    	    
''' 