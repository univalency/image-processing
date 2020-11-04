#In this code I simply apply the filter to create contours on the picture.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


img=mpimg.imread('house.png')
imgplot = plt.imshow(img)
print(img.shape)
dim_1 = img.shape[0]
dim_2 = img.shape[1]
print(dim_1 , dim_2)
print(img [:,:,0:1].reshape((dim_1,dim_2)).shape)


layer_1 = img [:,:,0:1].reshape((dim_1,dim_2))
layer_2 = img [:,:,1:2].reshape((dim_1,dim_2))
layer_3 = img [:,:,2:3].reshape((dim_1,dim_2))
layer_4 = img [:,:,3:].reshape((dim_1,dim_2))
convolution_mat = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
#convolution_mat = np.array([[0,0,0],[0,1,0],[0,0,0]])
#convolution_mat = np.array([[1,0,-1],[0,0,0],[-1,0,1]])


for k in range(0,3):
	for i in range (0,dim_1 - 2):
		for j in range (0, dim_2 - 2):
			a=(img [:,:,k:k+1] [i:i + 3,j:j + 3] * convolution_mat).sum()
			if(a<0):
				img [:,:,k:k+1][i,j]=0
			if(a>255):
				img [:,:,k:k+1][i,j]=255
			if(a>=0 and a<=255):
				img [:,:,k:k+1][i,j]=a

plt.imshow((img * 255).astype(np.uint8))
plt.show()


print((np.array([[1,2,3],[3,2,1],[1,1,1]])*np.array([[1,1,1],[1,2,3],[3,2,1]])).sum())

