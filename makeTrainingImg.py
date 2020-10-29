# -*- coding: utf-8 -*-
"""
Created on Sat May 19 12:55:07 2018

@author: 益慶
"""
import random
import matplotlib.pyplot as plt
import numpy as np
 
noise_img=images[:]

def Line(img):
     v_heigh= 10
     v_width= 2
     h_heigh= 2
     h_width= 10
     d_length=15
     vpoint_h=random.randint(1,10)
     vpoint_w=random.randint(1,30)
     hpoint_h=random.randint(1,30)
     hpoint_w=random.randint(1,10)
     dpoint_h=random.randint(1,10)
     dpoint_w=random.randint(8,16)
     for i in range(v_width):
         for j in range(v_heigh):
             img[vpoint_w+i,vpoint_h+j]=1
             
     for p in range(h_width):
         for q in range(h_heigh):
             img[hpoint_w+p,hpoint_h+q]=1
             
     for l in range(d_length):
         if dpoint_w>12:
             img[dpoint_w-l,dpoint_h+l]=1
         else:
             img[dpoint_w+l,dpoint_h+l]=1
         

def gauss_noisy(image):
     row,col,ch= image.shape
     mean = 0
     var = 0.01
     sigma = var**0.5
     gauss = np.random.normal(mean,sigma,(row,col,ch))
     gauss = gauss.reshape(row,col,ch)
     noisy = image + gauss
     for x in range(0,32):
        for y in range(0,32):
            for z in range(0,3):
                if noisy[x,y,z]<0:
                    noisy[x,y,z]=0
     return noisy
 

def peppersalt(img, n, m):
    """
    Add peppersalt to image
    :param img: the image you want to add noise
    :param n: the total number of noise (0 <= n <= width*height)
    :param m: different mode
    m=1:add only white noise in whole image
    m=2:add only black noise in whole image
    m=3:add black and white noise in whole image
    m=4:add gray scale noise range from 0 to 1
    m=5:add color noise in whole image,RGB is combined randomly with every channel ranges from 0 to 1
    :return: the processed image
    """
    
    if m == 1:
        for i in range(n):
            x = random.randint(0,31)
            y = random.randint(0,31)
            img[x, y, 0] = 1
            img[x, y, 1] = 1
            img[x, y, 2] = 1
    elif m == 2:
        for i in range(n):
            x = random.randint(0,31)
            y = random.randint(0,31)
            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 0
    elif m == 3:
        for i in range(n):
            x = random.randint(0,31)
            y = random.randint(0,31)
            flag = np.random.random()
            if flag > 0.5:
                img[x, y, 0] = 1
                img[x, y, 1] = 1
                img[x, y, 2] = 1
            else:
                img[x, y, 0] = 0
                img[x, y, 1] = 0
                img[x, y, 2] = 0
    elif m == 4:
        for i in range(n):
            x = random.randint(0,31)
            y = random.randint(0,31)
            flag = int(np.random.random())
            img[x, y, 0] = flag
            img[x, y, 1] = flag
            img[x, y, 2] = flag
    elif m == 5:
        for i in range(n):
            x = random.randint(0,31)
            y = random.randint(0,31)
            f1 = np.random.random()
            f2 = np.random.random()
            f3 = np.random.random()
            img[x, y, 0] = f1
            img[x, y, 1] = f2
            img[x, y, 2] = f3
    return img
    

#Line
for img in noise_img:
#    n=random.randint(3,8)
    n=8
    for num in range(n):
        Line(img)

#GaussNoise
#index=0     
#for img in noise_img:
#    noise_img[index]=gauss_noisy(img)
#    index+=1
     
#PepperSalt
#index=0 
#for img in noise_img:
#    noise_img[index]=peppersalt(img, 150, 5)
#    index+=1
    
plt.imshow(noise_img[0])
plt.show()  
plt.imshow(noise_img[4253])
plt.show()
plt.imshow(noise_img[2499])
plt.show()