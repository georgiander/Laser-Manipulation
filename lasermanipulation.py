import matplotlib.pyplot as plt
import numpy as np
import PIL
import cv2
import glob


def ft(x):
    fft=np.fft.fftshift(np.fft.fft2(x))
    return fft

def invft(x):
    ifft=np.fft.ifft2(np.fft.ifftshift(x))
    return ifft

def fouriertest(image):
    f_size = 8
    fourier = np.fft.fftshift(np.fft.fft2(image))
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(np.log(np.abs(fourier)))
    ax[0].set_title('Masked Fourier', fontsize = f_size)
    ax[1].imshow(image, cmap='gray')
    ax[1].set_title('Grayscale Image', fontsize = f_size)
    ax[2].imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(fourier))), cmap='gray')
    ax[2].set_title('Transformed Image', 
                     fontsize = f_size)
    plt.show()

def GSplot(initial, fourier, iterations):
    peak=(np.abs(initial)) #find amplitude of initial image
    fpeak=(np.abs(fourier))
    A=initial
    for i in range(0,iterations): #iterate function to get good results
        initimg=abs(peak)*np.exp(1j*np.angle(A))  #define the initial image as the initial amplitude*e^i(phase)
        fourimg=ft(initimg)# transform image
        newimg=abs(fpeak)*np.exp(1j*np.angle(fourimg)) #define new image as the fourier amplitude*e^i(fourier phase)
        A=invft(newimg) #transfer image back to image plane
        f_size=10
        print('Computing iteration ', i+1)  
    fig, ax = plt.subplots(1,3) #Plot the wanted fourier plane, the phase mask, and the final fourier plane
    ax[0].imshow(fourier, cmap='gray') #wanted fourier
    ax[0].set_title('Fourier Image', fontsize = f_size)
    ax[1].imshow(np.angle(A), cmap='gray') #Phase of A
    ax[1].set_title('Phase Mask', fontsize = f_size)
    ax[2].imshow(np.abs(fourimg)) #New Fourier Image
    ax[2].set_title('Rescontructed Image', fontsize = f_size)
    plt.show()

def GS(initial, fourier, iterations):
    peak=(np.abs(initial)) #find amplitude of initial image
    fpeak=(np.abs(fourier))
    A=initial
    for i in range(0,iterations): #iterate function to get good results
        initimg=abs(peak)*np.exp(1j*np.angle(A))  #define the initial image as the initial amplitude*e^i(phase)
        fourimg=ft(initimg)# transform image
        newimg=abs(fpeak)*np.exp(1j*np.angle(fourimg)) #define new image as the fourier amplitude*e^i(fourier phase)
        A=invft(newimg) #transfer image back to image plane
        print('Computing iteration', i+1) 
    img=np.angle(A)
    while(True):
        cv2.imshow(fourier, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def plotImage(y, photodirectory):
    photo = photodirectory + '\\'+'avg'  + str(y) +'.jpg'  #define the image you would like to plot
    img=cv2.imread(photo)
    plt.imshow(img)
    plt.show()

def showIsolation(y, xul, yul, xlr, ylr, photodirectory, waittime=5):
    photo = photodirectory + '\\'+'avg' + str(y) +'.jpg' #define your image
    img=cv2.imread(photo)
    cv2.rectangle(img, (xul, yul), (xlr, ylr), (0, 0, 255), 1) #create a red rectangle in the space you would like to take measurements from the upper left (xul, yul) and the lower right corners (xlr, ylr).
    cv2.imshow('', img)
    cv2.waitKey(waittime) #for a constant image, enter 0 for waittime.
    cv2.destroyAllWindows

def Calibrate(samplesize, xul, yul, xlr, ylr, photodirectory, increment): #samplesize:amount of pictures you want to take  increment: increments between 0 and 255 you are making
    mean=np.zeros((samplesize, 1))
    for i in range(mean.size): #find the mean for all photos in directory
        y=i*increment #level between 0:255 you are working at
        photo = photodirectory + '\\' +'avg' +str(y) + '.jpg' #define image
        img=cv2.imread(photo, 0)
        showIsolation(y, xul, yul, xlr, ylr, photodirectory, 500) #show isolated section of images
        image=img[yul:ylr, xul:xlr] #isolate same area in order to work with
        mean[i]=np.mean(image) #take average of isolated area
        print('isolated mean for SLM level',y, 'is', mean[i])
    y=np.linspace(0, ((samplesize-1)*increment), samplesize) #define the y-axis based on increments and sample size
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(y, mean, c='b') #plot the averaged data vs. the SLM level (y)
    plt.show()

def AverageCalibrate():
    images=glob.glob(r"YOURPATHHERE") #find plots made from Calibrate()

    image_data=[] #create array of graphs
    for img in images:
        timg=cv2.imread(img, 0)
        image_data.append(timg)

    avg_img=image_data[0]
    for i in range(len(image_data)): #combine graphs into one super graph that combines them all.
        if i==0:
            pass
        else:
            alpha=1.0/(i+1)
            beta=1.0-alpha
            avg_img=cv2.addWeighted(image_data[i], alpha, avg_img, beta, 0.0)

    cv2.imwrite('avggraphs.jpg', avg_img) #save graph of average graphs
    avg_img=cv2.imread('avggraphs.jpg') #show said graph

def GaussianCut(x0, xsize=10, X=10):
    '''Slices a Gaussian Curve down the y=y0 axis and gives a sigma value'''
    x = np.linspace(x0-xsize, x0+xsize,50)
    Z=np.e**(-4*((x-x0)**2)/(X**2)) #E(x,y) where X is the FWHM in X at e^-1
    sigma=np.sqrt(X**2/4)
    Gaussian=plt.figure()
    plt.title('Sigma = %i' %sigma)
    plt.plot(x,Z)
    plt.show()

def BuildGaussian(x0, y0, xsize=10, ysize=10, sigma=10):
    '''Builds a Gaussian Curve in 3D space'''
    X = np.linspace(x0-xsize,x0+xsize,50)
    Y = np.linspace(y0-ysize,y0+ysize,50)
    X, Y = np.meshgrid(X,Y)
    Z=np.e**(-((X-x0)**2+(Y-y0)**2)/sigma**2)
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.plot_surface(X,Y, Z, cmap='plasma')
    plt.show()

def fullGaussian(x0, y0, xsize=10, ysize=10, eHM=10, maskType ='contour'):
    '''Defines a Gaussian Based on a FWHM of e^-1. Combines BuildGaussian and GaussianCut. Outputs a Slice, Gaussian and Gaussian Mask'''
    maskTypes= ['contour', 'gradient','dlc', 'dlg']
    if maskType not in maskTypes:
        raise ValueError("Invalid mask type. Expected 'gradient' or 'contour'")
    X3d = np.linspace(x0-xsize,x0+xsize,700) #define X in 2 and 3D
    X2d = np.linspace(x0-xsize,x0+xsize,700)
    Y = np.linspace(y0-ysize,y0+ysize,700)
    X, Y = np.meshgrid(X3d,Y)
    sigma=np.sqrt(eHM**2/4) #eHM= Full Width Half Maximum of e^-1
    Z2d=np.e**(-4*((X2d-x0)**2)/(eHM**2))
    Z3d=np.e**(-((X3d-x0)**2+(Y-y0)**2)/sigma**2)

    GaussianCut=plt.figure() #Plot Slice
    plt.title('Middle Splice')
    ax=plt.axes()
    ax.set_xlabel('Sigma = %i' %sigma)
    plt.plot(X2d, Z2d)
    GaussianCut.canvas.manager.set_window_title('Splice')

    Gaussian=plt.figure() #Plot 3D Gaussian
    plt.title('Full Gaussian')
    ax1=plt.axes(projection='3d')
    map=ax1.plot_surface(X,Y, Z3d, cmap='plasma')
    Gaussian.colorbar(map)
    Gaussian.canvas.manager.set_window_title('3D Gaussian')

    if maskType =='contour': #Plot Mask
        mask=plt.figure(figsize=(8,6), dpi=80)
        ax2=plt.gca()
        contour=ax2.contourf(X, Y, Z3d, cmap='plasma') #Gives a Contour Map (Easy to Load)
        mask.colorbar(contour)
        mask.canvas.manager.set_window_title('Gaussian Mask (Contour)')
        plt.title('Gaussian Mask')
    if maskType == 'gradient':
        mask=plt.figure(figsize=(8,6), dpi=80)
        ax2=plt.gca()
        gradient=plt.scatter(X, Y, c=Z3d, cmap='plasma') #Gives a Scatter Gradient (Slow to Load)
        mask.colorbar(gradient)
        mask.canvas.manager.set_window_title('Gaussian Mask (Gradient)')
        plt.title('Gaussian Mask')
    if maskType =='dlc':
        mask=plt.figure(figsize=(6,6), dpi=80)
        ax2=plt.gca()
        contour=ax2.contourf(X, Y, Z3d, cmap='gray') #Gives a Contour Map that can be saved for GSA
        mask.canvas.manager.set_window_title('Gaussian Mask (Contour)')
        plt.axis('off')
    if maskType == 'dlg':
        mask=plt.figure(figsize=(6,6), dpi=80)
        ax2=plt.gca()
        gradient=plt.scatter(X, Y, c=Z3d, cmap='gray') #Gives a Scatter Gradient that can be saved for GSA
        mask.canvas.manager.set_window_title('Gaussian Mask (Gradient)')
        plt.axis('off')
        
   
    plt.show()

def GaussianMask(x0, y0, xsize=10, ysize=10, sigma=10, maskType ='contour'):
    maskTypes= ['contour', 'gradient', 'dlc', 'dlg']
    if maskType not in maskTypes:
        raise ValueError("Invalid mask type. Expected type 'gradient', 'contour', or GSA types")
    
    X3d = np.linspace(x0-xsize,x0+xsize,700) #define X in 2 and 3D
    Y = np.linspace(y0-ysize,y0+ysize,700)
    X, Y = np.meshgrid(X3d,Y)
    Z3d=np.e**(-((X3d-x0)**2+(Y-y0)**2)/sigma**2)
    if maskType =='contour':
        mask=plt.figure(figsize=(8,6), dpi=80) #Plot Mask
        ax2=plt.gca()
        contour=ax2.contourf(X, Y, Z3d, cmap='plasma') #Gives a Contour Map (Easy to Load)
        mask.colorbar(contour)
        mask.canvas.manager.set_window_title('Gaussian Mask (Contour)')
        plt.title('Gaussian Mask')
    if maskType == 'gradient':
        mask=plt.figure(figsize=(8,6), dpi=80) #Plot Mask
        ax2=plt.gca()
        gradient=plt.scatter(X, Y, c=Z3d, cmap='plasma') #Gives a Scatter Gradient (Slow to Load)
        mask.colorbar(gradient)
        mask.canvas.manager.set_window_title('Gaussian Mask (Gradient)')
        plt.title('Gaussian Mask')
    if maskType =='dlc':
        mask=plt.figure(figsize=(6,6), dpi=80) #Plot Mask
        ax2=plt.gca()
        contour=ax2.contourf(X, Y, Z3d, cmap='gray') #Gives a Contour Map that can be saved for GSA
        mask.canvas.manager.set_window_title('Gaussian Mask (Contour)')
        plt.axis('off')
    if maskType == 'dlg':
        mask=plt.figure(figsize=(6,6), dpi=80) #Plot Mask
        ax2=plt.gca()
        gradient=plt.scatter(X, Y, c=Z3d, cmap='gray') #Gives a Scatter Gradient that can be saved for GSA
        mask.canvas.manager.set_window_title('Gaussian Mask (Gradient)')
        plt.axis('off')
        
   
    plt.show()

GaussianMask(0, 0, maskType='gradient')