import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from mpl_toolkits.mplot3d import Axes3D

#########################################################
#3.1
#import the img and resize it
img = cv2.imread('puppy.jpg')
img_noise = cv2.imread('puppy_.40.jpg')
#img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
#cv2.imshow('noFilter', img.astype(np.uint8))
imgdata = np.zeros(img.shape, np.float64)
img_noise_data = np.zeros(img_noise.shape, np.float64)

#define the size of a filter
filterSize = 9;

# #get a random filter
# fltr= np.random.randn(filterSize,filterSize)
# fltr = fltr / fltr.sum()
#
# #plt.matshow(fltr)
# #plt.show()

# #apply the filter
# img_filtered = np.zeros(img_noise_data.shape, np.float64)
#
# img_filtered[:, :, 0] = cv2.filter2D(img_noise[:, :, 0], -1, fltr)
# img_filtered[:, :, 1] = cv2.filter2D(img_noise[:, :, 1], -1, fltr)
# img_filtered[:, :, 2] = cv2.filter2D(img_noise[:, :, 2], -1, fltr)
# #
# # #output the filtered img
# # cv2.imshow('img_filetred', img_filtered.astype(np.uint8))
# # cv2.waitKey(0)



#########################################################
#3.2.1 smoothing and denoising



# #=================set filtersize = 3 ====================================
# filterSize = 3
# #apply the gaussian filter
# img_gaussian = cv2.GaussianBlur(img_noise, (filterSize,filterSize), 0)
# #cv2.imwrite('img_gausian_3.jpg',img_gaussian)
# cv2.imshow('img_gaussian', img_gaussian.astype(np.uint8))
# #cv2.waitKey(0)
#
# img_median = cv2.medianBlur(img_noise, ksize= filterSize)
# #cv2.imwrite('img_median_3.jpg',img_median)
# cv2.imshow('img_median', img_median.astype(np.uint8))
# cv2.waitKey(0)
#
# #=================set filtersize = 9 ====================================
# filterSize = 3
# #apply the gaussian filter
# img_gaussian = cv2.GaussianBlur(img_noise, (filterSize,filterSize), 0)
# #cv2.imwrite('img_gausian_9.jpg',img_gaussian)
# cv2.imshow('img_gaussian', img_gaussian.astype(np.uint8))
# #cv2.waitKey(0)
#
# img_median = cv2.medianBlur(img_noise, ksize= filterSize)
# #cv2.imwrite('img_median_9.jpg',img_median)
# cv2.imshow('img_median', img_median.astype(np.uint8))
# cv2.waitKey(0)
#
# #=================set filtersize = 27 ====================================
# filterSize = 27
# #apply the gaussian filter
# img_gaussian = cv2.GaussianBlur(img_noise, (filterSize,filterSize), 0)
# #cv2.imwrite('img_gausian_27.jpg',img_gaussian)
# cv2.imshow('img_gaussian', img_gaussian.astype(np.uint8))
# #cv2.waitKey(0)
#
# img_median = cv2.medianBlur(img_noise, ksize= filterSize)
# #cv2.imwrite('img_median_27.jpg',img_median)
# cv2.imshow('img_median', img_median.astype(np.uint8))
# cv2.waitKey(0)


#############################################
# #3.2.2 edge detection
#
# #original image
# img_edge_ori = cv2.Canny(img,100,150)
# #cv2.imwrite('img_edge_ori.jpg',img_edge_ori)
# #cv2.imshow('img_edge_1', img_edge_ori.astype(np.uint8))
#
#
#noisy image
# img_edge_noise = cv2.Canny(img_noise,250,300)
# # cv2.imwrite('img_edge_noise.jpg',img_edge_noise)
# # cv2.imshow('img_edge_2', img_edge_noise.astype(np.uint8))
# # cv2.waitKey(0)
#
# # camera_trap image
# img_trap = cv2.imread('camera_trap.jpg')
# #cv2.imshow('img_trap', img_trap.astype(np.uint8))
# img_edge_trap_noise = cv2.Canny(img_trap,200,300)
# # cv2.imwrite('img_edge_trap_noise.jpg',img_edge_trap_noise)
# # cv2.imshow('img_edge_trap_noise', img_edge_trap_noise.astype(np.uint8))
#
#
# filterSize = 9
# img_trap_clean = cv2.GaussianBlur(img_trap,(filterSize,filterSize),0)
# img_trap_clean = img_trap_clean+100
# # cv2.imwrite('img_trap_clean.jpg',img_trap_clean)
# cv2.imshow('img_trap_clean', img_trap_clean.astype(np.uint8))
#
# img_edge_trap_clean = cv2.Canny(img_trap_clean,100,150)
# # cv2.imwrite('img_edge_trap_clean.jpg',img_edge_trap_clean)
# cv2.imshow('img_edge_trap_clean', img_edge_trap_clean.astype(np.uint8))
# cv2.waitKey(0)

##############################################
# #4.1 Fourier Transformation
# img_gray =  cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img_gray = cv2.resize(img_gray, (0, 0), fx=0.5, fy=0.5)
#
# xx, yy = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, img_gray, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
# plt.show()
#
# # Take the 2-D DFT and plot the magnitude of the corresponding Fourier coefficients
# F2_img_gray = np.fft.fft2(img_gray.astype(float))
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# Y = (np.linspace(-int(img_gray.shape[0]/2), int(img_gray.shape[0]/2)-1, img_gray.shape[0]))
# X = (np.linspace(-int(img_gray.shape[1]/2), int(img_gray.shape[1]/2)-1, img_gray.shape[1]))
# X, Y = np.meshgrid(X, Y)
#
# # Standard plot: range of values makes small differences hard to see
# # ax.plot_surface(X, Y, np.fft.fftshift(np.abs(F2_img_gray)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
#
# # # Log(magnitude + 1) plot: shrinks the range so that small differences are visible
# # ax.plot_surface(X, Y, np.fft.fftshift(np.log(np.abs(F2_img_gray)+1)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
#
# # # Plot the magnitude as image
# # plt.show()
#
# # # Plot the magnitude and the log(magnitude + 1) as images (view from the top)
# # # magnitudeImage = np.fft.fftshift(np.abs(F2_img_gray))
# # # magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
# # # magnitudeImage = ski.img_as_ubyte(magnitudeImage)
# # # cv2.imshow('Magnitude plot', magnitudeImage)
# # logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_img_gray)+1))
# # logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
# # logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
# # cv2.imshow('Log Magnitude plot', logMagnitudeImage)
# # cv2.imwrite('img_Log_Magnitude_plot_ori.jpg', logMagnitudeImage)
# # cv2.waitKey(0)

##############################################
#4.2 Frequency Analysis

img_gray =  cv2.cvtColor(img_noise, cv2.COLOR_RGB2GRAY)
#img_gray = cv2.resize(img_gray, (0, 0), fx=0.5, fy=0.5)
small = img_gray
smallNoisy = smallNoisy = np.zeros(small.shape, np.float64)

noise = np.random.randn(small.shape[0], small.shape[1])
smallNoisy = np.zeros(small.shape, np.float64)
sigma = 0.2 * small.max()/noise.max()
# Color images need noise added to all channels
if len(small.shape) == 2:
    smallNoisy = small + sigma * noise
else:
    smallNoisy[:, :, 0] = small[:, :, 0] + sigma * noise
    smallNoisy[:, :, 1] = small[:, :, 1] + sigma * noise
    smallNoisy[:, :, 2] = small[:, :, 2] + sigma * noise


col = int(smallNoisy.shape[1]/2)
colData = small[0:smallNoisy.shape[0], col, 0]

F_colData = np.fft.fft(colData.astype(float))
# Plot the magnitude of the Fourier coefficients as a stem plot
# Notice the use off fftshift() to center the low frequency coefficients around 0
#xvalues = np.linspace(-int(len(colData)/2), int(len(colData)/2)-1, len(colData))
#markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData)), 'g')
xvalues = np.linspace(0, len(colData), len(colData))
markerline, stemlines, baseline = plt.stem(xvalues, (np.abs(F_colData)), 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.show()

print(np.abs(F_colData))

##############################################
#5.1
# graySmall = img_gray
# graySmall =cv2.cvtColor(img_noise, cv2.COLOR_RGB2GRAY)
# # Explore the Butterworth filter
# # U and V are arrays that give all integer coordinates in the 2-D plane
# #  [-m/2 , m/2] x [-n/2 , n/2].
# # Use U and V to create 3-D functions over (U,V)
# U = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
# V = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
# U, V = np.meshgrid(U, V)
# # The function over (U,V) is distance between each point (u,v) to (0,0)
# D = np.sqrt(X*X + Y*Y)
# # create x-points for plotting
# xval = np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1])
# # Specify a frequency cutoff value as a function of D.max()
# D0 = 0.25 * D.max()
#
# # The ideal lowpass filter makes all D(u,v) where D(u,v) <= 0 equal to 1
# # and all D(u,v) where D(u,v) > 0 equal to 0
# idealLowPass = D <= D0
#
# # Filter our small grayscale image with the ideal lowpass filter
# # 1. DFT of image
# print(graySmall.dtype)
# FTgraySmall = np.fft.fft2(graySmall.astype(float))
# # 2. Butterworth filter is already defined in Fourier space
# # 3. Elementwise product in Fourier space (notice fftshift of the filter)
# FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(idealLowPass)
# # 4. Inverse DFT to take filtered image back to the spatial domain
# graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
#
# # Save the filter and the filtered image (after scaling)
# idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
# graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
# #cv2.imwrite("idealLowPass.jpg", idealLowPass)
# #cv2.imwrite("grayImageIdealLowpassFiltered.jpg", graySmallFiltered)
#
# # Plot the ideal filter and then create and plot Butterworth filters of order
# # n = 1, 2, 3, 4
# plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
# colors='brgkmc'
# # for n in range(1, 5):
# #     # Create Butterworth filter of order n
# #     H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
# #     # Apply the filter to the grayscaled image
# #     FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(H)
# #     graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
# #     graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
# #     #cv2.imshow("grayImageButterworth-n" + str(n), graySmallFiltered)
# #     #cv2.imshow('H', H)
# #     #cv2.waitKey(0)
# #     # cv2.destroyAllWindows()
# #     H = ski.img_as_ubyte(H / H.max())
# #     #cv2.imwrite("butter-n" + str(n) + ".jpg", H)
# #     # Get a slice through the center of the filter to plot in 2-D
# #     slice = H[int(H.shape[0]/2), :]
# #     plt.plot(xval, slice, colors[n-1], label='n='+str(n))
# #     plt.legend(loc='upper left')
#
# #plt.show()
# #plt.savefig('butterworthFilters.jpg', bbox_inches='tight')

################################################################
# #5.2 High pass
# n=10
# # Create Butterworth filter of order n
# H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
# # Apply the filter to the grayscaled image
# FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(H)
# graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
# graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
# #cv2.imshow("grayImageButterworth-n" + str(n), graySmallFiltered)
# #cv2.imshow('H', H)
# #cv2.waitKey(0)
# # cv2.destroyAllWindows()
# H = ski.img_as_ubyte(H / H.max())
# #cv2.imwrite("butter-n" + str(n) + ".jpg", H)
# # # Get a slice through the center of the filter to plot in 2-D
# # slice = H[int(H.shape[0]/2), :]
# # plt.plot(xval, slice, colors[n-1], label='n='+str(n))
# # plt.legend(loc='upper left')
#
# img_high_pass = img_gray - graySmallFiltered
# cv2.imshow('high pass', img_high_pass)
# cv2.imshow('low pass', graySmallFiltered)
# cv2.waitKey(0)
