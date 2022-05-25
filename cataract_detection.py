import os

from collections import defaultdict

from skimage import io
import UI


import cv2 as cv
import numpy as np
from io import BytesIO

from PIL import Image,ImageTk
import tkinter as tk 
import tkinter.filedialog
root = tk.Tk()
root.title("CATARACT DETECTION SYSTEM")
# specify size of window.
root.geometry("500x500")
 
 
# Create label
l = tk.Label(root, text = "CATARACT DETECTOR",width=500,height=1,bg='blue',fg='black')
l.pack()

# Create label
l = tk.Label(root, text = "UPLOAD IMAGES TO CHECK CATARACT",width=500,height=1,bg='lightblue',fg='black')
l.pack()
 
# Create button for upload.
b1 = tk.Button(root, text = "UPLOAD",command=lambda:upload_file() )

b1.pack()
l#b=tk.Label(root)
#lb.pack()
 # Create text widget and specify size.

def upload_file():
   
    file=tkinter.filedialog.askopenfilename(initialdir=os.getcwd(),title="select image")
    img=Image.open(file,'r')
    img = img.resize((100,100))
    img=ImageTk.PhotoImage(img)
    lb=tk.Label(root)
    
    lb.configure(image=img)
    lb.pack()

    
    image = io.imread(file)
    b1 = tk.Button(root, text = "DETECT",command=lambda:result(image) )
    b1.pack()
def result(image):
    r_min = 10
    r_max = 200
    delta_r = 1
    num_thetas = 100
    bin_threshold = 0.4
    min_edge_threshold = 100
    max_edge_threshold = 200 
    
    #
    dst_IMG= preprocessing(image)
    height,width = dst_IMG.shape
    mask = np.zeros((height,width), np.uint8)
       
    dst_IMG= dst_IMG.astype(np.uint8)
    edge_image = cv.Canny(dst_IMG, min_edge_threshold, max_edge_threshold)
        
    if edge_image is not None:
       circle_img, circles = find_hough_circles(dst_IMG, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold)      
    #cv2_imshow(circle_img)
    #cv.imwrite('imageafterHOUGH1.jpeg', circle_img)

    for i in circles[0::]:
        cv.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)

    masked_data = cv.bitwise_and(dst_IMG, dst_IMG, mask=mask)

    # Apply Threshold
    _,thresh = cv.threshold(mask,1,255,cv.THRESH_BINARY)

    # Find Contour
    contours, _ = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv.boundingRect(contours[0])

    # Crop masked_data
    cropedimg= masked_data[y:y+h,x:x+w]


    #cv2_imshow(cropedimg) 
    #cv.imwrite('imageafterMASKING1.jpeg', cropedimg)

#"""**Features extraction**"""

    # getting mean value
    mean = np.mean(cropedimg)  
    # printing mean value
    #print("Mean Value for image : " ,mean)
    # getting variance
    variance = np.var(cropedimg)
    # printing varince
    #print("variance for image : " ,variance)
#based on texturefeatures of images, calculated the threshold value for the mean intensity and variance 
    #using diagnostic opinion-based parameter thresahold as
    mean_threshold=55.2 
    var_threshold=2200

    if mean >= mean_threshold and variance >= var_threshold:
    #cv2_imshow(roi_img)
        percent,msg=percentcalculate(cropedimg)
        message=("Total percenatge of affected area:" +str(percent)+"\nstage:"+str(msg))
       
        
    else:
    #cv2_imshow(roi_img)
        message="Healthy eyes"
   
    #Fact = """A man can be arrested in Italy for wearing a skirt in public."""
    T = tk.Text(root, height = 5, width = 52)
    T.pack()
  
    # Insert The Fact.
    T.insert(tk.END, message)
 
#tk.mainloop()
    

 # """ Gaussian Kernel Creator via given length and sigma"""
def gkernel(l=5, sig=1):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = (1/(2*3.14*np.square(sig)))*(np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig)))
    return kernel / np.sum(kernel)


def find_hough_circles(img, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process = True):
    img_height, img_width = edge_image.shape[:2] #image size
    dtheta = int(360 / num_thetas)# R and Theta ranges
    thetas = np.arange(0, 360, step=dtheta)
    rs = np.arange(r_min, r_max, step=delta_r)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    circle_candidates = []
    for r in rs:
        for t in range(num_thetas):
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
    accumulator = defaultdict(int)

    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y][x] != 0: #white pixel
        # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = x - rcos_t
                    y_center = y - rsin_t
                    accumulator[(x_center, y_center, r)] += 1

    output_img = img.copy()

    out_circles = []

    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold: 
      # Shortlist the circle for final result
            out_circles.append((x, y, r, current_vote_percentage))

    if post_process :
        pixel_threshold = 5
        postprocess_circles = []
        for x, y, r, v in out_circles:
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
                postprocess_circles.append((x, y, r, v))
        out_circles = postprocess_circles

    for x, y, r, v in out_circles:
        output_img = cv.circle(output_img, (x,y), r, (0,255,0), 2)

    return output_img, out_circles
def preprocessing(img):
    global dst_IMG
    if (img.shape[0] > 120 and img.shape[1] >120):
      roi_img = cv.resize(img,(120,120)) #resizing 120*120 pixels
    else:
        roi_img= img
    R, G, B = roi_img[:,:,0], roi_img[:,:,1], roi_img[:,:,2] #get the red, blue and green dimension matrices of the RGB image 
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    g_kernel = gkernel(5,1)
    dst_IMG = cv.filter2D(imgGray,-1,g_kernel)
    return dst_IMG
   
import skimage.filters
def percentcalculate(image):
    t = skimage.filters.threshold_otsu(image)
        # create a binary mask with the threshold found by Otsu's method
    binary_mask = dst_IMG > t
        #plt.imshow(binary_mask, cmap='gray')
        #plt.show()
    affectedPixels = np.count_nonzero(binary_mask)
    totalpixels= binary_mask.size
        #print("Total pixels of image:",totalpixels)
        #print("Total affected pixels of image:",affectedPixels)
    affectedpercenatge=(affectedPixels/totalpixels)*100
        #print("Total percenatge of affected  pixels of image:",affectedpercenatge)
    if affectedpercenatge  > 0 and affectedpercenatge < 10:
        message="MILD Stage Cataract"
    elif  affectedpercenatge  > 10 and affectedpercenatge < 50:
        message="MODERATE stage Cataract"
    elif affectedpercenatge  > 50 and affectedpercenatge < 90:
        message="PRONOUNCED stage Cataract"
    else: 
        message="SEVERE stage Cataract"
    return affectedpercenatge,message

tk.mainloop() 
    
    

