# This is tuned for edge detection.
import cv2
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


def preProcessImg(img):
    #remove background
    gray = img.copy()
    # for Black Background use THRESH_BIN
    #ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # for white Background use THRESH_BIN_INV
    ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.medianBlur(otsu, 5)
    # fill the holes inside
    im2, contours, hierarchy = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(otsu, [cnt], -1, 255, -1)
    binImg = cv2.bitwise_and(img, otsu)
    #cv2.imshow('otsu', otsu)
    #cv2.waitKey(10)
    return binImg, otsu


# this function is needed for the createTrackbar step downstream
def detect_edge(resz,kernel=5, scale=0.50,thresh=50):
    gray = cv2.cvtColor(resz, cv2.COLOR_BGR2GRAY)
    gray, otsu = preProcessImg(gray)
    grayDN = cv2.fastNlMeansDenoising(gray, h=3,templateWindowSize=7)
    #grayDN = denoise_tv_chambolle(gray, weight=0.1, multichannel=False)
    #grayDN = denoise_tv_bregman(gray, weight=1)

    #grayDN = denoise_wavelet(gray, multichannel=False) # good for finger prints
    #Normalize 0-255 if using scimg
    grayDN = cv2.normalize(grayDN, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    hist = cv2.equalizeHist(grayDN)

    edgeL = cv2.Laplacian(hist, cv2.CV_8U,ksize=kernel,scale=scale)
    #edgeL = cv2.adaptiveThreshold(hist, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                              cv2.THRESH_BINARY_INV, 7, 5)

    edgeL = postProcessImg(edgeL,thresh)

    return edgeL, otsu



def postProcessImg(img1,thresh=100):
    ret, bin = cv2.threshold(img1, thresh, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cnt.size / 2 <= 5:
            cv2.drawContours(bin, [cnt], -1, 0, -1)
            continue

        ellipse = cv2.fitEllipse(cnt)
        e_w = ellipse[1][0]
        e_l = ellipse[1][1]
        if e_l < 15:
            cv2.drawContours(bin, [cnt], -1, 0, -1)
    return bin

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


# this function gets solid mask using color mask
def getSolidClrMask(img, H=70,S=30,V=80):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # skin mask
    lower = np.array([0, S, V], dtype="uint8")
    upper = np.array([H, 255, 255], dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)
    #cv2.imshow("mask",mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    #cv2.imshow("img",img)
    #cv2.imshow("Fixed",mask)
    #cv2.waitKey(0)
    #plt.imshow(hsv, interpolation='none')
    #plt.show()
    return mask

  # this function gets solid mask using adaptive threshold
def getSolidAdptv(img):
    gray = cv2.cvtColor(resz, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(5,5))
    H,W = gray.shape
    mask = np.zeros((H,W), dtype = np.uint8)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 13, 3)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    cv2.imshow("mask", thresh)
    cv2.waitKey(0)

    return

def watershed(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    cv2.imshow("watershed", img)
    cv2.waitKey(0)

def grabcuts(img):
    mask = getSolidClrMask(img)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    x, y, w, h = cv2.boundingRect(cnt)
    mask[mask==255]=0
    cv2.grabCut(img, mask, (x,y,w,h), bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    cv2.imshow("mask",mask2)
    cv2.waitKey(0)
    return


if __name__ == '__main__':
    # read the experimental image
    files = glob.glob("../../../projects/palm/images/11K_Hand/Fhand180/*")
    for file in files:
        #img = cv2.imread('../../../projects/palm/images/11K_Hand/Fhand/Hand_0000851.jpg', 1)
        img = cv2.imread(file, 1)
        resz = cv2.resize(img, None, fx=1.0 / 2, fy=1.0 / 2, interpolation=cv2.INTER_CUBIC)
        #getSolidAdptv(resz)
        getSolidClrMask(resz)
        #watershed(resz)
        #grabcuts(resz)
        #img1, thresh = detect_edge(resz)
        #cv2.imshow('Org',resz)
        #cv2.imshow('Edge',img1)
        #cv2.imshow('Thresh',thresh)
        #cv2.waitKey(0)
    cv2.destroyAllWindows()
