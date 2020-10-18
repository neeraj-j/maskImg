import numpy as np
import sys
import cv2
import pycocotools.mask as maskUtils
from rdp import rdp
import math
import libs.edgeDetect

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

#cv2 to qimage conversion
gray_color_table = [qRgb(i, i, i) for i in range(256)]

def qImgToCv2(img, copy=False):
    # img = self.image
    swap = True  # swap RGB
    if img.format() == QImage.Format_RGB888:
        depth = 3  # format = cv2.CV_8UC3
    elif img.format() == QImage.Format_Indexed8:
        depth = 1  # format = cv2.CV_8U
        swap = False
    elif img.format() == QImage.Format_RGB32 or \
                                      img.format() == QImage.Format_ARGB32 or \
                                      img.format() == QImage.Format_ARGB32_Premultiplied:
        # img = img.convertToFormat(QImage.Format_RGB888)
        depth = 4  # format = cv2.CV_8UC4

    ptr = img.constBits()
    ptr.setsize(img.byteCount())
    if depth == 1:
        cim = np.array(ptr).reshape(img.height(), img.width())
    else:
        cim = np.array(ptr).reshape(img.height(), img.width(), depth)
    # if swap:
    #    cim = cv2.cvtColor(cim, cv2.COLOR_RGB2BGR)
    return cim.clone() if copy else cim


def cv2ToQimg(im, copy=False):
    if im is None:
        return QImage()

    # im = self.cv2Image

    if im.dtype == np.uint8:
        if len(im.shape) == 2:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                # qim = qim.rgbSwapped()
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                # qim = qim.rgbSwapped()
                return qim.copy() if copy else qim

#Get pvertex and compute fit mask points
def getMaskPoints(img, qtPoints):
    points = []
    for point in qtPoints.points:
        x = int(point.x())
        y = int(point.y())
        points.append((x,y))
    L = len(points)
    #convert to n x 1 x 2 array
    points = np.asarray(points).reshape(L, 1, 2)
    x, y, w, h = cv2.boundingRect(points)
    xmins = x; ymins = y; xmaxs = x+w; ymaxs = y+h
    H, W, D = img.shape
    # clip the phlenge from the image
    mask = img[ymins:ymaxs, xmins:xmaxs, :]
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    pad = 7
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE )
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.medianBlur(gray, 5)

    ret, adpT = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    adpT = cv2.morphologyEx(adpT, cv2.MORPH_CLOSE, kernel, iterations=3)
    cv2.imshow("Gray", gray)
    cv2.imshow("adpT", adpT)
    cv2.waitKey(20)
    # remove pading
    h,w = adpT.shape
    adpT = adpT[pad:h - pad, pad:w - pad]
    #put mask back in image for correct dimentions
    mask = np.zeros((H,W), dtype=np.uint8)
    mask[ymins:ymaxs, xmins:xmaxs] = adpT
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    N,_,_ = hull.shape
    hull = hull.reshape(N,2)
    points = []
    for i in range(N):
        points.append(QPointF(hull[i][0], hull[i][1]))
    return points

def getBoxSegment(qmask, imagePath, hsv, maskFile):
    shpMask = qImgToCv2(qmask.toImage().convertToFormat(QImage.Format_Indexed8))
    img = cv2.imread(imagePath)
    imgMask = libs.edgeDetect.getSolidClrMask(img, hsv[0], hsv[1], hsv[2])
    fnlMask = cv2.bitwise_and(shpMask,imgMask)
    segment = maskUtils.encode(np.asfortranarray(fnlMask, dtype=np.uint8))
    im2, contours, hierarchy = cv2.findContours(fnlMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    fnlMask[fnlMask==1] = 255
    #cv2.imshow("FInal Mask", fnlMask)
    #cv2.waitKey(30)

    cv2.imwrite(maskFile,fnlMask)

    return segment, x, y, (x + w), (y + h)

def getSegment(qmask, maskFile):
    # This is to get mask from palm print
    if False:
        nparr = qImgToCv2(pixMap.toImage())
        gray = cv2.cvtColor(nparr, cv2.COLOR_RGBA2GRAY)
        im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.imshow("Orig", gray)
        for cnt in contours:
            if cv2.contourArea(cnt) < 14:
                    cv2.drawContours(gray,[cnt],-1,0,-1)

        #cv2.imshow("Processed", gray)
        #cv2.waitKey(0)
        # Generates polygon format Dont delete.. for future reference.
        #segment = np.flip(np.transpose(np.nonzero(gray)),axis=1).flatten().tolist()
        # RLE format for lines
        segment = maskUtils.encode(np.asfortranarray(gray, dtype=np.uint8))
        '''
        Tried multiple segments. Not working
        im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segment = []
        for cnt in contours:
            cntf = np.flip(cnt, axis=1)
            seg = cntf.flatten().tolist()
            segment.append(seg)
        '''
    # get bounding box for the mask
    nMask = qImgToCv2(qmask.toImage().convertToFormat(QImage.Format_Indexed8))

    segment = maskUtils.encode(np.asfortranarray(nMask, dtype=np.uint8))

    im2, contours, hierarchy = cv2.findContours(nMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get longest contour, No longer needed as we are using mask
    maxlen = 0
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > maxlen:
            maxlen = perimeter
            maxCnt = cnt
    x, y, w, h = cv2.boundingRect(maxCnt)
    #nMask = cv2.normalize(nMask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #cv2.imshow("Mask", nMask)
    #cv2.waitKey(30)

    # save pixmap to file for debugging
    file = QFile(maskFile)
    file.open(QIODevice.WriteOnly)
    # maskPixmap.save(file, "PNG")
    qmask.save(file, "PNG")

    return segment,x,y,(x+w),(y+h)

def detect_blur(image):
    img = qImgToCv2(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


