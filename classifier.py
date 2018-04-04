import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def train(traindata,trainlabel,epochs=100):
    w =  np.zeros(len(traindata[0]))
    eta = 1
    for epoch in range(0,epochs):
        for x,y in zip(traindata,trainlabel):
            r = x.dot(w) >= 0
            hata = y-r
            w+= eta * hata * x
    return w
def test(w,testdata):
    return testdata.dot(w) >= 0 
def getImageShape(img):
    
    # convert the resized image to grayscale, blur it slightly,
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 10000)
    # Find thresh
    ret,thresh = cv2.threshold(blurred,127,255,1)
    # Find contours
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Detect traffic sign type
    for cnt in contours:
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
    if len(approx) < 4:
        return "notcircle"
    elif len(approx) >=4:
        return "circle"
def getExistingColors(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)
    hist = find_histogram(clt)
    r,b = getColorSimilarity(hist, clt.cluster_centers_)
    return r,b
def getColorSimilarity(hist, centroids):
    threshold = 124
    red = np.array([ 255,0,0])
    blue = np.array([ 0,0,255])
    black = np.array([0,0,0])
    r = 0
    b = 0

    for (percent, color) in zip(hist, centroids):
        
        if(ColorDistance(color.astype("uint8"),red.astype("uint8")) < threshold ):
            # RED
            r = 1
        elif (ColorDistance(color.astype("uint8"),blue.astype("uint8")) < threshold ):
            #BLUE
            b = 1
        if(ColorDistance(color.astype("uint8"),black.astype("uint8")) < 20 ):
            #BLACK (BUT NOT BLUE)
            b = 0
        #else:
            #print("CANNOT RESOLVE COLOR")
        
    return r,b    
def ColorDistance(rgb1,rgb2):
    # Distance between two rgb colors
    d = abs((float(rgb1[0])-float(rgb2[0]))**2 + (float(rgb1[1])-float(rgb2[1]))**2 +(float(rgb1[2])-float(rgb2[2])))**(1/float(2))
    #print("d" ,d)
    return d  
def find_histogram(clt):
    """
    create a histogram with k clusters
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist 

def getdata(path,isTest = 0):
    # Read image
    img = cv2.imread(path)
    # Detect shape
    shape = 1 if(getImageShape(img) == "circle") else 0
    # Detect colors (dominantcolor = 1 eğer resimde mavi renk varsa)
    r,b = getExistingColors(img)
    dominantColor = 1 if(b==1) else 0;
    if (isTest):
        return np.array([shape,dominantColor])
    return np.array([[shape,dominantColor]])
    

def oldmain():
    ## Create train data
    traindata = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    trainlabel = np.array([0,0,0,1])
    ## Train 
    w = train(traindata,trainlabel,100)
    ## Create test data
    testdata = np.array([1,0,1])
    ## Print test result
    print("trdata\n",traindata)
    print("trlabel",trainlabel)
    print ( test(w,testdata)    )

def main():
    traindata = np.empty((0,2), int)
    label =[]
    # Get parketme-durma datas ( parketme-durma label is 0)
    for i in range (1,13):
        path = "./trafikisaretleri/egitim/parketme-durma/"+str(i)+".png"
        traindata = np.append(traindata,getdata(path), axis=0)
        label.append(0)
    # Get tehlike uyari datas ( tehlike-uyari label is 1)
    for i in range (1,16):
        path = "./trafikisaretleri/egitim/tehlike-uyari/"+str(i)+".png"
        traindata = np.append(traindata,getdata(path), axis=0)
        label.append(1)

    # Lets Train it
    trainlabel = np.array(label)

    w =  train(traindata,trainlabel,100)

    ## Testing
    path = "./trafikisaretleri/test/parketme-durma/20.png"
    ## Create test data
    testdata = getdata(path,1)
    print("daire,mavi",testdata)

    print("trdata\n",traindata)
    print("trlabel",trainlabel)
    ## Print test result
    print ( test(w,np.array([1,1]))    )


main()