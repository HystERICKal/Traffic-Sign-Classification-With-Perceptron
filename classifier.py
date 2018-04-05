# Name  : Halil İbrahim Bestil
# ID    : 15011013
# Bulunan karakteristik özellikler : 
#   - Tabelada mavi renk bulunması
#   - Tabela şekli (Üçgen,Daire)  

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Resimde bulunan shape'i tespit eder. Daire/Üçgen tipinde döndürür
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
# Resim içerisinde mavi ve kırmızı renkleri var/yok biçiminde döndürür
def getExistingColors(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)
    hist = find_histogram(clt)
    r,b = getColorSimilarity(hist, clt.cluster_centers_)
    return r,b
# Renk benzerliğini bulan fonksiyon
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
# iki rgb renk arasındaki euclidian farkını bulur
def ColorDistance(rgb1,rgb2):
    # Distance between two rgb colors
    d = abs((float(rgb1[0])-float(rgb2[0]))**2 + (float(rgb1[1])-float(rgb2[1]))**2 +(float(rgb1[2])-float(rgb2[2])))**(1/float(2))
    #print("d" ,d)
    return d  
# Resimdeki renklerin tespiti için ara fonksiyon
def find_histogram(clt):
    """
    create a histogram with k clusters
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist 
# Test fonksiyonu
def test(w,testdata):
    t = np.array(testdata)
    return t.dot(w) >= 0    
# Perceptron algoritması
def perceptron(X, Y,epochs):
    w = np.zeros(len(X[0]))
    eta = 1

    for t in range(epochs):
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                w = w + eta*X[i]*Y[i]

    return w
# Dosyadan resmi okur ve preprocess aşamalarını yapar
def getdata(path):
    # Read image
    img = cv2.imread(path)
    # Detect shape
    shape = 1 if(getImageShape(img) == "circle") else 0
    # Detect colors (dominantcolor = 1 eğer resimde mavi renk varsa)
    r,b = getExistingColors(img)
    dominantColor = 1 if(b==1) else 0;
    return [shape,dominantColor,-1]
# Epoch değeri 50 için confusion matrixini yazdırır
def main():
    trdata = []
    trlabel = []

    ## Get signs for training dataset
    for i in range (1,13):
        path = "./trafikisaretleri/egitim/tehlike-uyari/"+str(i)+".png"
        trdata.append(getdata(path))
        trlabel.append(1)
    for i in range (1,13):
        path = "./trafikisaretleri/egitim/parketme-durma/"+str(i)+".png"
        trdata.append(getdata(path))
        trlabel.append(-1)

    ## Train the perceptron
    X = np.array(trdata)
    w = perceptron(X,trlabel,50)
    print(w)

    y_true = []
    y_pred = []
    ## Get signs for test dataset
    for i in range (17,32):
        path = "./trafikisaretleri/test/tehlike-uyari/"+str(i)+".png"
        y_true.append(True);
        y_pred.append(test(w,getdata(path)))
    for i in range (14,20):
        path = "./trafikisaretleri/test/parketme-durma/"+str(i)+".png"
        y_true.append(False);
        y_pred.append(test(w,getdata(path)))

    ## Print Confusion Matrix
    print("epoch değeri 50 olduğu durum için başarı matrisi:")
    print("(t-u :tehlike-uyari , p-d: parketme-durma )")
    x,y = confusion_matrix(y_true, y_pred,labels=[True,False])
    print("     Tahmin \ Gerçek  |  t-u  |  p-d")
    print("     t-u              |  ",x[0]," |  ",x[1])
    print("     p-d              |  ",y[0],"  |  ",y[1])
# Ödevde istenen 1den 50 ye kadar olan eğitim sonuçlarını yazdırır
def getPredictionForEpochValue():
    trdata = []
    trlabel = []

    ## Get signs for training dataset
    for i in range (1,13):
        path = "./trafikisaretleri/egitim/tehlike-uyari/"+str(i)+".png"
        trdata.append(getdata(path))
        trlabel.append(1)
    for i in range (1,13):
        path = "./trafikisaretleri/egitim/parketme-durma/"+str(i)+".png"
        trdata.append(getdata(path))
        trlabel.append(-1)

    ## Train the perceptron
    X = np.array(trdata)

    epoch =[]
    result =[]
    for i in range(1,50):
        print(i," epoch için değerlendirme yapılıyor")
        epoch.append(i)
        w = perceptron(X,trlabel,i)
        y_true = []
        y_pred = []
        ## Get signs for test dataset
        for i in range (17,32):
            path = "./trafikisaretleri/test/tehlike-uyari/"+str(i)+".png"
            y_true.append(True);
            y_pred.append(test(w,getdata(path)))
        for i in range (14,20):
            path = "./trafikisaretleri/test/parketme-durma/"+str(i)+".png"
            y_true.append(False);
            y_pred.append(test(w,getdata(path)))
       
        result.append( accuracy_score(y_true, y_pred))

    plt.plot(epoch, result)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('1 - 50 arası eğitim sonuçları')
    plt.show()



main()  
getPredictionForEpochValue()
