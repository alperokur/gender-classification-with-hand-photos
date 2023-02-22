'''

    Bu projede 11K Hands veritabanı kullanılarak, el fotoğrafları üzerinden cinsiyet belirleme işlemi üzerinde çalışılmıştır.
    İki nokta arasındaki uzaklık yöntemi kullanılarak sınıflandırma sonucunda yaklaşık olarak %94-95 oranında bir doğruluk payı alınmıştır.

'''

'''

    Uygulama adımları:

    11k Hands sitesi üzerinden (https://sites.google.com/view/11khands); 
        veritabanı fotoğrafları (https://drive.google.com/open?id=1KcMYcNJgtK1zZvfl_9sTqnyBUTri2aP2),
        fotoğrafların bilgilerini içeren (https://drive.google.com/file/d/1RC86-rVOR8c93XAfM9b9R45L7C2B0FdA/view?usp=sharing)
    dosyalar indirilip aynı klasör içerisine verildi.

    Fotoğrafların bilgilerini içeren HandInfo.csv dosyasından;
        id,age,gender,skinColor,accessories,nailPolish,aspectOfHand,imageName,irregularities
        "0000000",27,"male","fair",0,0,"dorsal right","Hand_0000002.jpg",0
        ...

    fotoğrafların dosya isimleri ("Hand_0000002.jpg") ve cinsiyetleri ("male") bilgileri okunup dizilere atandı. 
    
    Yine dosya isimlerinden fotoğraflar okunup, fotoğraflar üzerinde her eklem noktası arasındaki uzaklık tespit edildi ve bu uzaklıklarda diziye atandı.

    Dosya ismi, cinsiyet ve eklem noktaları arasındaki uzaklık dizileri birleştirilip "data.csv" adındaki dosyaya yazdırıldı.
    Bu yazdırma işlemini noktalar arasındaki uzaklık işlemi uzun sürdüğü için yaptık ki sınıflandırma kısmında bu işlemi yapmadan,
    sadece "data.csv" üzerindeki dosya içeriğini okuyarak hızlı bir şekilde sınıflandırma yapabileceğiz.

    SVM'de linear kernel kullanılarak "data.csv" dosyasındaki eklem noktaları arasındaki uzaklıklar X_train, cinsiyetler ise y_train olarak ayarlandı.
    Tabi bunların bir kısmını 'train_test_split' ile yüzdesel olarak ayırdık. Örneğin 10.000 tane dosya var ise %25 ini test olarak verdik.
    SVM sonucunda yüzdesel bir tahmini değer elde edip bunu çıktı olarak verdik.

'''

import csv # ".csv" uzantılı veri dosyalarını okuyabilmemiz için gerekli olan kütüphane
import cv2 # Fotoğrafları okuyabilmemiz için gerekli olan opencv kütüphanesi
import math # Eklem noktalarının uzunluğunu hesaplarken karekök formülü için kullandığımız kütüphane
import matplotlib.pyplot as plt
import mediapipe as mp # El üzerinde eklem noktalarının koordinatlarını belirlemeye yarayan kütüphane
import numpy as np
import os # Dosya uzantıları için kullanmış olduğumuz kütüphane
import pandas as pd # Veri işlmesi için kullandığımız kütüphane
import time
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split # Eğitim ile testleri aynı anda verebilmemiz için gerekli olan kütüphane
from sklearn.svm import SVC # Sınıflandırma için gerekli olan kütüphane
from sklearn.tree import DecisionTreeClassifier

imageFolder = 'Hands/' # Veritabanı fotoğraflarının bulunduğu klasörün uzantısı
infoFileName = 'HandInfo.csv' # Veritabanındaki fotoğrafların bilgilerini içeren dosyanın ismi
dataFileName = 'data.csv' # Fotoğraflar üzerinde işlem gerçekleştirdikten sonra oluşturduğumuz veri dosyasının ismi

infoFile = pd.read_csv(infoFileName) # "HandInfo.csv" dosyasını pandas kütüphanesi ile okuduk.
imageName = infoFile.iloc[:, 7].values # okunan dosyada 8. sütundaki değerleri fotoğraf dosyalarının isimleri olarak, 'imageName' dizisine atadık.
gender = infoFile.iloc[:, 2].values # okunan dosyada 3. sütundaki değerleri cinsiyet olarak, 'gender' dizisine atadık.
mp_hands = mp.solutions.hands # mediapipe kütüphanesi ile el üzerinde işlem yapmayı yarayan kod
hand = mp_hands.Hands(static_image_mode = True, max_num_hands=1) # eğer fotoğraf sabit ise, static_image_mode'u True; eğer fotoğraf sabit değil ise (webcam) static_image_mode'u False yaptık.

names = [
    "Extra Trees",
    "Hist Gradient Boosting",
    "Random Forest",
    "Bagging",
    "Decision Tree Forest",
    "SVC Poly",
    "Ada Boost",
    "SVC RBF",
    "SVC Linear",
]

classifiers = [
    ExtraTreesClassifier(),
    HistGradientBoostingClassifier(),
    RandomForestClassifier(),
    BaggingClassifier(),
    DecisionTreeClassifier(),
    SVC(kernel='poly'),
    AdaBoostClassifier(),
    SVC(kernel='rbf'),
    SVC(kernel='linear'),
]

def HandPoints(image_path): # Dosya uzantısından fotoğrafların üzerindeki noktaları diziye atan fonksiyon
    image = cv2.imread(image_path) # Dosya uzantısından fotoğrafı opencv kütüphanesi ile okuduk
    results = hand.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # mediapipe kütüphanesi ile fotoğraf üzerinde eklem noktalarının koordinatları belirleme işlemi gerçekleştirilip 'results' adındaki değişkene atandı.
    distance_between_two_points = [] # iki nokta arasındaki uzaklıkları tuttuğumuz dizi
    if results.multi_hand_landmarks: # eğer el üzerinde eklem koordinatları belirlendi ise:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks): # Fotoğraftaki el sayısı için:
            for j in range(len(mp_hands.HandLandmark)-1): # eklem noktalarının bir eksik sayısı kadar j değeri:
                for k in range(j+1,len(mp_hands.HandLandmark)): # eklem noktarlının sayısı kadar da k değeri:
                    # (0-1, 0-2, ..., 1-2, 1-3, ... 19-20) diye giden eklem koordinatları birbirinden çıkartılıp, uzunlukları √(x^2+y^2+z^2) işlemine göre bulunuyor.
                    # bulunan uzunluklar 'distance_between_two_points' adındaki diziye ekleniyor
                    length = math.sqrt(
                        math.pow(hand_landmarks.landmark[mp_hands.HandLandmark(j)].x-hand_landmarks.landmark[mp_hands.HandLandmark(k)].x, 2) + 
                        math.pow(hand_landmarks.landmark[mp_hands.HandLandmark(j)].y-hand_landmarks.landmark[mp_hands.HandLandmark(k)].y, 2) +
                        math.pow(hand_landmarks.landmark[mp_hands.HandLandmark(j)].z-hand_landmarks.landmark[mp_hands.HandLandmark(k)].z, 2))
                    distance_between_two_points.append(length)
    return distance_between_two_points # dizi geri döndürülüyor

def CreateFile(name): # data.csv dosyasını oluşturma fonksiyonu
    with open(name, 'w') as f: # "data.csv" dosyasını yazma modunda açtık
        c = csv.writer(f) # dosyaya yazmak için csv kütüphanesindeki writer fonksiyonunu hazırladık
        for i in range(len(infoFile.index)): # 
            data = [imageName[i], gender[i]] # Fotoğraf ismi ile cinsiyet bilgisini 'data' isminde bir diziye attık
            points = HandPoints(imageFolder + imageName[i]) # Fotoğraftaki iki nokta arasındaki uzaklıkları bulup bunları 'points' adında bir diziye aktardık.
            for j in range(len(points)): # "points" dizisinin satır sayısı kadar:
                data.append(points[j]) # 'data' ismindeki diziye bu noktaları yanyana olacak şekilde ekledik. (Örn: Hand_0000002.jpg,male,0.15296173890108972,0.3050357301295396,...)
            if len(data) > 2: # Eğer ki 'data' dizi uzunluğu 2'den büyük ise: -> Buradaki amaç girdi olarak vermiş olduğumuz fotoğrafta eklem noktalarının tespiti yapılamadı ise sınıflandırmanın doğru yapılabilmesi için o fotoğrafı es geçmek (Hand_0000002.jpg'den Hand_0000004.jpg'e atlamak gibi düşün)
                c.writerow(data) # Fotoğraftaki iki nokta arasındaki uzaklıkları bulup, satır satır "data.csv" dosyasına yazdırdık.
            print('Process: %' + f"{100/(len(infoFile.index)/(i+1)):.2f}") # Burada yazdırma işlemi yapılırken, işlemin % kaç olduğunu terminal ekranında çıktı olarak gösterdik.

def SVM(clf_name, clf_value):
    start = time.time()
    clf = clf_value
    clf.fit(X_train, y_train)
    end = time.time()
    print(f"Classifiers: {clf_name:<24} | Predication: {(clf.score(X_test,y_test)):.2%} | Time: {(end-start):<8}")

if not os.path.exists(dataFileName): # Eğer "data.csv" dosyası yok ise (dosyayı oluşturmayı kastediyoruz)
    CreateFile(dataFileName)
df = pd.read_csv(dataFileName) # "data.csv" dosyasını pandas kütüphanesi ile okuduk.
X = df.iloc[:, 2:].values # "data.csv" dosyasındaki 3. sütundan (0,1,2) son sütuna kadar iki nokta arasındaki uzunluk değerlerini 'X' adındaki diziye aktardık.
y = df.iloc[:, 1].values # "data.csv" dosyasındaki 2. sütundaki (0,1) cinsiyet değerlerini 'y' adındaki diziye aktardık.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # Aktardığımız değerlerin %75'i train (eğitim), %25'i de test olarak ayarlandı.

while True:
    for i in range(len(classifiers)):
        SVM(names[0], classifiers[0])
        SVM(names[2], classifiers[2])
        SVM(names[5], classifiers[5])
