#!/usr/bin/env python
# coding: utf-8

# GÖGÜS KANSERİ SINIFLANDIRMASI

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

# warning library
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data=pd.read_csv(r'C:\Users\medin\breast cancer\data.csv')


# In[3]:


data.drop(['Unnamed: 32','id'], inplace = True, axis = 1) 


# In[4]:


data


# In[5]:


data.isnull().sum()


# In[6]:


data = data.rename(columns = {"diagnosis":"target"}) #diagnosis isimli sutunun adını target olarak değiştirdim


# In[7]:


data.head()


# In[8]:


sns.countplot(data["target"])
print(data.target.value_counts()) #verilerimin kaç adet kanser olduğunu gösterdi


# In[9]:


data["target"] = [1 if i.strip() == "M" else 0 for i in data.target] #datamın target içerisindeki harf değerleri sayısallaştırdım


# In[10]:


print(len(data)) #data sayısı (uzunluğu)


# In[11]:


print("Data shape ", data.shape) #satır sütun sayısı


# In[12]:


data.info() 


# In[13]:


data.describe()


# 1 ve 5. sütunlarda gördüğümüz gibi çok büyük sayı farkı var bunu ayarlamamız gerekiyor. çünkü büyük sayılar küçük sayılara baskın gelebilir

# KEŞİFSEL VERİ ANALİZİ (exploratory data analysis; EDA)

# In[14]:


# Correlation #nümerik değerlere sahip bir setim olduğu için korelasyon uyguladım. #korelasyon haritası çalış
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Veri Setinin Korelasyon Grafiği")
plt.show()


# In[15]:


threshold = 0.75 #sadece 0.75 den yüksek değerleri göster demek
filtre = np.abs(corr_matrix["target"]) > threshold #targeti al threshold ile karşılaştır
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Korelasyon Grafiği Threshold 0.75")


# In[16]:


# box plot 
data_melted = pd.melt(data, id_vars = "target",
                      var_name = "features",
                      value_name = "value")


# In[17]:


plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


# pair plot 
sns.pairplot(data[corr_features], diag_kind = "kde", markers = "+",hue = "target")
plt.show()


# OUTLİER DETECTİON (LOCAL OUTLİER) #aykırı değer tespiti

# local outlier algoritması belirli bir veri nokasının komşularına göre yerel yoğunluk sapması hesaplayan denetimsiz bir algoritmadır.

# In[ ]:


# Outlier (Aykırı Değer Tespiti)
y = data.target    #datamı x ve y olarak ayırdım
x = data.drop(["target"],axis = 1) #target sütununu çıkartıyorum sayısal değerler olmadığı için
columns = x.columns.tolist() #öz nitelikleri colums içine depoladık 


# In[ ]:


clf = LocalOutlierFactor() 
y_pred = clf.fit_predict(x)
X_score = clf.negative_outlier_factor_


# In[ ]:


outlier_score = pd.DataFrame()
outlier_score["score"] = X_score


# In[ ]:


# threshold
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()


# In[ ]:


plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1],color = "blue", s = 50, label = "Outliers")
plt.scatter(x.iloc[:,0], x.iloc[:,1], color = "k", s = 3, label = "Data Points")


# In[ ]:


radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0], x.iloc[:,1], s = 1000*radius, edgecolors = "r",facecolors = "none", label = "Outlier Scores")
plt.legend()
plt.show()


# In[ ]:


# drop outliers
x = x.drop(outlier_index)
y = y.drop(outlier_index).values


# In[ ]:


# Veris setini %70 eğitim ve %30 test verisi olarak ikiye böldüm
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# Standardizasyon

# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe()
X_train_df["target"] = Y_train


# In[ ]:


# Standartizasyondan sonra elde ettiğim veriyi görselleştirdim
X_train_df.head()


# In[ ]:


# Standartizasyon sonrası veri setimin istatistiksel özelliklerini görselleştirdim.
X_train_df.describe()


# In[ ]:


# Tekrardan box plot ile öznitelik değerlerinin nasıl dağıldığını görselleştirdim.
data_melted = pd.melt(X_train_df, id_vars = "target",
                      var_name = "features",
                      value_name = "value")
plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()


# KNN MODELİNİN OLUŞTURULMASI

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)  #BAŞARI ORANINI TEST VERİ SETİ İLE ELDE ETTİK
acc = accuracy_score(Y_test, y_pred)
print("KNN modelinin başarı oranı: ",acc)
print("Confisuon matrix: \n",cm)


# 109 tane kötü huylu kanserin 106 tanesini doğru tahmin ettik, 62 tane iyi huylu kanserin 57 tanesini iyi huylu olarak tahmin ettik.
# 8 değeri yanlış tahmin ettik. Bu yüzden başarı oranım %95.3 olarak geldi. 

# 8 adet yanlış tahmin yüzde %4.7 oranında başarımı düşürdü.

# In[ ]:


class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


sns.heatmap(pd.DataFrame(cm), annot = True, cmap = 'YlGnBu',
           fmt ='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('KNN Karışıklık Matrisi', y = 1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmini Değerler')
plt.show()


# KNN EN İYİ PARAMETRE BULMA

# In[ ]:


sayac = 1
for i in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors = i)
    knn_yeni.fit(X_train, Y_train)
    print(sayac, " ", "Dogruluk orani %", knn_yeni.score(X_test,Y_test)*100)
    sayac += 1


# En yüksek başarıyı 6 komşulukta aldım ve yeni bir KNN modeli eğittim.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
cm_knn = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
print("KNN modelinin başarı oranı: ",acc)
print("Confisuon matrix: \n",cm_knn)


# In[ ]:


class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


sns.heatmap(pd.DataFrame(cm_knn), annot = True, cmap = 'YlGnBu',
           fmt ='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('KNN En İyi Parametre Karışıklık Matrisi', y = 1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmini Değerler')
plt.show()


# In[ ]:


# PCA ile boyut indirgeme yaparak iki boyutlu bir veri elde ediyoruz sonra 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components = 2)
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca, columns = ["p1","p2"])
pca_data["target"] = y
sns.scatterplot(x = "p1", y = "p2", hue = "target", data = pca_data)
plt.title("PCA: p1 vs p2")


X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca, y, test_size = 0.3, random_state = 42)


# In[ ]:


# Boyut indirgeme ile elde ettiğim veri için bir KNN modeli daha eğittim. Önce en iyi parametreyi buldum.
sayac = 1
for i in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors = i)
    knn_yeni.fit(X_train_pca, Y_train_pca)
    print(sayac, " ", "Dogruluk orani %", knn_yeni.score(X_test_pca,Y_test_pca)*100)
    sayac += 1


# In[ ]:


# En iyi başarı oranını 5 komşulukta elde ettim ve bir model eğittim confisuon matrisini yazdırdım.
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_pca, Y_train_pca)
y_pred_pca = knn.predict(X_test_pca)
cm_pca = confusion_matrix(Y_test_pca, y_pred_pca)
acc = accuracy_score(Y_test_pca, y_pred_pca)
print("KNN modelinin başarı oranı: ",acc)
print("Confisuon matrix: \n",cm_pca)


# In[ ]:


class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


sns.heatmap(pd.DataFrame(cm_pca), annot = True, cmap = 'YlGnBu',
           fmt ='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('PCA KNN Karışıklık Matrisi', y = 1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmini Değerler')
plt.show()


# In[ ]:


Y_test_pca


# In[ ]:


y_pred_pca


# In[ ]:




