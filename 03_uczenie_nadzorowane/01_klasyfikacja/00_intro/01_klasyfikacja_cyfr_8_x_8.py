# -*- coding: utf-8 -*-
# import bibliotek
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(1)

# %% pobranie danych
# dane przedstawiają 1797 ręcznie zapisanych cyfry w rozmiarze 8x8 pikseli
data_raw = load_digits()
data = data_raw.data
images = data_raw.images
target = data_raw.target

# %% wyświetlenie danych
print('Rozmiar danych:', data.shape)
print('Pierwszy element:\n', data[0])

print('Pierwszy obraz:\n', images[0])

# %% wyświetlenie obrazów, domyślna mapa kolorów
plt.figure()
for idx, (image, label) in enumerate(list(zip(images, target))[:15]):
    # print(idx, image, label)
    plt.subplot(3, 5, idx + 1)
    plt.imshow(image)
    plt.title(f'Cyfra: {label}')
plt.show()
    
# %% wyświetlenie obrazów, skala szarości
plt.figure()
for idx, (image, label) in enumerate(list(zip(images, target))[:15]):
    # print(idx, image, label)
    plt.subplot(3, 5, idx + 1)
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.title(f'Cyfra: {label}')    
plt.show()

# %% przygotowanie danych
X = data.copy()
y = target.copy()

# %% podział danych na zbiór treningowy i testowy    
X_train, X_test, y_train, y_test = train_test_split(X, y)    

# %% utworzenie klasyfikatora
classifier = SVC(gamma=0.001)

# %% trenowanie klasyfikatora
classifier.fit(X_train, y_train)

# %% dokładność modelu
print(classifier.score(X_test, y_test))

# %% predykcja na podstawie modelu
y_pred = classifier.predict(X_test)

# %% wyświetlenie kilku predykcji
test_images = X_test.reshape((450, 8, 8))
for idx, (image, label) in enumerate(list(zip(test_images, y_pred))[:15]):
    # print(idx, image, label)
    plt.subplot(3, 5, idx + 1)
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.title(f'Predykcja: {label}')    
plt.show()

# %% wyświetlenie błędnych predykcji
#incorrect = []
#for i, j in zip(enumerate(y_test), enumerate(y_pred)):
#    if i[1] != j[1]:
#        incorrect.append(i[0])
        
incorrect = [i[0] for i, j in zip(enumerate(y_test), enumerate(y_pred)) if i[1] != j[1]]        
     
idx_image = 1        
for idx, (image, label) in enumerate(list(zip(test_images, y_pred))):
    
    if idx in incorrect:
        # print(idx, image, label)
        plt.subplot(3, 5, idx_image)
        plt.imshow(image, cmap=plt.cm.gray_r)
        plt.title(f'y_test: {y_test[idx]} y_pred: {label}')    
        idx_image += 1
plt.show()       

# %% ocena modelu, raport klasyfikacji
print('Raport klasyfikacji:')
print(classification_report(y_test, y_pred))


# %% ocena modelu, macierz konfuzji
print('Macierz konfuzji/Macierz pomyłek:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.set()
plt.figure()
sns.heatmap(cm, cmap=sns.cm.rocket_r)