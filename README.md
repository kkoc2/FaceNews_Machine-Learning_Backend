# FaceNews_Machine-Learning_Backend

app.py dosyası flask serverda calistirilacak backend script'idir.

## Uygulama
Flask uygulaması tcp 5001 portunu kullanacak.

"/fit_PA" yada "/fit_LR" ve  "/predict_PA" yada "/predict_LR" 4 farkli cagirilabilecek API'imiz vardir.

fit api'i database'den tum datalari alir makine ogrenmesine sokar, test eder ve en son olarak da "trained_model_PA.pkl" yada "trained_model_LR.pkl" isminde model olusturur.

predict api'i ise tahminde bulunulmasi istenen title ve text icerigini alir ve olusturulan modeli uzerinden tahminleme yapar.

## API
## PassiveAggressiveClassifier
fit_PA :
RestAPI ile "Get" metodunu kullanir. Olusturulan modelin Accuracy ve F1-Score'unu liste olarak dondurur. 
Örn : (0.9549431321084865, 0.9478744939271256)

predict_PA :
RestAPI ile "POST" medodunu kullanir. Post edilecek 2 datayi bekler, title ve text datasi JSON formatinda gonderilmelidir.
Donus olarak tahminin sonucu dondurur. Örn : ("FAKE" ve "REAL" olarak)

## LogisticRegression
fit_LR :
RestAPI ile "Get" metodunu kullanir. Olusturulan modelin Accuracy ve F1-Score'unu liste olarak dondurur. 
Örn : (0.9549431321084865, 0.9478744939271256)

predict_LR :
RestAPI ile "POST" medodunu kullanir. Post edilecek 2 datayi bekler, title ve text datasi JSON formatinda gonderilmelidir.
Donus olarak tahminin sonucu ve olasiligi dondurur. Örn : ('RELIABLE', 99.98570855838544) %99 oraninda FAKE yada RELIABLE'dir.

## FLASK uygulamasının yüklenmesi ve çalıştırılması:
```bash
pip install --upgrade pip
pip install virtualenv
virtualenv fakenews
source fakenews/bin/activate
pip install flask

FLASK_APP=app.py flask run
```

## NLTK Kütüphanesi:
```python
Sistemde tek seferlik 'nltk' kutuphanesini guncelliyor.
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
```
