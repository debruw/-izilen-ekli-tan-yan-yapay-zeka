Projemiz yapılan çizimi tanıyan bir yapay sinir ağı. 
Kullanıcının bir çizim yapması isteniyor ve yapay sinir ağımız yapılan çizimi tanımaya çalışıyor.
Projenin yapılmasındaki temel amaç yapay sinir ağlarının çalışma yapısının anlaşılması ve ortaya insanlar için eğlenceli bir oyun çıkarmak.

Yapay sinir ağını oluşturmak için Python kütüphanelerinden Numpy, Keras, Sklearn ve Matplotlib kullanıldı.
Web altyapısı için HTML, CSS, Javascript, JSON ve Python web framework'ü olan Flask kullanıldı.

Nasıl çalışıyor?
1. Kullanıcıdan bir resim çizmesi isteniyor. Kullanıcı resmi çizerken javascript kodumuz kanvas üzerinde çizilen noktaların koordinatlarını kaydediyor.
2. Kaydedilen bu noktalar her 250 ms’de bir yapay sinir ağına gönderilip kontrol ediliyor.
3. Yapay sinir ağı gelen koordinatları öncelikle 28*28 piksel boyutuna uyacak şekilde dönüştürüyor. Bu dönüşümü yapmamamızın sebebi yapay sinir ağımızın bu boyutta resimleri girdi olarak almasıdır.
4. Dönüşüm yapıldıktan sonra resim Numpy dizisi olarak tekrardan oluşturuluyor.
5. Oluşturulan bu dizi kaydedilen ağırlıklar üzerinden kontrol ediliyor ve bir cevap oluşturuluyor.
6. Oluşturulan bu cevap geri döndürülüyor ve ekrana yazılıyor.

Dosya açıklamaları
- Control.html : Web sayfasından aldığımız verileri kontrol için kullandığımız kodlar ve 
açıklamaları.

- Control.ipynb : Web sayfasından aldığımız verileri kontrol için kullandığımız kodlar 
Jupyter notebook dosyası.

- Control.py : Web sayfasından aldığımız verileri kontrol için kullandığımız kodlar 
Python dosyası.

- doodle_recognizer.html : Yapay sinir ağımızı oluşturduğumuz ve eğittiğimiz kodlar ve 
açıklamaları.

- doodle_recognizer.ipynb : Yapay sinir ağımızı oluşturduğumuz ve eğittiğimiz kodlar 
Jupyter notebook dosyası.

- doodle_recognizer.py : Yapay sinir ağımızı oluşturduğumuz ve eğittiğimiz kodlar 
Python dosyası.

- my_model.h5 : Yapay sinir ağının eğitimden sonraki ağırlıklarının kaydedildiği dosya.

- Flask : Ana proje klasörümüz. Python Flask çerçevesi ile hazırladığımız 
web sayfası ve kodlarının bulunduğu klasör.

Copyright 2017 Ebru ÇAKAR

Bu yazılım MIT lisansına tabidir.

This software is licenced with MIT licence
