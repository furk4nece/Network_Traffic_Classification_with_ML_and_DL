# Makine Öğrenmesi ve Derin Öğrenme Yöntemleriyle Ağ Trafiği Sınıflandırılması

Bu proje kapsamında derin öğrenme ve makine öğrenmesi algoritmaları kullanılarak ağ trafiği sınıflandırması performansının iyileştirilmesi hedeflenmiştir. Gerçek dünya ağ akışlarını yansıtan veri setlerinde sıklıkla karşılaşılan sınıf dengesizliği problemi sentetik veri üretme algoritmaları ile giderilerek modellerin başarılarına olan etkileri deneysel olarak incelenmiştir. 

Ağ trafiği sınıflandırma süreçlerinde sıklıkla karşılaşılan çoğunluk sınıfını ezberleme eğilimi kırılmış ve nadir anomali durumlarına karşı algoritmaların tespit yeteneği güçlendirilmiştir.

## Proje Ekibi
* **Geliştiriciler:** Furkan ECE, Emir KÖSE 
* **Danışman:** Dr. Öğr. Üyesi Durmuş Özkan Şahin
* **Kurum:** Ondokuz Mayıs Üniversitesi Mühendislik Fakültesi Bilgisayar Mühendisliği Bölümü 

## Kullanılan Teknolojiler ve Yöntemler
* **Veri Seti:** UNSW-NB15
* **Sınıflandırma Algoritmaları:** Çok Katmanlı Algılayıcı, Rastgele Orman, CatBoost, XGBoost
* **Yeniden Örnekleme Algoritmaları:** SMOTE, Borderline-SMOTE, ADASYN, Rastgele Aşırı Örnekleme, SMOTEENN, SMOTETomek

## Dosya Yapısı ve İçerik

Proje dizininde her bir sınıflandırma algoritması için ayrı betikler oluşturulmuştur. Tüm betiklerde sentetik veri üretimi için dinamik komşu hesaplama sistemi kullanılarak sınıfların orijinal yapısı korunmuştur.

* `xg_boost.py`: Özellik mühendisliği uygulanmadan, ham veriler üzerinden XGBoost algoritmasının farklı veri artırma teknikleriyle test edildiği kod dosyası.
* `cat_boost.py`: Kategorik özniteliklere karşı dirençli olan CatBoost algoritması ile model eğitimi gerçekleştiren betik.
* `mlp.py`: Çok Katmanlı Algılayıcı mimarisi kullanılarak ağ trafiği sınıflandırması yapan kod dosyası.
* `random_forest.py`: Karar ağaçları tabanlı Rastgele Orman algoritması ile sentetik veri üretim yöntemlerinin performansını ölçen dosya.

## Deneysel Sonuçlar

### Veri Artırma Tekniklerinin Çok Katmanlı Algılayıcı (MLP) Algoritmasına Etkileri

| Algoritma | Genel Doğruluk | Makro F1 Skoru |
| :--- | :--- | :--- |
| Veri Artırma Olmadan | 0.8342 | 0.54 |
| ADASYN | 0.8305 | 0.54 |
| SMOTE | 0.8348 | 0.54 |
| Borderline-SMOTE | 0.8298 | 0.53 |
| Rastgele Aşırı Örnekleme | 0.8354 | 0.53 |
| SMOTEENN | 0.8231 | 0.48 |
| SMOTETomek | 0.8383 | 0.53 |

### Veri Artırma Tekniklerinin Rastgele Orman (RF) Algoritmasına Etkileri

| Algoritma | Genel Doğruluk | Makro F1 Skoru |
| :--- | :--- | :--- |
| Veri Artırma Olmadan | 0.8437 | 0.56 |
| ADASYN | 0.8435 | 0.58 |
| SMOTE | 0.8407 | 0.57 |
| Borderline-SMOTE | 0.8392 | 0.57 |
| Rastgele Aşırı Örnekleme | 0.8449 | 0.56 |
| SMOTEENN | 0.8398 | 0.52 |
| SMOTETomek | 0.8383 | 0.57 |

### Veri Artırma Tekniklerinin CatBoost Algoritmasına Etkileri

| Algoritma | Genel Doğruluk | Makro F1 Skoru |
| :--- | :--- | :--- |
| Veri Artırma Olmadan | 0.8462 | 0.56 |
| ADASYN | 0.8465 | 0.57 |
| SMOTE | 0.8473 | 0.56 |
| Borderline-SMOTE | 0.8475 | 0.56 |
| Rastgele Aşırı Örnekleme | 0.8483 | 0.57 |
| SMOTEENN | 0.8269 | 0.50 |
| SMOTETomek | 0.8471 | 0.56 |

### Veri Artırma Tekniklerinin XGBoost Algoritmasına Etkileri

| Algoritma | Genel Doğruluk | Makro F1 Skoru |
| :--- | :--- | :--- |
| Veri Artırma Olmadan | 0.8401 | 0.53 |
| ADASYN | 0.8525 | 0.61 |
| SMOTE | 0.8541 | 0.62 |
| Borderline-SMOTE | 0.8544 | 0.62 |
| Rastgele Aşırı Örnekleme | 0.8532 | 0.62 |
| SMOTEENN | 0.8535 | 0.61 |
| SMOTETomek | 0.8536 | 0.62 |

Veri dengeleme işlemleri algoritmaların çoğunluk sınıfını ezberleme eğilimini kırmış ve nadir anomali durumlarına karşı tespit yeteneğini güçlendirmiştir.
Sınıflandırma performanslarında en keskin sıçrama XGBoost algoritması üzerinde kaydedilmiş ve SMOTE yöntemiyle makro F1 skoru 0.62 seviyesine ulaşmıştır.
CatBoost ve Rastgele Orman algoritmaları yeniden örnekleme işlemlerine karşı yüksek yapısal kararlılıklarını korumuştur.


## Kurulum ve Çalıştırma

Projenin yerel bilgisayarınızda hatasız çalışabilmesi için gerekli kütüphaneleri aşağıdaki komut ile kurabilirsiniz.

```bash
pip install pandas numpy scikit-learn xgboost catboost imbalanced-learn

