import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek

print("1. Veri yükleniyor...")

train_df = pd.read_csv('UNSW_NB15_training-set.csv')
test_df  = pd.read_csv('UNSW_NB15_testing-set.csv')

target = 'attack_cat'

X_train = train_df.drop(columns=[target, 'id'], errors='ignore')
y_train = train_df[target]

X_test = test_df.drop(columns=[target, 'id'], errors='ignore')
y_test = test_df[target]

print("2. Kategorik veriler dönüştürülüyor...")

cat_cols = ['proto', 'service', 'state']

combined = pd.concat([X_train, X_test], axis=0)
combined = pd.get_dummies(combined, columns=cat_cols)

X_train = combined.iloc[:len(X_train), :]
X_test  = combined.iloc[len(X_train):, :]

X_train = X_train.fillna(0)
X_test  = X_test.fillna(0)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

print("3. Veriler ölçeklendiriliyor (Sentetik algoritmaların matematiği için zorunlu)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("4. Sentetik veri üretim algoritması uygulanıyor...")

# Sentetik veri üretmek istemiyorsan buraya "Yok" yazabilirsin
secilen_yontem = "SMOTE + ENN" 
print(f"Kullanılan Yöntem: {secilen_yontem}")

if secilen_yontem == "Yok":
    print("Sentetik veri üretilmiyor, orijinal veri seti kullanılacak...")
    X_train_resampled = X_train
    y_train_resampled = y_train
else:
    siniflar, sayilar = np.unique(y_train, return_counts=True)
    baslangic_dagilimi = dict(zip(siniflar, sayilar))
    
    orneklem_sozlugu = {}
    for sinif, sayi in zip(siniflar, sayilar):
        if sayi < 3000:
            orneklem_sozlugu[sinif] = int(sayi * 1.5) # 3 katı
        else:
            orneklem_sozlugu[sinif] = sayi
            
    # --- DİNAMİK KOMŞU HESAPLAMA SİSTEMİ ---
    # Sadece verisi artırılacak sınıfların mevcut boyutlarını buluyoruz
    artirilacak_sinif_sayilari = [sayi for sinif, sayi in zip(siniflar, sayilar) if orneklem_sozlugu[sinif] > sayi]
    
    if artirilacak_sinif_sayilari:
        en_kucuk_sinif_boyutu = min(artirilacak_sinif_sayilari)
        # Komşu sayısı mevcut eleman sayısından 1 eksik olmalıdır. En az 1 olmasını garanti ediyoruz.
        komsu_sayisi = max(1, en_kucuk_sinif_boyutu - 1)
        smote_komsu = min(5, komsu_sayisi)   # SMOTE genelde en fazla 5 komşuya bakar
        adasyn_komsu = min(15, komsu_sayisi) # ADASYN'i 15 komşuya kadar serbest bırakıyoruz
    else:
        smote_komsu = 5
        adasyn_komsu = 15
    # ----------------------------------------
    
    temel_smote = SMOTE(sampling_strategy=orneklem_sozlugu, random_state=42, k_neighbors=smote_komsu)
    
    yeniden_ornekleme_yontemleri = {
        "SMOTE + ENN": SMOTEENN(smote=temel_smote, random_state=42),
        "SMOTE + Tomek": SMOTETomek(smote=temel_smote, random_state=42),
        "Borderline-SMOTE": BorderlineSMOTE(sampling_strategy=orneklem_sozlugu, random_state=42, k_neighbors=smote_komsu),
        "ADASYN": ADASYN(sampling_strategy=orneklem_sozlugu, random_state=42, n_neighbors=adasyn_komsu),
        "Rastgele Asiri Ornekleme": RandomOverSampler(sampling_strategy=orneklem_sozlugu, random_state=42),
        "SMOTE": temel_smote
    }
    
    uygulanacak_algoritma = yeniden_ornekleme_yontemleri[secilen_yontem]
    X_train_resampled, y_train_resampled = uygulanacak_algoritma.fit_resample(X_train, y_train)
    
    yeni_siniflar, yeni_sayilar = np.unique(y_train_resampled, return_counts=True)
    bitis_dagilimi = dict(zip(yeni_siniflar, yeni_sayilar))
    
    print("\nSınıf Dağılımındaki Değişim ve Üretilen Veri Sayısı:")
    print(f"{'Sınıf Adı':<16} | {'Başlangıç':<10} | {'Bitiş':<10} | {'Üretilen/Silinen':<15}")
    print("-" * 60)
    for sinif_kodu in range(len(le.classes_)):
        sinif_adi = le.inverse_transform([sinif_kodu])[0]
        baslangic = baslangic_dagilimi.get(sinif_kodu, 0)
        bitis = bitis_dagilimi.get(sinif_kodu, 0)
        degisim = bitis - baslangic
        isaret = "+" if degisim > 0 else ""
        print(f"{sinif_adi:<16} | {baslangic:<10} | {bitis:<10} | {isaret}{degisim:<15}")
    print("-" * 60)
    print(f"Eski Toplam Boyut: {sum(baslangic_dagilimi.values())}")
    print(f"Yeni Toplam Boyut: {X_train_resampled.shape[0]}\n")

print("5. Model kuruluyor ve eğitim başlıyor...")

model = XGBClassifier(
    random_state=42
)

model.fit(
    X_train_resampled,
    y_train_resampled,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("6. Tahmin yapılıyor...")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)


print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"\nGenel Doğruluk Oranı: {acc:.4f}")