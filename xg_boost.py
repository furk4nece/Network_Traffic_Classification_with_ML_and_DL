import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek

print("1. Veri yükleniyor...")

train_df = pd.read_csv('UNSW_NB15_training-set.csv')
test_df  = pd.read_csv('UNSW_NB15_testing-set.csv')

target = 'attack_cat'

def muhendislik_ozellikleri_ekle(df):
    epsilon = 1e-8
    df['sbytes_per_spkt'] = df['sbytes'] / (df['spkts'] + epsilon)
    df['dbytes_per_dpkt'] = df['dbytes'] / (df['dpkts'] + epsilon)
    df['sloss_rate'] = df['sloss'] / (df['spkts'] + epsilon)
    df['dloss_rate'] = df['dloss'] / (df['dpkts'] + epsilon)
    df['total_bytes'] = df['sbytes'] + df['dbytes']
    df['total_pkts'] = df['spkts'] + df['dpkts']
    df['bytes_per_dur'] = df['total_bytes'] / (df['dur'] + epsilon)
    df['pkts_per_dur'] = df['total_pkts'] / (df['dur'] + epsilon)
    return df

print("2. Özellik mühendisliği uygulanıyor...")
train_df = muhendislik_ozellikleri_ekle(train_df)
test_df = muhendislik_ozellikleri_ekle(test_df)

X_train = train_df.drop(columns=[target, 'id'], errors='ignore')
y_train = train_df[target]

X_test = test_df.drop(columns=[target, 'id'], errors='ignore')
y_test = test_df[target]

print("3. Kategorik veriler dönüştürülüyor...")

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

print("4. Özellik seçimi yapılıyor...")

mi = mutual_info_classif(X_train, y_train)

mi_series = pd.Series(mi, index=X_train.columns)
mi_series = mi_series.sort_values(ascending=False)

top_features = mi_series.head(50).index

X_train = X_train[top_features]
X_test  = X_test[top_features]

print("Seçilen özellik sayısı:", len(top_features))

print("5. Veriler ölçeklendiriliyor...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ----- Seçim Parametresi -----
# Sentetik veri üretmek İSTEMİYORSANIZ "Yok" yazın.
secilen_yontem = "Yok" 
print(f"Kullanılan Yöntem: {secilen_yontem}")
# -----------------------------

if secilen_yontem == "Yok":
    print("6. Sentetik veri üretilmiyor, orijinal veri seti kullanılacak...")
    X_train_resampled = X_train
    y_train_resampled = y_train
else:
    print("6. Sentetik veri üretim algoritması uygulanıyor...")
    
    siniflar, sayilar = np.unique(y_train, return_counts=True)
    baslangic_dagilimi = dict(zip(siniflar, sayilar))
    
    orneklem_sozlugu = {}
    for sinif, sayi in zip(siniflar, sayilar):
        if sayi < 3000:
            orneklem_sozlugu[sinif] = int(sayi * 1.8)
        else:
            orneklem_sozlugu[sinif] = sayi
    
    temel_smote = SMOTE(sampling_strategy=orneklem_sozlugu, random_state=42)
    
    yeniden_ornekleme_yontemleri = {
        "SMOTE + ENN": SMOTEENN(smote=temel_smote, random_state=42),
        "SMOTE + Tomek": SMOTETomek(smote=temel_smote, random_state=42),
        "Borderline-SMOTE": BorderlineSMOTE(sampling_strategy=orneklem_sozlugu, random_state=42),
        "ADASYN": ADASYN(sampling_strategy=orneklem_sozlugu, random_state=42, n_neighbors=15),
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

print("7. Model kuruluyor ve eğitim başlıyor...")

model = XGBClassifier(
    random_state=42
)

model.fit(
    X_train_resampled,
    y_train_resampled,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("8. Tahmin yapılıyor...")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nGenel Doğruluk Oranı: {acc:.4f}")

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=le.classes_))