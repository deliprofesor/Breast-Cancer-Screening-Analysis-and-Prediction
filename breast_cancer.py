import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_ind
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import streamlit as st
import joblib

# Veri setini yükleme
data = pd.read_csv('CSAW-CC_breast_cancer_screening_data.csv')

# Veri yapısını inceleme
print(data.info())
print(data.describe())
print(data.head())

# Kategorik verilerde eksik değerleri en yaygın kategori ile doldurma
data['x_cancer_laterality'].fillna(data['x_cancer_laterality'].mode()[0], inplace=True)
data['x_type'].fillna(data['x_type'].mode()[0], inplace=True)

# Sayısal verileri standartlaştırma
scaler = StandardScaler()
data[['x_age', 'rad_r1', 'libra_breastarea', 'libra_densearea']] = scaler.fit_transform(data[['x_age', 'rad_r1', 'libra_breastarea', 'libra_densearea']])

# Eksik değerlerin analizi
missing_values = data.isnull().sum()
print("Eksik Değerler:")
print(missing_values)

# Eksik veri yüzdeleri
total = len(data)
missing_percentage = (missing_values / total) * 100
print("Eksik veri yüzdeleri:")
print(missing_percentage)

# Yaş dağılımını görselleştirme
plt.figure(figsize=(10, 6))
sns.histplot(data['x_age'].dropna(), bins=20, kde=True, color='blue')
plt.title('Yaş Dağılımı')
plt.xlabel('Yaş')
plt.ylabel('Frekans')
plt.show()

# Kanser türü ve kanser lateralliği arasındaki ilişkiyi görselleştirme
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='x_type', hue='x_cancer_laterality', palette='viridis')
plt.title('Kanser Türü ve Laterallik')
plt.xlabel('Kanser Türü')
plt.ylabel('Frekans')
plt.legend(title='Laterallik')
plt.show()

# Muayene yılına göre kanser teşhis oranları
plt.figure(figsize=(10, 6))
sns.histplot(data['exam_year'].dropna(), bins=15, kde=True, color='green')
plt.title('Muayene Yıllarına Göre Dağılım')
plt.xlabel('Yıl')
plt.ylabel('Frekans')
plt.show()

# Kanser türlerine göre yaş dağılımı
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='x_type', y='x_age', palette='coolwarm')
plt.title('Kanser Türlerine Göre Yaş Dağılımı')
plt.xlabel('Kanser Türü')
plt.ylabel('Yaş')
plt.show()

# Lymph node metastasis'e göre yaş dağılımı
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='x_lymphnode_met', y='x_age', palette='muted')
plt.title('Lymph Node Metastasis ve Yaş Dağılımı')
plt.xlabel('Lymph Node Metastasis')
plt.ylabel('Yaş')
plt.show()

# Radyolojik skor dağılımını görselleştirme
plt.figure(figsize=(10, 6))
sns.histplot(data['rad_r1'].dropna(), bins=20, kde=True, color='purple')
plt.title('Radyolojik Skor Dağılımı')
plt.xlabel('Radyolojik Skor (rad_r1)')
plt.ylabel('Frekans')
plt.show()

# Libra Breast Area ve Dense Area arasında ilişkiyi görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='libra_breastarea', y='libra_densearea', hue='x_type', palette='coolwarm')
plt.title('Libra Breast Area ve Dense Area Arasındaki İlişki')
plt.xlabel('Libra Breast Area')
plt.ylabel('Libra Dense Area')
plt.show()

# Libra Percent Density ile Kanser Türü arasındaki ilişkiyi görselleştirme
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='x_type', y='libra_percentdensity', palette='muted')
plt.title('Kanser Türlerine Göre Libra Percent Density')
plt.xlabel('Kanser Türü')
plt.ylabel('Libra Percent Density')
plt.show()

# Eksik veriler için bir strateji uygulama
data['x_age'].fillna(data['x_age'].median(), inplace=True)
data['x_lymphnode_met'].fillna(0, inplace=True)

# Temizlenmiş veri setini kaydetme
data.to_csv('csaw_cc_cleaned.csv', index=False)
print("Temizlenmiş veri seti kaydedildi.")

# Sayısal sütunlardaki eksik verileri medyan ile doldurmak
numeric_data = data.select_dtypes(include=['float64', 'int64'])
numeric_data_filled = numeric_data.fillna(numeric_data.median())

# Korelasyon matrisini hesaplama
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data_filled.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Özellikler Arası Korelasyonlar')
plt.show()

sns.pairplot(data, hue='x_type', vars=['x_age', 'rad_r1', 'libra_breastarea', 'libra_densearea'])
plt.show()

# Kanser türü gruplarına göre t-test'i
cancer_type_1 = data[data['x_type'] == 1]['x_age'].dropna()
cancer_type_2 = data[data['x_type'] == 2]['x_age'].dropna()

# T-test'i hesaplama
t_stat, p_val = ttest_ind(cancer_type_1, cancer_type_2, nan_policy='omit')
print(f"T-istatistiği: {t_stat}, P-değeri: {p_val}")

# Veri hazırlığı ve modelleme
data_cleaned = data.dropna(subset=['x_type'])  # 'x_type' kolonu için eksik verileri kaldırma
X = data_cleaned[['x_age', 'rad_r1', 'libra_breastarea', 'libra_densearea']]
y = data_cleaned['x_type']

# Eksik verileri ortalama ile doldurmak
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# SMOTE uygulama
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Random Forest modelini oluşturma
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
print(classification_report(y_test, y_pred))

# Grid Search ile hiperparametre optimizasyonu
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3)
grid_search.fit(X_train_resampled, y_train_resampled)
print(f"En iyi parametreler: {grid_search.best_params_}")

# En iyi parametrelerle model oluşturma
best_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced'  # Azınlık sınıflarına daha fazla ağırlık ver
)

# SMOTE ile dengelenmiş verilerle model eğitimi
best_rf.fit(X_train_resampled, y_train_resampled)

# Test verisi ile tahmin yapma
y_pred = best_rf.predict(X_test)

# Performans raporunu yazdırma
print(classification_report(y_test, y_pred))

# Özelliklerin önemini görselleştirme
feature_importance = model.feature_importances_
feature_names = X.columns

plt.barh(feature_names, feature_importance)
plt.title('Özelliklerin Önemi')
plt.show()

# Kanser türlerinin yıllara göre dağılımını görselleştirme
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='exam_year', hue='x_type')
plt.title('Kanser Türlerinin Yıllara Göre Dağılımı')
plt.xlabel('Yıl')
plt.ylabel('Frekans')
plt.show()

# KMeans Kümeleme
data_clean = data.dropna(subset=['x_age', 'rad_r1', 'libra_breastarea', 'libra_densearea'])
X_clean = data_clean[['x_age', 'rad_r1', 'libra_breastarea', 'libra_densearea']]
kmeans = KMeans(n_clusters=3, random_state=42)
data_clean['cluster'] = kmeans.fit_predict(X_clean)
data['cluster'] = None  # Initialize the 'cluster' column with NaN values
data.loc[data_clean.index, 'cluster'] = kmeans.fit_predict(X_clean)

# Kümeleme sonuçlarını görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='libra_breastarea', y='libra_densearea', hue='cluster', palette='viridis')
plt.title('K-Means Kümeleme Sonuçları')
plt.xlabel('Libra Breast Area')
plt.ylabel('Libra Dense Area')
plt.show()

# Libra Percent Density ile Kanser Türü arasındaki ilişkiyi görselleştirme
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='x_type', y='libra_percentdensity', palette='muted')
plt.title('Kanser Türlerine Göre Libra Percent Density')
plt.xlabel('Kanser Türü')
plt.ylabel('Libra Percent Density')
st.pyplot(plt)  # Streamlit ile görselleştirmeyi görüntüleme

# Modeli kaydetme
joblib.dump(model, 'breast_cancer_model.pkl')

# Streamlit'te modelin kaydedildiğini gösterme
st.write("Model kaydedildi: breast_cancer_model.pkl")
