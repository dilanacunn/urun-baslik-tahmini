## Nasıl Çalıştırılır?

1. Google Colab veya Jupyter Notebook ortamı açın.
2. `urun_baslik_tahmin.ipynb` dosyasını yükleyin.  
3. Veri setini `data/` klasörüne ekleyin.  
4. Hücreleri sırayla çalıştırarak çıktıları gözlemleyin.

Not: Veri setini HuggingFace üzerinden veri setini indirin.
Kaynak: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/raw_meta_All_Beauty/full-00000-of-00001.parquet

# Ürün Başlığından Açıklama ve Kategori Tahmini (GenAI + ML Projesi)

Bu projede e-ticaret verileri kullanılarak iki temel görev gerçekleştirildi.

1. Başlıktan açıklama üretimi (Text2Text Generation)
2. Başlıktan kategori tahmini (Gözetimli Öğrenme)

Proje, Google Colab üzerinde hazırlanmış olup GitHub'a .ipynb ve görsel çıktılarla birlikte yüklendi.

---

## 1. Veri Seti

- Kullanılan veri: Amazon ürün verileri (örnek olarak 3000 satır alındı.)
- Kullanılan sütunlar: product_title, product_description, category
- Dosya formatı: `.parquet`

---

## 2. Keşifsel Veri Analizi (EDA)

- value_counts() ile kategori dağılımı incelendi.
- Ürün başlıklarının ve açıklamaların uzunluklarına bakıldı.
- Aşırı kısa/uzun başlıklar ve boş açıklamalar filtrelendi.
- Matplotlib kullanılarak temel grafikler oluşturuldu.

Kategori Dağılımı:

![image](kategori-dagilimi.png)

---

## 3. Başlıktan Açıklama Üretimi

- Projede HuggingFace üzerindeki `t5-small` modeli ile ürün başlıklarından açıklama üretildi. 
- Modelin eğitim formatı gereği `"generate description: "` komutu başlıklara eklendi.

### Kullanılan yöntem:
- `pipeline("text2text-generation")`
- Rastgele seçilen 5 başlık için açıklama üretimi.
- Üretilen açıklamalar tablo halinde sunuldu.

### Örnek Üretim:
![T5 Açıklama Örnekleri 2](![image](t5-aciklama-uretimi2.png)

Modelin çıktıları oldukça tutarlı ve başlıkla uyumlu metinler üretmiştir.

---

## 4. Başlıktan Kategori Tahmini (Gözetimli Öğrenme)

- Başlık verileri `TfidfVectorizer` ile vektörleştirildi.
- İki farklı model eğitildi:
  - `LogisticRegression`
  - `RandomForestClassifier`
- `train_test_split` ile veri %80 eğitim / %20 test olarak bölündü
- `class_weight='balanced'` ve `SMOTE` ile dengesiz veri problemi ele alındı.

---

### Performans Ölçümleri:
- Accuracy: **%84**
- F1 Score: **0.82**
- `classification_report` ile detaylı sonuçlar analiz edildi.

---

## 5. Kümeleme

### Yöntem 1: TF-IDF + KMeans

- Başlıklar TF-IDF vektörlerine dönüştürüldü.
- `KMeans(n_clusters=5)` ile kümeleme yapıldı.
- Küme örnekleri tablo halinde gösterildi.

### Yöntem 2: Sentence-Transformers + KMeans

- `all-MiniLM-L6-v2` modeli ile başlıklar vektörleştirildi (embedding boyutu = 384).
- Aynı şekilde KMeans ile kümeler oluşturuldu.

Anlamsal Küme Dağılımı:

![image](anlamsal-kume-dagilimi.png)

---

## Öğrendiklerim

- HuggingFace ile metinden metin üretimi
- TF-IDF ve embedding yöntemleriyle metin vektörleştirme
- Lojistik regresyon ve rastgele orman ile sınıflandırma
- SMOTE ile dengesiz veri setiyle başa çıkma
- KMeans ile kümeleme ve anlamlılık kontrolü

---

## Notlar

- Bu proje, doğal dil işleme (NLP), metin sınıflandırma ve gözetimli öğrenme gibi konuların temelini uygulamalı olarak öğrenmek için hazırlandı.
- Başlangıç seviyesindeki veri bilimi projeleri için örnek teşkil eder.
- Kodlar, açıklamalar ve grafikler dosyada yer almakta. 
- Gelişime açıktır, katkılarınızı beklerim.
