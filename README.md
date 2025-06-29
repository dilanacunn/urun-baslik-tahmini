## Nasıl Çalıştırılır?

1. Google Colab veya Jupyter Notebook ortamı açın  
2. `urun_baslik_tahmin.ipynb` dosyasını yükleyin  
3. Veri setini `data/` klasörüne ekleyin  
4. Hücreleri sırayla çalıştırarak çıktıları gözlemleyin

Not: Veri setini yükleyemiyorsanız veya hata alıyorsanız HuggingFace üzerinden veri setini indirin.
Kaynak: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/raw_meta_All_Beauty/full-00000-of-00001.parquet
Format: Parquet

# Ürün Başlığından Açıklama ve Kategori Tahmini

Bu projede e-ticaret verileri kullanılarak iki temel görev gerçekleştirilmiştir:

1. Başlıktan açıklama üretimi (Text2Text Generation)
2. Başlıktan kategori tahmini (Gözetimli Öğrenme)

Proje, Google Colab üzerinde hazırlanmış olup GitHub'a .ipynb ve görsel çıktılarla birlikte yüklenmiştir.

---

## 1. Veri Seti

- Kullanılan veri: Amazon ürün verileri (örnek olarak 3000 satır alınmıştır)
- Kullanılan sütunlar: product_title, product_description, category
- Dosya formatı: `.parquet`

---

## 2. Keşifsel Veri Analizi (EDA)

- value_counts() ile kategori dağılımı incelenmiştir.
- Ürün başlıklarının ve açıklamaların uzunluklarına bakılmıştır.
- Matplotlib kullanılarak temel grafikler oluşturulmuştur.

Kategori Dağılımı:
![image](https://github.com/user-attachments/assets/1f113d19-4a83-4049-848c-15214bfb3c2b)


---

## 3. Başlıktan Açıklama Üretimi

- HuggingFace üzerinden t5-small modeli kullanılmıştır.
- pipeline("text2text-generation") fonksiyonu ile açıklama tahmini yapılmıştır.
- Yaklaşık 10 başlık için açıklama üretilmiş ve örnek çıktı gözlemlenmiştir.

---

## 4. Başlıktan Kategori Tahmini (Gözetimli Öğrenme)

- TfidfVectorizer ile başlık verisi sayısallaştırılmıştır.
- İki farklı model denenmiştir:
  - Logistic Regression
  - Random Forest
- train_test_split ile veri %80 eğitim, %20 test olarak ayrılmıştır.
- Model performansı accuracy_score, f1_score ve classification_report ile değerlendirilmiştir.

Örnek çıktı:

Accuracy: 84%
F1 Score: 0.82


---

## 5. Kümeleme

### TF-IDF + KMeans

- TF-IDF vektörleri ile KMeans algoritması uygulanmıştır.
- Kümelerden örnek başlıklar alınarak yorum yapılmıştır.

### Sentence-Transformers + KMeans

- Gelişmiş vektörleme için all-MiniLM-L6-v2 modeli ile embedding yapılmıştır.
- Her başlık 384 boyutlu vektöre dönüştürülmüş ve yeniden KMeans ile kümeleme uygulanmıştır.
- Küme dağılımı dengeli ve tematik olarak anlamlı bulunmuştur.

Anlamsal Küme Dağılımı:
![image](https://github.com/user-attachments/assets/178d81ef-f418-4482-8252-a00942063d5a)

---

## Dosya Yapısı

urun-baslik-tahmini/
├── notebook/
│   └── urun_baslik_tahmin_projesi.ipynb   
├── data/
│   └── full-00000-of-00001.parquet        
├── ekran-goruntuleri/
│   ├── kategori-dagilimi.png             
│   └── anlamsal-kume.png               
├── README.md 


---

## Notlar

- Bu proje, doğal dil işleme (NLP), metin sınıflandırma ve gözetimli öğrenme gibi konuların temelini uygulamalı olarak öğrenmek için hazırlanmıştır.
- Başlangıç seviyesindeki veri bilimi projeleri için örnek teşkil eder.
- Kodlar, açıklamalar ve grafikler dosyada yer almaktadır.  
- Gelişime açıktır, katkılarınızı beklerim.
