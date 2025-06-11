# Proje Adı: Ardışık Vuruşlarla Difüzyon Modelleri ile Eskiz Üretimi

Bu proje, ardışık vuruşlar (stroke-by-stroke) kullanarak el çizimi eskizler üreten koşullu bir difüzyon modeli (conditional diffusion model) uygulamasıdır. Model, bir çizimi tek seferde oluşturmak yerine, insanın doğal çizim sürecini taklit ederek vuruşları gerçekçi ve yorumlanabilir bir şekilde art arda üretir. Modeller, Google'ın [Quick, Draw!](https://quickdraw.withgoogle.com/data/) veri setindeki `kedi`, `otobüs` ve `tavşan` kategorileri kullanılarak eğitilmiştir.




## 🎯 Genel Bakış

Projenin temel amacı, `kedi`, `otobüs` ve `tavşan` olmak üzere üç farklı kategori için ayrı ayrı koşullu difüzyon modelleri tasarlamak ve eğitmektir. Her model, ilgili kategoriye ait çizimleri, vuruşların vektör verilerini sıralı bir şekilde üreterek oluşturmayı öğrenir. Tüm süreç, veri hazırlığından model değerlendirmesine kadar tekrarlanabilir bir Jupyter Notebook'ta belgelenmiştir.

---

## 📁 Dosya Yapısı

Proje dizini aşağıdaki gibi organize edilmiştir:

TECHNICAL_ASSIGNMENT/
├── Technical_Assignment.ipynb  # Ana Notebook: Kod, açıklamalar ve sonuçlar
├── run_all.py                  # Tüm işlem adımlarını çalıştıran betik
├── train_diffusion.py          # Difüzyon modelini eğitmek için betik
├── sample_diffusion.py         # Eğitilmiş modelden örnek üretmek için betik
├── diffusion_model.py          # Difüzyon modelinin mimarisini içerir
├── dataset_utils.py            # Veri yükleme ve işleme yardımcı fonksiyonları
├── metrics.py                  # FID ve KID metriklerinin implementasyonu
├── data/                         # (Oluşturulacak) .ndjson veri setlerinin saklanacağı dizin
├── subset/                       # Her kategori için önceden tanımlanmış train/test ayrımları
│   ├── cat/
│   │   └── indices.json
│   ├── bus/
│   │   └── indices.json
│   └── rabbit/
│       └── indices.json
├── https://www.google.com/search?q=cat_fake_sample_animation.gif   # 'Kedi' kategorisi için üretilen animasyon
├── https://www.google.com/search?q=bus_fake_sample_animation.gif   # 'Otobüs' kategorisi için üretilen animasyon
└── https://www.google.com/search?q=rabbit_fake_sample_animation.gif # 'Tavşan' kategorisi için üretilen animasyon

