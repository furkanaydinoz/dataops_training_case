# Model Mimarileri - Hap Bilgiler

> MNIST deneyinde kullanılan 10 farklı modelin kısa tarihçesi, bileşenleri ve geliştirme amaçları.

---

## 1. MLP (Multi-Layer Perceptron) Ailesi

### SimpleMLP
| Özellik | Değer |
|---------|-------|
| Katmanlar | 784→128→64→10 |
| Parametre | ~100K |
| Aktivasyon | ReLU |

**Hap Bilgisi:**
- 1958: Rosenblatt perceptron'dan, 1986: Backpropagation ile modern MLP doğdu
- En temel neural network yapısı
- Feature engineering gerektirir (MNIST için flatten)

**Neden var?** Baseline için. Kompleks mimariye gerek olup olmadığını test etmek için.

---

### WideMLP
| Özellik | Değer |
|---------|-------|
| Katmanlar | 784→512→256→10 |
| Parametre | ~530K |
| Aktivasyon | ReLU |

**Hap Bilgisi:**
- Width (genişlik) artırılarak kapasite artırma stratejisi
- Daha az derinlik = daha az vanishing gradient riski
- Universal Approximation Theorem: yeterli width ile her fonksiyonu approx edebilir

**Neden var?** "Deeper is better" yerine "wider is better" hipotezini test.

---

### DeepMLP
| Özellik | Değer |
|---------|-------|
| Katmanlar | 784→64 (×6 hidden) →10 |
| Parametre | ~57K |
| Aktivasyon | ReLU |

**Hap Bilgisi:**
- Depth (derinlik) artırma stratejisi
- Her katman soyutlamalı feature'lar oluşturur
- Derinlik arttıkça vanishing gradient problemi başlar

**Neden var?** "Narrow but deep" yaklaşımını test. Kompleks non-linearity.

---

## 2. CNN (Convolutional Neural Network) Ailesi

### BasicCNN
| Özellik | Değer |
|---------|-------|
| Katmanlar | Conv→Pool→Conv→Pool→FC→FC |
| Parametre | ~110K |
| İlk Kullanım | 1989 (LeCun) |

**Hap Bilgisi:**
- **Convolution**: 1960'larda visual cortex'ten ilham alındı, 1989'da LeCun MNIST için kullandı
- **Receptive field**: Her nöron sadece lokal alana bakıyor
- **Parameter sharing**: Aynı filtre tüm görselde paylaşılır
- **Translation equivariance**: Görsel kaysa bile öğrenilen pattern korunur

**Neden var?** Image processing için spatial invariance ve locality. MLP'den farkı: weight sharing ile parametre verimliliği.

---

### DepthwiseCNN
| Özellik | Değer |
|---------|-------|
| Katmanlar | Depthwise→Pointwise→Pool→... |
| Parametre | ~10K |
| İlk Kullanım | 2014 (Inception, Xception) |

**Hap Bilgisi:**
- **Depthwise Separable Convolution**: Spatial ve channel convolution'ları ayırır
- **Pointwise (1×1) Conv**: Channel'ları mix etmek için
- **Computational efficiency**: ~10× daha az işlem
- 2014: Inception (Google) → 2016: Xception (Chollet)

**Neden var?** Computational maliyeti düşürmek için. Aynı receptive field ama çok daha az parametre.

---

### EfficientCNN
| Özellik | Değer |
|---------|-------|
| Katmanlar | Conv→DW→PW→Pool→DW→PW→Pool→GAP→FC |
| Parametre | ~80K |
| İlk Kullanım | 2017 (MobileNet) |

**Hap Bilgisi:**
- **Compound scaling**: Depth, width, resolution dengeli artırma
- **MobileNet**: Sınırlı computational kaynak için tasarlandı
- Depthwise separable + pointwise kombinasyonu
- Depth multiplier ile accuracy-speed trade-off

**Neden var?** Mobile/device deployment için optimize. DepthwiseCNN + dengeli scaling.

---

## 3. ResNet (Residual Network)

### MiniResNet
| Özellik | Değer |
|---------|-------|
| Katmanlar | Conv→ResBlock×3→GAP→FC |
| Parametre | ~130K |
| İlk Kullanım | 2015 (He et al., Microsoft) |

**Hap Bilgisi:**
- **Skip Connection**: Input doğrudan output'a eklenir
- **2015**: ImageNet'de 152 katman ile ilk kez "deeper is better" mümkün oldu
- **Problem çözümü**: 30+ katman sonrası degradation (accuracy düşmesi)
- **Gradient flow**: Skip connection sayesinde gradient direkt akabilir
- **Identity mapping**: x → F(x) + x yerine x → y formülü

**Neden var?** "Deeper networks can't learn" problemi çözmek için. Skip connection ile gradient flow kolaylaştı.

**Anahtar Formül:**
```
output = F(x) + x  (skip connection)
```

---

## 4. ViT (Vision Transformer)

### MiniViT
| Özellik | Değer |
|---------|-------|
| Katmanlar | PatchEmb→PosEmb→TransformerBlock×2→Pool→FC |
| Parametre | ~40K |
| İlk Kullanım | 2020 (Dosovitskiy et al., Google) |

**Hap Bilgisi:**
- **2017**: Transformer NLP'de (Attention is All You Need)
- **2020**: ViT - Transformer'ı görsele uyguladı
- **Patch Embedding**: 7×7 patch'ler = 16 token (4×4 grid)
- **Self-Attention**: Her patch tüm diğer patch'lere bakabilir
- **Inductive bias**: CNN'den farklı olarak locality yok, sadece similarity

**Neden var?** CNN'in local receptive field sınırlamasını aşmak için. Global attention ile uzun mesafe dependencies.

---

## 5. CapsuleNetwork

### CapsuleNetwork
| Özellik | Değer |
|---------|-------|
| Katmanlar | Conv→PrimaryCaps→DigitCaps→Length |
| Parametre | ~320K |
| İlk Kullanım | 2017 (Sabour, Hinton, Nvidia) |

**Hap Bilgisi:**
- **2011**: Hinton "What is wrong with convolutional neural networks?" konuşması
- **2017**: Dynamic Routing Between Capsules makalesi
- **Problem**: MaxPool bilgi kaybı (spatial relationship'ler gidiyor)
- **Çözüm**: Vector-valued activations (scalars yerine)
- **Dynamic Routing**: Child capsule'lar parent'ları seçer (öğrenilen)
- **Equivariance**: Girdi döndüğünde capsule vektörü de döner

**Neden var?** MaxPool'un yarattığı spatial bilgi kaybını önlemek için. Part-whole relationship korunur.

---

## 6. GNN (Graph Neural Network)

### TinyGNN
| Özellik | Değer |
|---------|-------|
| Katmanlar | Patch→NodeEmb→MP×2→Pool→FC |
| Parametre | ~50K |
| İlk Kullanım | 2016 (Scarselli et al.) |

**Hap Bilgisi:**
- **2009**: Gori, Scarselli - ilk GNN paper
- **2016**: Li, Tadepalli - message passing framework
- **Graph Representation**: 9 node (3×3 grid), edges = adjacency
- **Message Passing**: Her node komşularından mesaj alır ve güncellenir
- **Node features**: Herbir patch ayrı node olarak temsil edilir
- **Aggregation**: Mean pooling ile node'lar birleştirilir

**Neden var?** Non-Euclidean data (graph, social network, molecule) için. Görseli graph olarak modellemek.

---

## 7. EfficientNetStyle

### EfficientNetStyle
| Özellik | Değer |
|---------|-------|
| Katmanlar | Stem→MBConv×5→GAP→FC |
| Parametre | ~100K |
| İlk Kullanım | 2019 (Tan et al., Google Brain) |

**Hap Bilgisi:**
- **2019**: EfficientNet - compound scaling (depth×1.4, width×1.4, resolution×1.4)
- **MBConv (Mobile Inverted Bottleneck)**: 2018 MobileNetV2
- **Expansion ratio**: 1× veya 6× channel genişlemesi
- **Swish activation**: x × sigmoid(x), ReLU'dan daha smooth
- **SE (Squeeze-Excitation)**: 2018 Hu et al. - channel attention
  - Squeeze: Global average pooling (HxW → 1×1)
  - Excitation: FC→ReLU→FC→Sigmoid (C → C/4 → C)

**Neden var?** Accuracy ve efficiency arasında optimal denge kurmak için. Compound scaling ile.

---

## Özet Zaman Çizelgesi

```
1958  ─── Perceptron (Rosenblatt)
1986  ─── Backpropagation (Rumelhart, Hinton, Williams)
1989  ─── CNN for MNIST (LeCun)
1998  ─── LeNet-5
2009  ─── GNN concept (Gori, Scarselli)
2011  ─── Capsule idea (Hinton)
2014  ─── Inception, VGGNet
2015  ─── ResNet (skip connections)
2016  ─── Message Passing GNN
2017  ─── Transformers (Attention is All You Need)
2017  ─── CapsuleNetwork (Sabour, Hinton)
2017  ─── MobileNetV1 (Depthwise Separable)
2018  ─── MobileNetV2 (MBConv, SE blocks)
2018  ─── Squeeze-Excitation (Hu et al.)
2019  ─── EfficientNet (compound scaling)
2020  ─── Vision Transformer (ViT)
```

---

## Component Karşılaştırma

| Component | Yıl | Kim | Ne İçin |
|-----------|-----|-----|---------|
| Perceptron | 1958 | Rosenblatt | Temel neuron modeli |
| Backprop | 1986 | Rumelhart et al. | Gradient hesaplama |
| Convolution | 1960s | Visual cortex | Spatial locality |
| ReLU | 2010 | Nair, Hinton | Dying gradient problemi |
| Dropout | 2014 | Srivastava | Overfitting önleme |
| BatchNorm | 2015 | Ioffe, Szegedy | Internal covariate shift |
| Skip Connection | 2015 | He et al. | Gradient flow |
| Depthwise Sep Conv | 2014 | Sifre, Mallat | Computation efficiency |
| Attention | 2017 | Vaswani et al. | Long-range dependencies |
| SE Block | 2018 | Hu et al. | Channel attention |
| MBConv | 2018 | Sandler et al. | Inverted residual block |

---

## Model Seçim Rehberi (Ne Zaman Hangisi?)

| Senaryo | Önerilen Model |
|--------|----------------|
| Basit baseline | SimpleMLP |
| Hızlı inference | DepthwiseCNN |
| Mobil/device | EfficientNetStyle, MobileNet |
| Çok derin ağ | ResNet (skip connections) |
| Global context | ViT (self-attention) |
| Spatial relationships | CapsuleNetwork |
| Graph data | GNN |
| Sınırlı data | WideMLP (daha az overfit riski) |
| Large dataset | ViT (transformer) |

---

## Model Accuracy Karşılaştırması (MNIST)

| Model | Accuracy | Parametre |
|-------|----------|-----------|
| SimpleMLP | ~97% | 100K |
| WideMLP | ~97.5% | 530K |
| DeepMLP | ~96.5% | 57K |
| BasicCNN | ~98% | 110K |
| DepthwiseCNN | ~97% | 10K |
| EfficientCNN | ~97.5% | 80K |
| MiniResNet | ~98.5% | 130K |
| MiniViT | ~97% | 40K |
| CapsuleNet | ~97.5% | 320K |
| TinyGNN | ~95% | 50K |

**Sonuç:** Tüm modeller %95+ accuracy → **Data > Architecture** hipotezi doğrulandı.

---

## 8. FANET (Feature Adaptive Network with Enhanced Transformers)

### FANET
| Özellik | Değer |
|---------|-------|
| Katmanlar | CNNStem→ConvAttn×3→GAP→FC |
| Parametre | ~80K |
| İlk Kullanım | 2026 (Bu Proje) |

**Hap Bilgisi:**
- **2026**: CNN + Transformer hybrid modeli
- **MultiScaleCNNStem**: ViT'in linear patch projection'ı yerine hierarchical CNN feature extraction
- **ConvolutionalMHSA**: QKV projection için linear yerine depthwise convolution kullanır
- **Local + Global Attention**: DWConv local receptive field + attention global context

**Neden var?** CNN'in lokal feature extraction'ı ile Transformer'ın global attention'ını birleştirmek için.

**ViT ile Farkı:**
| Aspect | ViT | FANET |
|--------|-----|-------|
| Input Processing | Linear patch projection | Hierarchical CNN |
| Attention | Pure global (MHSA) | Local + global (Conv QKV) |
| Inductive Bias | None | Local via conv in QKV |

**MicroGPT Entegrasyonu:**
- `microgpt-experiment/fanet_microgpt.py` dosyasında FANET-MicroGPT implementasyonu
- Conv1DStem: Sequence için 1D convolution stem
- ConvolutionalMHSA1D: 1D depthwise convolutions in QKV
- Standard transformer architecture + FANET attention enhancement

**Component'ler:**
- `DepthwiseSeparableConv`: MobileNet-style efficient convolution
- `MultiScaleCNNStem`: 3-stage hierarchical feature extraction
- `ConvolutionalMHSA`: Attention with local receptive field in QKV
- `HybridConvTransformerBlock`: CNN + Transformer hybrid block

**Dosyalar:**
- MNIST: `mnist-experiment/models/fanet.py`
- MicroGPT: `microgpt-experiment/fanet_microgpt.py`
- Diagram: `mnist-experiment/architecture_diagrams/fanet.svg`
