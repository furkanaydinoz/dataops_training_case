# DataOps Experiment

> "Veriniz sağlıklıysa model mimariniz model başarımını üst düzey etkilemez."

İki deney içeren kapsamlı bir repo:

## 1. MNIST Experiment

10 farklı derin öğrenme modeli ile data > architecture hipotezinin testi.

**Modeller:** MLP (3 çeşit), CNN (3 çeşit), ResNet, ViT, CapsuleNet, GNN

**Sonuç:** Tüm modeller %95+ accuracy — mimari farkı minimal.

Detaylar: [mnist-experiment/report.md](mnist-experiment/report.md)

## 2. MicroGPT on TinyStories

Karpathy'nin microGPT makalesinden ilham alan LLM deneyi.

**Model:** 4-layer transformer, 256 dim, 4 heads

**Sonuç:** TinyStories üzerinde anlamlı metin üretimi.

Detaylar: [microgpt-experiment/report.md](microgpt-experiment/report.md)

## Kurulum

```bash
pip install torch torchvision matplotlib numpy tqdm

# MNIST deneyi
python mnist-experiment/train.py

# MicroGPT deneyi
python microgpt-experiment/train.py
```

## Sonuçlar

Görseller ve metrikler için:
- `mnist-experiment/results/`
- `microgpt-experiment/results/`