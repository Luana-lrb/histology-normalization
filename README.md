# Análise Comparativa de Algoritmos de Normalização de Imagens Histológicas

## Sobre o Projeto

Este repositório contém a implementação e análise comparativa de diversos algoritmos de normalização de coloração para imagens histológicas, desenvolvido como parte de um projeto de Iniciação Científica.

A normalização de coloração é uma etapa crucial no processamento de imagens histológicas, pois reduz a variabilidade introduzida por diferentes protocolos de coloração, scanners e condições de aquisição, melhorando a performance de análises subsequentes.

## Objetivos

- Implementar e comparar diferentes técnicas de normalização de imagens histológicas
- Comparar o comportamento dos métodos sob diferentes imagens de referência
- Fornecer uma base experimental organizada para análises qualitativas e quantitativas futuras


## Algoritmos Implementados

### Métodos Single-target (Utilizam uma única imagem de referência)

- **Reinhard** - Transferência de estatísticas de cor
- **Macenko** - Decomposição de manchas baseada em SVD
- **Vahadane** - Decomposição esparsa não-negativa
- **Modified Reinhard** - Variação do método Reinhard
- **Histogram Matching** - Correspondência de histogramas de cor
- **Zeng et al.** - Método de normalização baseado em aprendizado profundo

### Método Multi-target

**Mult-Target Macenko:**
- Extensão do Macenko para múltiplas imagens alvo. 
- Estratégia utilizada: avg-post.

### Implementação Original dos Autores

**Zeng et al. – Adaptive Color Deconvolution:**

- Implementado utilizando o código original disponibilizado pelos autores.
- Baseado em TensorFlow 1.x e Python 3.6
- Executado em ambiente isolado para garantir fidelidade ao método.


## Estrutura do Repositório (Pré-definição)
```
.
├── adaptive_color_deconvolution/   # Implementação original do Zeng et al.
│   ├── __init__.py
│   ├── stain_normalizer.py
│   └── acd.py
├── data/
│   ├── mini_raw/                   # Amostra das imagens originais
│   ├── mini_processed/             # Imagens da amostra normalizadas    
│   ├── raw/                        # Imagens originais
│   ├── processed/                  # Imagens normalizadas
│   ├── reference/                  # Imagens de referência
│   └── annotations/ 
├── src/           
│   ├── normalization/
│       ├── base.py
│       ├── histogram_matching.py
│       ├── macenko.py
│       ├── modified_reinhard.py
│       ├── multitarget_macenko.py
│       ├── reinhard.py
│       ├── run_all_normalizers.py
│       ├── utils.py
│       ├── vahadane.py
│       └──  zeng.py             
└── README.md
```

## Execução

> python run_all_normalizers.py

Este script executa automaticamente todas as estratégias, com exeção da normalização do Zeng at. al, que deve ser executada separadamente.

## Avaliação

As análises quantitativas e qualitativas incluem:
- Comparação visual entre imagens normalizadas
- Estudo do impacto da referência na distribuição de cores
- Avaliação futura do impacto em tarefas downstream (classificação e segmentação)

## Resultados
Os resultados detalhados dos experimentos podem ser encontrados em `data/processed`. 

## Dataset Utilizado

-  Dataset de displasia oral - Adriano(UFU)

## Referências

1. Reinhard, E., et al. (2001). Color transfer between images.
2. Macenko, M., et al. (2009). A method for normalizing histology slides for quantitative analysis.
3. Vahadane, A., et al. (2016). Structure-preserving color normalization and sparse stain separation.

```
@software{barbano2022torchstain,
  author       = {Carlo Alberto Barbano and André Pedersen},
  title        = {EIDOSLAB/torchstain: v1.2.0-stable},
  month        = aug,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v1.2.0-stable},
  doi          = {10.5281/zenodo.6979540},
  url          = {https://doi.org/10.5281/zenodo.6979540}
}
```
4. Zheng, Y., et al. (2019). Adaptive color deconvolution for histological WSI normalization.

```
@article{zhengCMPB2019,
  title   = {Adaptive color deconvolution for histological WSI normalization},
  author  = {Yushan Zheng and Zhiguo Jiang and Haopeng Zhang and Fengying Xie and Jun Shi and Chenghai Xue},
  journal = {Computer Methods and Programs in Biomedicine},
  volume  = {170},
  pages   = {107-120},
  doi     = {doi.org/10.1016/j.cmpb.2019.01.008},
  year    = {2019}
}
```
### Observação
O método Adaptive Color Deconvolution (Zeng et al.) foi executado utilizando a implementação original disponibilizada pelos autores, que depende de TensorFlow 1.x e Python 3.6. Para garantir fidelidade ao algoritmo, foi utilizado um ambiente isolado apenas para esse método, enquanto os demais algoritmos foram executados em ambiente Python moderno.

## Autor

**Luana Rodrigues Borges**
- Instituição: Universidade Federal de Uberlândia - UFU
- Orientador: Prof. Dr. Marcelo Zanchetta do Nascimento
- Email: [luana.borges1@ufu.br]
- GitHub: [Luana-lrb](https://github.com/Luana-lrb)


**Status do Projeto**: 🚧 Em Desenvolvimento

**Última Atualização**: Janeiro 2026
