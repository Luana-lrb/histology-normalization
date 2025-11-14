# Análise Comparativa de Algoritmos de Normalização de Imagens Histológicas

## Sobre o Projeto

Este repositório contém a implementação e análise comparativa de diversos algoritmos de normalização de coloração para imagens histológicas, desenvolvido como parte de um projeto de Iniciação Científica.

A normalização de coloração é uma etapa crucial no processamento de imagens histológicas, pois reduz a variabilidade introduzida por diferentes protocolos de coloração, scanners e condições de aquisição, melhorando a performance de análises subsequentes.

## Objetivos

- Implementar e comparar diferentes técnicas de normalização de imagens histológicas
- Avaliar quantitativamente a eficácia de cada método
- Analisar o impacto da normalização em tarefas downstream (classificação, segmentação)
- Documentar vantagens, desvantagens e casos de uso de cada algoritmo

## Algoritmos Implementados

- **Reinhard** - Transferência de estatísticas de cor
- **Macenko** - Decomposição de manchas baseada em SVD
- **Vahadane** - Decomposição esparsa não-negativa
- **Modified Reinhard** - Variação modificada do método Reinhard
- **Mult-Target Macenko** - Extensão do Macenko para múltiplas imagens alvo
- **Histogram Matching** - Equalização de histogramas de cor
- **Zeng et al.** - Método de normalização baseado em aprendizado profundo

## Estrutura do Repositório (Pré-definição)
```
.
├── data/
│   ├── raw/                    # Imagens originais
│   ├── processed/              # Imagens normalizadas
│   ├── reference/              # Imagens de referência
│   └── annotations/            # Anotações e labels
├── src/
│   ├── normalization/          # Implementação dos algoritmos
│   │   ├── reinhard.py
│   │   ├── macenko.py
│   │   ├── vahadane.py
│   │   └── ...
│   ├── evaluation/             # Métricas de avaliação
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── utils/                  # Funções auxiliares
│   │   ├── io.py
│   │   ├── preprocessing.py
│   │   └── color_space.py
│   └── pipelines/              # Pipelines completos
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_algorithm_comparison.ipynb
│   ├── 03_quantitative_evaluation.ipynb
│   └── 04_visualizations.ipynb
├── experiments/
│   ├── configs/                # Arquivos de configuração
│   └── results/                # Resultados dos experimentos
├── .gitignore
└── README.md
```
## Uso

### Normalização de uma imagem

```python
from src.normalization import ReinhardNormalizer, MacenkoNormalizer

# Carregar imagem
image = load_image("data/raw/sample.png")
reference = load_image("data/reference/ref.png")

# Aplicar normalização
normalizer = ReinhardNormalizer()
normalized = normalizer.fit(reference).transform(image)

# Salvar resultado
save_image(normalized, "data/processed/sample_reinhard.png")
```

## Métricas de Avaliação

<!-- - **Métricas de similaridade de cor**: RMSE, SSIM, PSNR
- **Divergência de distribuição**: KL-divergence, Earth Mover's Distance
- **Métricas histológicas**: Consistência de manchas, preservação de estruturas
- **Impacto downstream**: Acurácia em classificação, IoU em segmentação -->

## Resultados

Os resultados detalhados dos experimentos podem ser encontrados em `experiments/results/`. Principais achados:

- [Será preenchido conforme os experimentos]

## Datasets Utilizados

-  Dataset de displasia oral - Adriano

## Referências

1. Reinhard, E., et al. (2001). Color transfer between images.
2. Macenko, M., et al. (2009). A method for normalizing histology slides for quantitative analysis.
3. Vahadane, A., et al. (2016). Structure-preserving color normalization and sparse stain separation.
- [Falta adicionar as outras referências]

## Autor

**Luana Rodrigues Borges**
- Instituição: Universidade Federal de Uberlândia - UFU
- Orientador: Marcelo Zanchetta do Nascimento
- Email: [luana.borges1@ufu.br]
- GitHub: [Luana-lrb](https://github.com/Luana-lrb)


**Status do Projeto**: 🚧 Em Desenvolvimento

**Última Atualização**: Novembro 2025
