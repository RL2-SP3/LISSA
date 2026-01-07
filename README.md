# LISSA - Layer of Integration with Scikit-learn and Signal Analysis 

**LISSA** é uma biblioteca em Python para **padronização, controle e reprodução de gráficos científicos**, prioritariamente, com foco em **séries temporais**, **visualização comparativa** e **configuração declarativa via JSON**.

Também, inclui-se nesta biblioteca algumas considerações. 

O objetivo central é **separar lógica de visualização do código de análise**, permitindo que praticamente todos os parâmetros gráficos (estilo, textos, escalas, cores, layout, idioma) sejam controlados externamente.

---

## Motivação

Em projetos de análise de dados e sinais, é comum que:
- código de análise fique poluído com comandos de plot;
- ajustes visuais sejam repetidos em vários scripts;
- a reprodutibilidade gráfica dependa de estado global (`rcParams`) mal documentado.

A **LissaFigure** resolve isso ao:
- encapsular o estado da figura em uma classe;
- centralizar configurações em arquivos JSON;
- permitir encadeamento de métodos (fluent interface);
- reduzir dependência de ajustes manuais por gráfico.

---

## Principais características

- Controle gráfico via **JSON**
- Interface fluente (`.method().method().method()`)
- Suporte a **séries temporais**
- Integração direta com **pandas** e **matplotlib**
- Tradução automática de labels (opcional)
- Uso extensivo de `matplotlib.rcParams`
- Separação clara entre:
  - configuração
  - criação da figura
  - métodos de plot
  - finalização/exportação

---

## Estrutura típica do projeto

```text
lissafigure/
├── lissa_figure.py          # Classe principal LissaFigure
├── defaults.py              # Dicionários de parâmetros padrão
├── utils.py                 # Funções auxiliares
├── dictionaries/
│   ├── dictionaries.json    # Traduções e unidades
│   └── new_headers.json
├── plots/
│   └── example_plot.json    # Configuração de plot
└── README.md

