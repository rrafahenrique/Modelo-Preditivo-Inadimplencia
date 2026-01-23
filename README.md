![Badge de Concluido](https://img.shields.io/badge/status-Conluído-green?style=for-the-badge)

# Desafio - Modelo Preditivo

## Objetivo Principal
Desenvolver um **modelo preditivo** capaz de estimar a probabilidade de inadimplência de cobranças mensais feitas aos clientes.

> [!IMPORTANT] 
> A **inadimplência** é definida da seguinte forma:
> Quando um pagamento for realizado com **5 dias ou mais de atraso** em relação à data de vencimento.

Existem *quatro bases de dados* com informações sobre os clientes, o comportamento mensal e os registros de pagamentos. Essas bases foram extraídas de um sistema de cobrança e representam um cenário realista de operação. As tabelas se relacionam principalmente por duas chaves:
- **ID_CLIENTE**: identifica cada cliente de forma única
- **SAFRA_REF**: representa o período de referência da cobrança 

A estrutura esperada de cada base é a seguinte:
- `base_cadastral.csv`: reúne informações cadastrais dos clientes, como data de cadastro, porte da empresa, CEP e domínio do e-mail. Cada linha representa um cliente único (ID_CLIENTE).
- `base_info.csv`: traz dados mensais relacionados ao cliente, como renda do mês anterior e número de funcionários. Cada linha representa um cliente em um determinado mês (ID_CLIENTE, SAFRA_REF).
- `base_pagamentos_desenvolvimento.csv`: cobranças e pagamentos já realizados. Cada linha representa uma cobrança mensal para um cliente (ID_CLIENTE, SAFRA_REF), com as datas de vencimento e pagamento disponíveis para construção da variável target.
- `base_pagamentos_teste.csv`: contém as cobranças mais recentes, para as quais o modelo deve prever a probabilidade de inadimplência. Cada linha representa uma cobrança mensal, sem a informação de pagamento.

O projeto deve conter todos os elementos necessários para **reproduzir a solução** e **avaliar as previsões geradas** para os casos presentes na base `base_pagamentos_teste.csv`.

Os seguintes itens são esperados:
- Um arquivo `.csv` chamado `submissao_case.csv`, contendo exatamente as seguintes colunas:
    - **ID_CLIENTE**
    - **SAFRA_REF**
    - **PROBABILIDADE_INADIMPLENCIA**

## Estrutura do Projeto
A estratégia utilizada aqui foi dividir o desafio em **três arquivos .ipynb**, cada um abordando uma etapa:
1. **PARTE I**: Manipulação do dataset com o objetivo de tratar valores ausentes e prepará-lo para a análise exploratória. O processo foi conduzido da seguinte forma:
- Identificação e contabilização dos dados ausentes;
- Definição de estratégias para imputação dos valores faltantes, incluindo:
- Uso da mediana;
- Uso da média;

> O código-fonte está disponível em: [DESAFIO - PARTE I.ipynb](https://github.com/rrafahenrique/Modelo-Preditivo-Inadimplencia/blob/main/DESAFIO%20-%20PARTE%20I.ipynb)
     

2. **PARTE II**: Realização da análise exploratória com o objetivo de compreender o comportamento das variáveis e extrair insights relevantes. Os processos incluíram:
- Construção de gráficos e visualizações para facilitar a interpretação dos dados;
- Identificação de padrões, tendências e relações entre as variáveis;
- Preparação do dataset para o treinamento dos modelos, incluindo:
- Codificação das variáveis categóricas por meio de `OneHotEncoder` e `OrdinalEncoder`;
- Engenharia de features a partir de variáveis temporais, com o tratamento e a extração de informações relevantes das colunas de data.  

> O código-fonte está disponível em: [DESAFIO - PARTE II.ipynb](https://github.com/rrafahenrique/Modelo-Preditivo-Inadimplencia/blob/main/DESAFIO%20-%20PARTE%20II.ipynb) 

3. **PARTE III**: Desenvolvimento e avaliação dos modelos de Machine Learning com foco na previsão da inadimplência. Nesta etapa, foram utilizados e comparados os seguintes algoritmos:
- Regressão Logística;
- Random Forest;
- Árvore de Decisão;
- K-Nearest Neighbors (KNN);
- LightGBM.

> O código-fonte está disponível em: [DESAFIO - PARTE III.ipynb](https://github.com/rrafahenrique/Modelo-Preditivo-Inadimplencia/blob/main/DESAFIO%20-%20PARTE%20III.ipynb)



