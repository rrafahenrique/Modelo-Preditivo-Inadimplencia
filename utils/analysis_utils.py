import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# É necessário instalar o pacote jinja2 (pip install jinja2)
def df_summary_report(df):
    """
    Descreve valores estatísticos da dataframe.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    Retorno
    -------
    pd.Series
        Valores estatísticos e quantitativos estilizados.
    """

    moda = df.mode().iloc[0]

    mem_consumo = df.memory_usage(deep=True, index=False)/1024**2
    total = sum(mem_consumo)

    freq = df.apply(
        lambda col: col.value_counts(dropna=True).iloc[0]
        if not col.dropna().empty else None
    )


    resumo = pd.DataFrame({
        'Coluna': df.columns,            #Lista os nomes das colunas do DataFrame.
        'Tipo': df.dtypes.values,        #Retorna o tipo de dado (dtype) de cada coluna.
        'Quantidade de Dados Não Vazios': df.notna().sum().values,  #Conta a quantidade de valores não nulos por coluna.
        'Quantidade de Dados Vazios': df.isna().sum().values,       #Conta a quantidade de valores nulos (NaN) por coluna.
        'Valores Únicos': df.nunique().values,       #Conta a quantidade de valores únicos de cada coluna
        'Valor mais Frequente': moda,    #Calcula a moda de cada coluna    
        'Frequência': freq,               #Mostra a frequência da moda
        'Porcentagem de Unicidade': ((df.nunique() / len(df)) * 100).round(2).values,    #Cardinalidade - baixa cardinalidade indica alta repetição; alta cardinalidade indica baixa repetição
        'Porcentagem de Valor Vazios (%)': (df.isna().mean() * 100).round(2).values,  #Calcula a porcentagem de valores nulos por coluna.
        f'Consumo de Memória - Total: {total:.2f} (MB)': mem_consumo    #Calcula a quantidade de memória usada pelo dataset
    }).reset_index(drop=True)

    #colormaps
    cmap_vazios = sns.light_palette("#BD2A2E", as_cmap=True)
    cmap_unicidade = sns.light_palette("#13678A", as_cmap=True)
    cmap_unique = sns.light_palette("#1f77b4", as_cmap=True)
    cmap_freq = sns.light_palette("#13678A", as_cmap=True)

    styled = (resumo.style
        .set_properties(**{
            'background-color': "#101719",
            'color': '#E0E0E0',  
            'border': '1px solid #2F3D40',
            'text-align': 'center'
        })
        .background_gradient(subset=['Porcentagem de Valor Vazios (%)'], cmap=cmap_vazios, vmin=0, vmax=100)
        .background_gradient(subset=['Porcentagem de Unicidade'], cmap=cmap_unicidade, vmin=0, vmax=100)
        .background_gradient(subset=["Valores Únicos"],cmap=cmap_unique)
        .background_gradient(subset=["Frequência"],cmap=cmap_freq)
        .background_gradient(subset=[f"Consumo de Memória - Total: {total:.2f} (MB)"], cmap=cmap_freq)
        .bar(subset=['Quantidade de Dados Vazios'], color="#BD2A2E")
        .format({'Porcentagem de Valor Vazios (%)': '{:.2f}', 'Porcentagem de Unicidade': '{:.2f}'})
        .set_table_styles([{
                'selector': 'th',
                'props': [
                    ('background-color', "#0c2845"),
                    ('color', 'white'),
                    ('text-align', 'center'),
                    ('font-size', '13px')
                ]
            }
        ])
        .set_properties(
            subset=pd.IndexSlice[:, resumo.columns[0]],**{
                'background-color': '#012030',
                'font-weight': 'bold',
                'color': '#FFFFFF'
            })
        
    )
    return styled
#----------------------------------------------------------------------------------------------------------------------
def df_describe_report(df):
    """
    Gera um relatório estatístico descritivo formatado para variáveis numéricas.

    A função calcula métricas como média, desvio padrão, mínimo,
    quartis e máximo, exibindo o resultado em formato transposto
    e com estilização visual utilizando pandas Styler.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame contendo os dados a serem analisados.

    Retorno
        Tabela estilizada com estatísticas descritivas.
    """

    stats = pd.DataFrame({
        "Média": df.mean(numeric_only=True),
        "Desvio Padrão": df.std(numeric_only=True),
        "Valor Mínimo": df.min(numeric_only=True),
        "Primeiro Quartil - Q1": df.quantile(0.25, numeric_only=True),
        "Segundo Quartil - Q2/Mediana": df.quantile(0.50, numeric_only=True),
        "Terceiro Quartil - Q3": df.quantile(0.75, numeric_only=True),
        "Valor Máximo": df.max(numeric_only=True)
    }).T.reset_index().rename(columns={"index": "Estatística"})
    
    #colormaps
    cmap_cor = sns.light_palette("#13678A", as_cmap=True)

    styled = (stats.style 
            .set_properties(**{ 
            'background-color': "#101719", 
            'color': '#E0E0E0', 
            'border': '1px solid #2F3D40', 
            'text-align': 'center' 
            }) 
    
        #gradient para todas as colunas numéricas 
        .background_gradient(cmap=cmap_cor) 
        .format(precision=3) # formatação numérica 
        .set_table_styles([{ 
            'selector': 'th', 
            'props': [ 
                ('background-color', "#0c2845"), 
                ('color', 'white'), 
                ('text-align', 'center'), 
                ('font-size', '13px') 
                ]} 
            ])
        .set_properties(
            subset=pd.IndexSlice[:, stats.columns[0]],**{
                'background-color': '#012030',
                'font-weight': 'bold',
                'color': '#FFFFFF'
        })
            
        ) 
    
    return styled
#----------------------------------------------------------------------------------------------
# Gráfico dados ausentes
def plot_missing_values(df):
    """
    Plota a proporção de valores ausentes (%) por coluna em um DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    Retorno
    -------
    pd.Series
        Série com o percentual de valores ausentes por coluna.
    """

    # Colunas com valores ausentes
    colunas_vazias = df.columns[df.isnull().any()]

    # Tratamento de erro
    if len(colunas_vazias) == 0:
        print("Nenhuma coluna com valores ausentes encontrada.")
        return pd.Series(dtype=float)

    # Percentual de valores ausentes
    dados_ausentes = ((df[colunas_vazias].isnull().sum() / len(df)) * 100).sort_values()
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.barh(dados_ausentes.index, dados_ausentes.values, color = "#BD2A2E")
    plt.title('Proporção de Valores Vazios por Coluna (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Percentual de Ausência (%)', fontsize=14)
    plt.ylabel('Colunas', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    for i, valor in enumerate(dados_ausentes.values):
        plt.text(
            valor + 0.1,
            i,
            f'{valor:.2f}%',
            va='center',
            fontsize=10
        )

    plt.show()
#----------------------------------------------------------------------------------------
def plot_proportion_bar(df, column):
    """
    Plota um gráfico de barras com a proporção percentual de uma variável categórica.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    column : str
        Coluna categórica a ser analisada.
    Retorno
    -------
    pd.Series
        Série com as proporções (%) por categoria.
    """
    palette = ["#008F8C" ,"#015958", "#023535", "#4C5958", "#BFBFBF", "#D92525"]
    
    # Cálculo da proporção
    proportions = df[column].value_counts(normalize=True) * 100

    # Plot
    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')
    plt.bar(
        proportions.index.astype(str),
        proportions.values,
        color=palette[:len(proportions)]
    )

    plt.title(f"Proporção do {column} (%)", fontsize=14, fontweight='bold')
    plt.ylabel("Proporção (%)", fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Valores acima das barras
    for i, valor in enumerate(proportions.values):
        plt.text(
            i,
            valor + 0.1,
            f'{valor:.2f}%',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    plt.show()
#---------------------------------------------------------------------------------
def plot_mean_and_distribution(df, coluna1, coluna2):
    """
    Plota lado a lado:
    1) Gráfico de barras com a média
    2) Boxplot com a distribuição dos valores

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    coluna1 : str
        Coluna categórica para agrupamento.
    coluna2 : str
        Coluna numérica.
    Retorno
    -------
    pd.Series
        Série com a média por categoria.
    """

    # Cálculo da média
    mean_values = df.groupby(coluna1)[coluna2].mean()
    
    palette = ["#008F8C" ,"#015958", "#023535"]
    palette_invertida = ["#023535","#015958", "#008F8C"]

    # Criação da figura
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    plt.style.use('ggplot')

    # -------- Gráfico de barras --------
    axes[0].bar(mean_values.index.astype(str), mean_values.values, color = palette)

    axes[0].set_title("Média de Funcionários por Tamanho da Empresa", fontsize=14, fontweight='bold')
    axes[0].set_xlabel(coluna1)
    axes[0].set_ylabel(f'Média de {coluna2}')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    for i, valor in enumerate(mean_values.values):
        axes[0].text(
            i,
            valor + (mean_values.max() * 0.02),
            f'{valor:.2f}',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    # -------- Boxplot --------
    sns.boxplot(x=coluna1, y=coluna2, data=df, palette=palette_invertida, ax=axes[1])

    axes[1].set_title("Distribuição de Funcionários por Tamanho de Empresa", fontsize=14, fontweight='bold')
    axes[1].set_xlabel(coluna1)
    axes[1].set_ylabel(coluna2)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
#--------------------------------------------------------------------------------------------------------------
def plot_segmento_por_porte(df):
    """
    Plota um gráfico de barras agrupadas mostrando a distribuição percentual
    dos segmentos industriais por porte de empresa.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame com os dados.
    """
    
    # Agrupamento e cálculo percentual
    porte_por_seg = (df.groupby("PORTE")["SEGMENTO_INDUSTRIAL"].value_counts(normalize=True).mul(100))
    palette = ["#023535", "#4C5958", "#BFBFBF"]

    df_plot = porte_por_seg.unstack() # Converte MultiIndex para tabela

    n_groups = len(df_plot.index)
    n_seg = len(df_plot.columns)

    x = np.arange(n_groups) # largura de cada barra
    largura=0.25

    # calculamos deslocamentos centrados: por ex. para 3 barras -> [-largura, 0, +largura]
    offsets = (np.arange(n_seg) - (n_seg - 1) / 2) * largura

    plt.figure(figsize = (12, 7))

    # Plot das barras
    for j, seg in enumerate(df_plot.columns):
        positions = x + offsets[j]
        plt.bar(positions, df_plot[seg].values, width=largura, label=seg, color = palette[j])

    # Eixos e título
    plt.title("Distribuição Percentual dos Segmentos por Porte", fontsize=14 + 2)
    plt.xticks(x, df_plot.index, fontsize=14)
    plt.ylabel("Porcentagem (%)", fontsize=14)

    # # Ajustar limite superior para que as labels caibam
    ymax = np.nanmax(df_plot.values)
    plt.ylim(0, ymax + max(5, 0.08 * ymax)) # margem de 5 unidades ou 8% do máximo

    # Labels de valor
    for j, seg in enumerate(df_plot.columns):
        positions = x + offsets[j]
        for i, valor in zip(positions, df_plot[seg].values):
            if not np.isnan(valor):
                plt.text(
                    i,
                    valor + (0.01 * ymax),
                    f"{valor:.1f}%",
                    ha='center',
                    va='bottom',
                    fontsize=14 - 2
                )

    plt.legend(title="Segmento Industrial")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    plt.show()
    #----------------------------------------------------------------------------------
def plot_pag(df):
    """
    Plota lado a lado:
    1) Gráfico de barras com a média do dia de pagamento
    2) Gráfico de barras com os dias mais frequentes de pagamento
    3) Gráfico de barras do atrado médio por dia de vencimento

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    Retorno
    -------
    pd.Series
        Série com a média por categoria.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20,6))
    plt.style.use('ggplot')

    #------------Gráfico 1-------------------------------
    df['dia'] = df['DATA_VENCIMENTO'].dt.day
    cont_dia = df['dia'].value_counts().sort_index()

    axes[0].bar(cont_dia.index, cont_dia.values, color = "#008F8C")
    axes[0].set_title('Distribuição por dia do mês')
    axes[0].set_xlabel('Dia do mês')
    axes[0].set_ylabel('Quantidade')

    #-----------Gráfico 2-----------------------------------

    df['dia_semana'] = df['DATA_VENCIMENTO'].dt.day_name()
    ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    cont_sem = df['dia_semana'].value_counts().reindex(ordem)

    axes[1].bar(cont_sem.index, cont_sem.values, color = "#008F8C")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_xlabel("Dias da Semana")
    axes[1].set_title('Distribuição por dia da semana')

    #--------------Gráfico 3-------------------------------------
    df['atraso'] = (df['DATA_PAGAMENTO'] - df['DATA_VENCIMENTO']).dt.days

    teste = df.groupby('dia')['atraso'].mean()

    axes[2].bar(teste.index, teste.values, color = "#008F8C")
    axes[2].set_title('Atraso Médio por Dia do Vencimento')
    axes[2].set_xlabel('Dia do Mês')
    axes[2].set_ylabel('Atraso Médio (Dias)')

    plt.tight_layout()
    plt.show()
#------------------------------------------------------------------------------------------
def plot_churned(df):
    """
    Plota lado a lado:
    São 3 gráficos de barras que representam o churned em relação: Segmento industrial,
    taxa e porte da empresa

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    Retorno
    -------
    pd.Series
        Série com a média por categoria.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20,6))
    plt.style.use('ggplot')

    #---------------------Gráfico 1---------------------------------------------
    seg_churned = df.groupby("SEGMENTO_INDUSTRIAL")["Churned"].value_counts().unstack()

    x = np.arange(len(seg_churned.index))
    largura = 0.35

    axes[0].bar(x - largura/2, seg_churned[0], largura, label='Não Churn', color="#008F8C")
    axes[0].bar(x + largura/2, seg_churned[1], largura, label='Churn', color="#D92525")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seg_churned.index)

    axes[0].set_title('Churn por Segmento Industrial')
    axes[0].set_xlabel('Segmento')
    axes[0].set_ylabel('Quantidade')

    axes[0].legend()

    # Adicionar valores nas barras
    for p in axes[0].patches:
        axes[0].annotate(
            f'{int(p.get_height())}',                # valor
            (p.get_x() + p.get_width() / 2, p.get_height()),  # posição
            ha='center',
            va='bottom',
            fontsize=9
        )

    #------------------Gráfico 2-----------------------------------------------
    taxa_churned = df.groupby("TAXA")["Churned"].value_counts().unstack()

    x = np.arange(len(taxa_churned.index))
    largura = 0.35

    axes[1].bar(x - largura/2, taxa_churned[0], largura, label='Não Churn', color="#008F8C")
    axes[1].bar(x + largura/2, taxa_churned[1], largura, label='Churn', color="#D92525")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(taxa_churned.index)

    axes[1].set_title('Churn por Taxa')
    axes[1].set_xlabel('Segmento')
    axes[1].set_ylabel('Quantidade')

    axes[1].legend()

    # Adicionar valores nas barras
    for p in axes[1].patches:
        axes[1].annotate(
            f'{int(p.get_height())}',                # valor
            (p.get_x() + p.get_width() / 2, p.get_height()),  # posição
            ha='center',
            va='bottom',
            fontsize=9
        )

    #----------------Gráfico 3----------------------------------------------
    porte_churned = df.groupby("PORTE")["Churned"].value_counts().unstack()

    x = np.arange(len(porte_churned.index))
    largura = 0.35

    axes[2].bar(x - largura/2, porte_churned[0], largura, label='Não Churn', color="#008F8C")
    axes[2].bar(x + largura/2, porte_churned[1], largura, label='Churn', color="#D92525")

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(porte_churned.index)

    axes[2].set_title('Churn por Porte da Empresa')
    axes[2].set_xlabel('Segmento')
    axes[2].set_ylabel('Quantidade')

    axes[2].legend()

    # Adicionar valores nas barras
    for p in axes[2].patches:
        axes[2].annotate(
            f'{int(p.get_height())}',                # valor
            (p.get_x() + p.get_width() / 2, p.get_height()),  # posição
            ha='center',
            va='bottom',
            fontsize=9
        )


    plt.tight_layout()
    plt.show()

