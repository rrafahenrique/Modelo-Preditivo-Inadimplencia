import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

# Função que personaliza o método describe() do pandas 
# É necessário instalar o pacote jinja2 (pip install jinja2)
def descricão(df):
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
    resumo = pd.DataFrame({
        'Coluna': df.columns,
        'Tipo': df.dtypes.values,
        'Quantidade de Dados Não Vazios': df.notna().sum().values,
        'Quantidade de Dados Vazios': df.isna().sum().values,
        'Valores Únicos': df.nunique(),
        'Porcentagem de Valor Vazios (%)': (df.isna().mean() * 100).round(2).values
        
    })

    styled = (resumo.style
        .set_properties(**{
            'background-color': "#0f010194", 
            'border-color': 'black',
            'text-align': 'center'
        })
        .background_gradient(subset=['Porcentagem de Valor Vazios (%)'], cmap='Reds')
        .bar(subset=['Quantidade de Dados Vazios'], color='#BE0804')
        .set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#0d253f'),
                    ('color', 'white'),
                    ('text-align', 'center'),
                    ('font-size', '12px')
                ]
            }
        ])
    )
    return styled

#-------------------------------------------------------------------------------------------
# Gráfico dados ausentes
def plot_missing_values(
    df: pd.DataFrame,
    figsize: tuple = (10, 5),
    bar_color: str = '#1f77b4'
) -> pd.Series:
    """
    Plota a proporção de valores ausentes (%) por coluna em um DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    figsize : tuple, opcional
        Tamanho da figura (default: (15, 10)).
    bar_color : str, opcional
        Cor das barras do gráfico.

    Retorno
    -------
    pd.Series
        Série com o percentual de valores ausentes por coluna.
    """

    # Colunas com valores ausentes
    colunas_vazias = df.columns[df.isnull().any()]

    if len(colunas_vazias) == 0:
        print("Nenhuma coluna com valores ausentes encontrada.")
        return pd.Series(dtype=float)

    # Percentual de valores ausentes
    dados_ausentes = (df[colunas_vazias].isnull().sum() / len(df)) * 100

    # Plot
    plt.figure(figsize=figsize)
    plt.style.use('ggplot')
    plt.bar(dados_ausentes.index, dados_ausentes.values)

    plt.title('Proporção de Valores Vazios por Coluna (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Colunas', fontsize=14, fontweight='bold')
    plt.ylabel('Percentual de Ausência (%)', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Percentuais acima das barras
    for i, valor in enumerate(dados_ausentes.values):
        plt.text(
            i,
            valor + 0.1,
            f'{valor:.2f}%',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    plt.show()

#---------------------------------------------------------------------------

def plot_proportion_bar(
    df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str = '',
    ylabel: str = 'Proporção (%)',
    figsize: tuple = (10, 5),
    show_values: bool = True,
    palette: List[str] = ['#45769E', '#288990', '#A5C5C2', '#75C9E3', '#FFB629']
) -> pd.Series:
    """
    Plota um gráfico de barras com a proporção percentual de uma variável categórica.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    column : str
        Coluna categórica a ser analisada.
    title : str
        Título do gráfico.
    xlabel : str, opcional
        Rótulo do eixo X.
    ylabel : str, opcional
        Rótulo do eixo Y (default: 'Proporção (%)').
    figsize : tuple, opcional
        Tamanho da figura.
    show_values : bool, opcional
        Exibe os valores percentuais acima das barras.
    palette : list, opcional
        Lista de cores para o gráfico.

    Retorno
    -------
    pd.Series
        Série com as proporções (%) por categoria.
    """

    # Cálculo da proporção
    proportions = df[column].value_counts(normalize=True) * 100

    # Plot
    plt.figure(figsize=figsize)
    plt.style.use('ggplot')
    plt.bar(
        proportions.index.astype(str),
        proportions.values,
        color=palette[:len(proportions)]
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Valores acima das barras
    if show_values:
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

#----------------------------------------------------------------------
def plot_proportion_pie(
    df: pd.DataFrame,
    column: str,
    title: str,
    figsize: tuple = (10, 6),
    autopct: str = '%1.2f%%',
    startangle: int = 90,
    palette: List[str] = ['#45769E', '#288990', '#A5C5C2', '#75C9E3', '#FFB629']
) -> pd.Series:
    """
    Plota um gráfico de pizza com a proporção percentual de uma variável categórica.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    column : str
        Coluna categórica a ser analisada.
    title : str
        Título do gráfico.
    figsize : tuple, opcional
        Tamanho da figura.
    autopct : str, opcional
        Formato do percentual exibido no gráfico.
    startangle : int, opcional
        Ângulo inicial do gráfico.
    palette : list, opcional
        Lista de cores para o gráfico.

    Retorno
    -------
    pd.Series
        Série com as proporções (%) por categoria.
    """

    # Cálculo da proporção
    proportions = df[column].value_counts(normalize=True) * 100

    # Plot
    plt.figure(figsize=figsize)
    plt.style.use('ggplot')

    plt.pie(
        proportions.values,
        labels=proportions.index.astype(str),
        autopct=autopct,
        colors=palette[:len(proportions)],
        startangle=startangle,
        counterclock=False,
        wedgeprops={'edgecolor': 'white'}
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------
def plot_mean_and_distribution(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title_bar: str = 'Média por Categoria',
    title_box: str = 'Distribuição por Categoria',
    figsize: tuple = (20, 6),
    bar_palette: List[str] = ['#45769E', '#288990', '#A5C5C2'],
    box_palette: str = 'Set2',
    show_values: bool = True
) -> pd.Series:
    """
    Plota lado a lado:
    1) Gráfico de barras com a média
    2) Boxplot com a distribuição dos valores

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    group_col : str
        Coluna categórica para agrupamento.
    value_col : str
        Coluna numérica.
    title_bar : str
        Título do gráfico de barras.
    title_box : str
        Título do boxplot.
    figsize : tuple
        Tamanho da figura.
    bar_palette : list
        Paleta de cores do gráfico de barras.
    box_palette : str
        Paleta do seaborn para o boxplot.
    show_values : bool
        Exibe valores numéricos sobre as barras.

    Retorno
    -------
    pd.Series
        Série com a média por categoria.
    """

    # Cálculo da média
    mean_values = df.groupby(group_col)[value_col].mean()

    # Criação da figura
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.style.use('ggplot')

    # -------- Gráfico de barras --------
    axes[0].bar(mean_values.index.astype(str), mean_values.values, color=bar_palette[:len(mean_values)])

    axes[0].set_title(title_bar, fontsize=14, fontweight='bold')
    axes[0].set_xlabel(group_col)
    axes[0].set_ylabel(f'Média de {value_col}')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    if show_values:
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
    sns.boxplot(x=group_col, y=value_col, data=df, palette=box_palette, ax=axes[1])

    axes[1].set_title(title_box, fontsize=14, fontweight='bold')
    axes[1].set_xlabel(group_col)
    axes[1].set_ylabel(value_col)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------
def plot_segmento_por_porte(
    df,
    col_porte='PORTE',
    col_segmento='SEGMENTO_INDUSTRIAL',
    figsize=(10, 6),
    largura=0.25,
    titulo="Representação do Segmento por Porte de Empresa",
    ylabel="Porcentagem (%)",
    fontsize=12
):
    """
    Plota um gráfico de barras agrupadas mostrando a distribuição percentual
    dos segmentos industriais por porte de empresa.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame com os dados.
    col_porte : str
        Coluna que representa o porte da empresa.
    col_segmento : str
        Coluna que representa o segmento industrial.
    figsize : tuple
        Tamanho da figura.
    largura : float
        Largura de cada barra.
    titulo : str
        Título do gráfico.
    ylabel : str
        Rótulo do eixo Y.
    fontsize : int
        Tamanho da fonte base.
    """

    # Agrupamento e cálculo percentual
    porte_por_seg = (df.groupby(col_porte)[col_segmento].value_counts(normalize=True).mul(100))

    df_plot = porte_por_seg.unstack() # Converte MultiIndex para tabela

    n_groups = len(df_plot.index)
    n_seg = len(df_plot.columns)

    x = np.arange(n_groups) # largura de cada barra

    # calculamos deslocamentos centrados: por ex. para 3 barras -> [-largura, 0, +largura]
    offsets = (np.arange(n_seg) - (n_seg - 1) / 2) * largura

    plt.figure(figsize=figsize)

    # Plot das barras
    for j, seg in enumerate(df_plot.columns):
        positions = x + offsets[j]
        plt.bar(positions, df_plot[seg].values, width=largura, label=seg)

    # Eixos e título
    plt.xticks(x, df_plot.index, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(titulo, fontsize=fontsize + 2)

    # # Ajustar limite superior para que as labels caibam
    ymax = np.nanmax(df_plot.values)
    plt.ylim(0, ymax + max(5, 0.08 * ymax)) # margem de 5 unidades ou 8% do máximo

    # Labels de valor
    for j, seg in enumerate(df_plot.columns):
        positions = x + offsets[j]
        for xi, valor in zip(positions, df_plot[seg].values):
            if not np.isnan(valor):
                plt.text(
                    xi,
                    valor + (0.01 * ymax),
                    f"{valor:.1f}%",
                    ha='center',
                    va='bottom',
                    fontsize=fontsize - 2
                )

    plt.legend(title="Segmento Industrial")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
