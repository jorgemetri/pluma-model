# src/feature_utils.py

import pandas as pd
import numpy as np
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import get_close_matches

# Bibliotecas de Terceiros
from astral.sun import sun
from astral import LocationInfo

# Scikit-Learn e Transformadores
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer  # Necessário para ativar o MICE
from sklearn.impute import IterativeImputer

# --- Definição Constante do Local ---
LAT_FIXA = -20.78 
LON_FIXA = -40.58
NOME_FUSO = 'America/Sao_Paulo' 

# ==============================================================================
# FUNÇÕES DE SUPORTE (EDA e Pré-processamento Funcional)
# ==============================================================================

def adicionar_periodo_dia(df: pd.DataFrame, coluna_timestamp_str: str) -> pd.DataFrame:
    """
    Adiciona uma coluna 'periodo' (Dia/Noite) a um DataFrame com base 
    na latitude, longitude e fuso horário fixos.
    """
    try:
        obj_fuso = pytz.timezone(NOME_FUSO)
        loc = LocationInfo(latitude=LAT_FIXA, longitude=LON_FIXA, timezone=NOME_FUSO)
        obs = loc.observer
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Erro: Fuso horário '{NOME_FUSO}' não reconhecido.")
        return df 

    df_proc = df.copy()

    # 1. Converter strings para datetimes (ISO 8601 com hífens)
    ts_naive = pd.to_datetime(df_proc[coluna_timestamp_str], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # 2. Aplicar o fuso horário (Aware)
    ts_aware = ts_naive.apply(lambda x: obj_fuso.localize(x, is_dst=None) if pd.notnull(x) else pd.NaT)

    # 3. Extrair datas únicas para cálculo solar
    datas_unicas = ts_aware.dropna().dt.date.unique()

    calculos_sol = {}
    for data_dt in datas_unicas:
        s = sun(obs, date=data_dt, tzinfo=obj_fuso) 
        calculos_sol[data_dt] = {
            'nascer_sol_temp': s['sunrise'],
            'por_sol_temp': s['sunset']
        }

    # 4. DataFrame de Lookup Solar
    if not calculos_sol:
        return df
        
    df_sol = pd.DataFrame.from_dict(calculos_sol, orient='index')
    df_sol.index.name = 'data_local_temp'
    df_sol.index = pd.to_datetime(df_sol.index).tz_localize(NOME_FUSO)

    # 5. Merge e Comparação
    df_proc['timestamp_aware_temp'] = ts_aware
    df_proc['data_local_temp'] = ts_aware.dt.normalize() 
    
    df_proc = df_proc.merge(df_sol, on='data_local_temp', how='left')

    condicao_dia = (df_proc['timestamp_aware_temp'] >= df_proc['nascer_sol_temp']) & \
                   (df_proc['timestamp_aware_temp'] <= df_proc['por_sol_temp'])

    df_proc['periodo'] = np.where(condicao_dia, 'Dia', 'Noite')

    # 6. Limpeza
    colunas_para_remover = [
        'nascer_sol_temp', 'por_sol_temp',
        'timestamp_aware_temp', 'data_local_temp'
    ]
    df_proc = df_proc.drop(columns=colunas_para_remover)
    
    return df_proc

def PreTraining(df: pd.DataFrame):
    """
    Realiza a limpeza, filtragem e engenharia de features finais antes do treino.
    Retorna y (Target) e X (Features).
    """
    df_proc = df.copy()
    
    # Engenharia Solar
    df_proc = adicionar_periodo_dia(df_proc, coluna_timestamp_str='t')
    df_proc = df_proc.rename(columns={'periodo': 'Dia/Noite'})
    
    # Filtra apenas o periodo 'Dia'
    df_proc = df_proc.loc[df_proc['Dia/Noite'] == 'Dia'].copy()

    # Engenharia de Tempo (Minutos)
    if 'Hora.1' in df_proc.columns:
        df_proc['Tempo'] = (pd.to_timedelta(df_proc['Hora.1']).dt.total_seconds() / 60).astype(int)

    # One-Hot Encoding
    if 'Campanha' in df_proc.columns:
        df_proc = pd.get_dummies(df_proc, columns=['Campanha'], dtype=int)

    # Remoção de colunas desnecessárias
    cols_drop = [
        "Unnamed: 0", "Amostragem", "Dia/Noite", "t", "n", 
        "Hora.1", "Ano", "Legenda Y", "Dia?", "Nascer do sol", 
        "Pôr do sol", "Hora", "Data"
    ]
    df_proc = df_proc.drop(columns=cols_drop, errors='ignore')

    # Separação X e y
    if 'Y' in df_proc.columns:
        y = df_proc['Y'].copy()
        X = df_proc.drop(columns=['Y'])
    else:
        y = None
        X = df_proc

    return y, X

def normalizar_dados(df):
    """Recebe um DataFrame e normaliza as colunas numéricas para a escala 0 a 1."""
    df_norm = df.copy()
    colunas_numericas = df_norm.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df_norm[colunas_numericas] = scaler.fit_transform(df_norm[colunas_numericas])
    return df_norm

def adicionar_caixa_estatisticas(ax, data_series):
    """Adiciona caixa de estatísticas e contagem de outliers ao gráfico."""
    stats = data_series.dropna().describe()
    q1, q2, q3 = stats.loc['25%'], stats.loc['50%'], stats.loc['75%']
    media, std = stats.loc['mean'], stats.loc['std']
    total_pontos = int(stats.loc['count'])

    iqr = q3 - q1
    inner_low, inner_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outer_low, outer_high = q1 - 3.0 * iqr, q3 + 3.0 * iqr

    mask_nivel2 = (data_series < outer_low) | (data_series > outer_high)
    contagem_nivel2 = mask_nivel2.sum()
    mask_total = (data_series < inner_low) | (data_series > inner_high)
    contagem_nivel1 = mask_total.sum() - contagem_nivel2

    texto_stats = (f"Total: {total_pontos}\nMédia: {media:.2f}\nStd: {std:.2f}\n"
                   f"Q1: {q1:.2f}\nQ2: {q2:.2f}\nQ3: {q3:.2f}")
    texto_outliers = (f"Nível 1: {contagem_nivel1}\nNível 2: {contagem_nivel2}")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
    ax.text(0.97, 0.97, f"{texto_stats}\n\n--- Outliers ---\n{texto_outliers}", 
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

def plotar_boxplots(df):
    plt.figure(figsize=(19, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title("Distribuição dos Dados (Boxplot)")
    plt.show()

def MedidasEstatisticas(df: pd.DataFrame, colunasUS3, decimals: int = 3) -> pd.DataFrame:
    if isinstance(colunasUS3, (str, bytes)):
        colunas = [colunasUS3]
    else:
        try:
            colunas = list(colunasUS3)
        except TypeError:
            colunas = [str(colunasUS3)]

    metricas = ["linhas_naonula", "q1", "q2", "q3", "media", "outlier1", "outlier2", "%outlier1", "%outlier2"]
    resultados = {}

    for col_req in colunas:
        if col_req not in df.columns:
            sugestoes = get_close_matches(col_req, df.columns.tolist(), n=3, cutoff=0.6)
            raise KeyError(f"Coluna '{col_req}' não encontrada. Sugestões: {sugestoes}?")

        s = pd.to_numeric(df[col_req], errors='coerce').dropna()
        n = len(s)

        if n == 0:
            resultados[col_req] = {m: np.nan if m != 'linhas_naonula' else 0 for m in metricas}
            resultados[col_req]['outlier1'] = 0
            resultados[col_req]['outlier2'] = 0
            continue

        q1, q2, q3 = s.quantile(0.25), s.quantile(0.50), s.quantile(0.75)
        media, iqr = s.mean(), q3 - q1
        out1 = ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum()
        out2 = ((s < (q1 - 3.0 * iqr)) | (s > (q3 + 3.0 * iqr))).sum()

        resultados[col_req] = {
            "linhas_naonula": int(n), "q1": round(q1, decimals), "q2": round(q2, decimals),
            "q3": round(q3, decimals), "media": round(media, decimals),
            "outlier1": int(out1), "outlier2": int(out2),
            "%outlier1": round(100 * out1 / n, decimals), "%outlier2": round(100 * out2 / n, decimals),
        }

    return pd.DataFrame(resultados).reindex(metricas)

def qtdColunasNulas(df):
    """
    Recebe um dataframe em pandas e retorna um df com as colunas nulas e suas contagens.
    """
    dados_faltantes = df.isnull().sum()
    colunas_com_nulos = dados_faltantes[dados_faltantes > 0]
    return colunas_com_nulos.to_frame().T


# ==============================================================================
# CLASSES TRANSFORMER (Compatíveis com Sklearn Pipeline)
# ==============================================================================

class FeatureCyclic(BaseEstimator, TransformerMixin):
    """
    Transforma variáveis cíclicas (Dia, Mês, DiaSemana, Hora) em coordenadas seno/cosseno.
    """
    def __init__(self, col_hora='Hora.1', col_dia='Dia', col_mes='Mes', col_dia_semana='Dia da semana'):
        self.col_hora = col_hora
        self.col_dia = col_dia
        self.col_mes = col_mes
        self.col_dia_semana = col_dia_semana

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Transformação da Hora (Minutos do dia - ciclo 1440 min)
        if self.col_hora in X.columns:
            time_deltas = pd.to_timedelta(X[self.col_hora], errors='coerce')
            minutes = time_deltas.dt.total_seconds() / 60.0
            X['Tempo_sin'] = np.sin(2 * np.pi * minutes / 1440)
            X['Tempo_cos'] = np.cos(2 * np.pi * minutes / 1440)
        
        # 2. Transformação do Dia do Mês (1-31)
        if self.col_dia in X.columns:
            X['Dia_sin'] = np.sin(2 * np.pi * X[self.col_dia] / 31)
            X['Dia_cos'] = np.cos(2 * np.pi * X[self.col_dia] / 31)

        # 3. Transformação do Mês (1-12)
        if self.col_mes in X.columns:
            X['Mes_sin'] = np.sin(2 * np.pi * X[self.col_mes] / 12)
            X['Mes_cos'] = np.cos(2 * np.pi * X[self.col_mes] / 12)

        # 4. Transformação do Dia da Semana (0-6)
        if self.col_dia_semana in X.columns:
            X['DiaSemana_sin'] = np.sin(2 * np.pi * X[self.col_dia_semana] / 7)
            X['DiaSemana_cos'] = np.cos(2 * np.pi * X[self.col_dia_semana] / 7)
            
        return X


class CappingTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica Capping (Winsorization) em outliers baseado no IQR.
    """
    def __init__(self, cols_to_cap=None, factor=1.5):
        self.cols_to_cap = cols_to_cap
        self.factor = factor
        self.limits_ = {}

    def fit(self, X, y=None):
        if self.cols_to_cap is None:
            self.cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.cols = self.cols_to_cap

        for col in self.cols:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.limits_[col] = (Q1 - (self.factor * IQR), Q3 + (self.factor * IQR))
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.limits_.items():
            if col in X.columns:
                X[col] = np.clip(X[col], lower, upper)
        return X


class OutlierIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Cria colunas binárias (flags) indicando se o valor original era um outlier.
    """
    def __init__(self, cols_to_check=None, factor=1.5):
        self.cols_to_check = cols_to_check
        self.factor = factor
        self.limits_ = {}

    def fit(self, X, y=None):
        if self.cols_to_check is None:
            self.cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.cols = self.cols_to_check

        for col in self.cols:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.limits_[col] = (Q1 - self.factor * IQR, Q3 + self.factor * IQR)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.limits_.items():
            if col in X.columns:
                new_col_name = f"{col}_outlier_ind"
                is_outlier = (X[col] < lower) | (X[col] > upper)
                X[new_col_name] = is_outlier.astype(int)
        return X


class MiceImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica imputação MICE (IterativeImputer).
    
    Args:
        cols_to_impute (list, opcional): Lista de colunas para imputar. 
                                         Se None, imputa TODAS as colunas numéricas.
        max_iter (int): Número máximo de iterações do MICE.
        random_state (int): Semente para reprodutibilidade.
    """
    def __init__(self, cols_to_impute=None, max_iter=10, random_state=42):
        self.cols_to_impute = cols_to_impute
        self.max_iter = max_iter
        self.random_state = random_state
        self.imputer = None
        self.cols_processing_ = None # Colunas que sofrerão o fit
        self.cols_remaining_ = None  # Colunas que não serão tocadas
        self.output_order_ = None

    def fit(self, X, y=None):
        self.output_order_ = X.columns.tolist()
        
        # 1. Definir quais colunas serão imputadas
        if self.cols_to_impute is None:
            # Comportamento padrão: Todas as numéricas
            self.cols_processing_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Comportamento manual: Apenas as listadas (validar se existem no df)
            self.cols_processing_ = [c for c in self.cols_to_impute if c in X.columns]
            
        # 2. Definir o resto (colunas categóricas ou numéricas que não queremos imputar)
        self.cols_remaining_ = [c for c in X.columns if c not in self.cols_processing_]
        
        # 3. Fit do Imputer
        if self.cols_processing_:
            self.imputer = IterativeImputer(max_iter=self.max_iter, random_state=self.random_state)
            self.imputer.fit(X[self.cols_processing_])
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # Se não houver colunas para processar, retorna X como está
        if not self.cols_processing_:
            return X
            
        # Separa os dados
        X_to_impute = X[self.cols_processing_]
        X_rest = X[self.cols_remaining_]
        
        # Aplica a transformação
        X_imputed_array = self.imputer.transform(X_to_impute)
        
        X_imputed = pd.DataFrame(
            X_imputed_array, 
            columns=self.cols_processing_, 
            index=X.index
        )
        
        # Reconcatena
        X_final = pd.concat([X_rest, X_imputed], axis=1)
        
        # Garante a ordem original das colunas
        X_final = X_final[self.output_order_]
        
        return X_final
    
class FeatureDateExtractor(BaseEstimator, TransformerMixin):
    """
    Processa uma coluna de data para extrair 'Dia' e 'Mes',
    e remove a coluna original ao final.
    """
    def __init__(self, col_data='Data'):
        self.col_data = col_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Evita SettingWithCopyWarning
        X = X.copy()
        
        # Verifica se a coluna existe
        if self.col_data in X.columns:
            # Garante que a coluna esteja em formato datetime
            # (Caso já esteja, isso não causa problemas)
            series_data = pd.to_datetime(X[self.col_data], errors='coerce')
            
            # Criação das colunas solicitadas
            X['Dia'] = series_data.dt.day
            X['Mes'] = series_data.dt.month
            
            # DICA EXTRA: Como sua próxima classe usa 'Dia da semana', 
            # você provavelmente vai querer criar ela aqui também:
            # X['Dia da semana'] = series_data.dt.dayofweek 
            
            # Remove a coluna original 'Data'
            X.drop(columns=[self.col_data], inplace=True)
            
        return X