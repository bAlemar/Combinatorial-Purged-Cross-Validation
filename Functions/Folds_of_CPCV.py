import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
from Data_and_Model import Dados
from sklearn.model_selection import cross_val_score
from cpcv import cpcv
def score_results(modelo,nome_modelo,X_train,y_train,n_splits=5,n_test_splits=2,intervalo=15):
    # Validação Cruzada
    cv_purge = cpcv(n_splits=n_splits,n_test_splits=n_test_splits,intervalo=intervalo)
    scores = cross_val_score(modelo,X_train,y_train,cv=cv_purge,scoring='roc_auc')
    # Salvando Scores da CV
    tipo_dado = 'binario'
    file_name = f'./resultados/cv_scores/{nome_modelo}_{tipo_dado}_CV_{n_splits}_{n_test_splits}.txt'
    np.savetxt(file_name,scores)
    return scores
def rank_chart_cpcv(arquivo,rank,n_splits,n_test_splits):
  # Carregando Base de Teste:
    funcao_modelo = Dados()
    funcao_modelo.Carregar_dados()
    funcao_modelo.Variaveis(categoria='binario')
    funcao_modelo.train_test_split()
    X_train = funcao_modelo.X_train
    y_train = funcao_modelo.y_train
    X_test = funcao_modelo.X_test
    y_test = funcao_modelo.y_test


    # Abra o arquivo em modo de leitura
    with open(arquivo, 'r') as cv_scores:
        # Leia as linhas do arquivo
        lines = cv_scores.readlines()

    # Remova possíveis caracteres de quebra de linha
    scores = [line.strip() for line in lines]

    # Converta as linhas em um array NumPy
    array_scores = np.array(scores, dtype=float)
    # Criando Df de scores, para organizar em ordem crescente
    df_scores = pd.DataFrame({
        'index':range(len(array_scores)),
        'scores':array_scores
    })
    df_scores.sort_values('scores',ascending=False,inplace=True)
    df_scores.reset_index(inplace=True)

    # Scores
    score = df_scores.iloc[rank,2]
    pos_score = df_scores[df_scores['scores'] == score]['index'].iloc[0]

    # Verificando folds dessa acurácia:

    cv = cpcv(n_splits, n_test_splits) 


    # Crie uma variável para contar os conjuntos
    count = 0

    train_index = []
    test_index = []

    for train_indices, test_indices in cv.split(X_train):
        if count == pos_score:
            train_index.append(train_indices)
            test_index.append(test_indices)
            break
        count += 1
    # Carregando o Fechamento Ibovespa
    df_og = funcao_modelo.data_og

    # Limitando Fechamento para Base de Treino
    tamanho_train = int(len(df_og) * 0.8)
    df_close = df_og.iloc[:tamanho_train]
    df_close_teste = df_og.iloc[tamanho_train:]
    # Pegando valores do IBovespa
    df_close = df_close['Adj Close']
    fechamento_ibovespa = df_close.values

    # # Indices 
    # Indice Completo
    indices_completos = list(range(len(df_close)))
    # Treino
    indices_treino = train_index[0]
    # Teste
    indices_teste = test_index[0]
    # Encontre os índices de overlap
    indices_overlap = [indice for indice in indices_completos if indice not in indices_treino and indice not in indices_teste]

    # Crie um DataFrame do Pandas com os dados
    df = pd.DataFrame({
        'Dia': list(df_close.index),
        'Fechamento': fechamento_ibovespa,
        'Tipo': 'Treino'
    })

    df.loc[indices_teste, 'Tipo'] = 'Teste'
    df.loc[indices_overlap, 'Tipo'] = 'Overlap'

    # Função Para Gerar Base de Dados do Gráfico
    def get_data_chart(df,tipo):
        df_treino = pd.DataFrame({'Dia':[],
                                'Fechamento':[]})
        for i in range(len(df)):
            if df.iloc[i,2] == tipo:
                df_treino.loc[i,'Fechamento'] = df.iloc[i,1]
            else:
                df_treino.loc[i,'Fechamento'] = None
            df_treino.loc[i,'Dia'] = df.iloc[i,0]
        return df_treino
    df_treino = get_data_chart(df,'Treino')
    df_teste = get_data_chart(df,'Teste')
    df_overlap = get_data_chart(df,'Overlap')

    # Gerando Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_treino['Dia'],
                            y= df_treino['Fechamento'],
                            mode='lines',
                            line={'color':'blue'},
                            name='Treino'
                            ))
    fig.add_trace(go.Scatter(x=df_teste['Dia'],
                            y= df_teste['Fechamento'],
                            mode='lines',
                            line={'color':'red'},
                            name='Teste'
                            ))
    fig.add_trace(go.Scatter(x=df_overlap['Dia'],
                            y= df_overlap['Fechamento'],
                            mode='lines',
                            line={'color':'black'},
                            name='Overlap'
                            ))
    fig.add_trace(go.Scatter(x=df_close_teste.index,
                            y= df_close_teste['Adj Close'],
                            mode='lines',
                            line={'color':'yellow'},
                            name='Dados Nunca Vistos'
                            ))
    fig.update_layout(
        title=f'Roc_Auc:{round(score,3)}',
        yaxis_title = 'Fechamento',
        xaxis_title = 'Data'
    )
    fig.show()