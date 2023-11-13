import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
from sklearn.preprocessing import StandardScaler
import pickle
class Dados():
    def __init__(self) -> None:
        pass
    def Carregar_dados(self,ativo_name='^BVSP',start='2002-01-01',end='2022-01-01'):
        df = yf.download(ativo_name,start,end)
        df['Target'] = np.where(df['Adj Close'].shift(-1).pct_change() > 0, 1, 0)
        self.data = df
        self.data_og = df
    def Variaveis(self,categoria='binario'):
        df = self.data
        high = df['High']
        low = df['Low']
        close = df['Adj Close']
        volume = df['Volume']
    
    
        df['CCI'] = ta.CCI(high,low,close,timeperiod=14) #Patel
        
        df['RSI'] = ta.RSI(close,timeperiod=14) #Patel
        
        #Stochastic KD:
        numerator = df['Adj Close'] - df['Low'].rolling(25).min()
        denominator = df['High'].rolling(25).max() - df['Low'].rolling(25).min()
        df['%K'] =  (numerator/denominator) * 100 #Patel
        
        df['%D'] = df['%K'].rolling(window=3).mean() #Patel
        
        #Larry WIlliam's R%
        numerator = df['High'].rolling(window=25).max() - df['Adj Close']
        denominator = df['High'].rolling(window=25).max() - df['Low'].rolling(window=25).min()
        df['%R'] = (numerator/denominator) * 100 #Patel
        
        #MACD
        df['MACD'] = df['Adj Close'].ewm(span=12).mean() - df['Adj Close'].ewm(span=26).mean() # Patel
        
        #ADOSC - Chaikin A/D Oscillator
        df['A/D'] = ta.ADOSC(high,low,close,volume) #Patel

        #Momento
        df['Momento'] = df['Adj Close'] - df['Adj Close'].shift(9)
        df = df.dropna() 
        
        if categoria == 'binario':
            df['CCI_bin'] = np.where(
                np.logical_or(df['CCI'] <= -100,np.logical_and(df['CCI'] > df['CCI'].shift(),df['CCI'] <= 100))
                ,1
                ,0
                )
            df['RSI_bin'] = np.where(
            np.logical_or(df['RSI'] <= 30,np.logical_and(df['RSI'] > df['RSI'].shift(),df['RSI'] <= 70))
            ,1
            ,0
            ) 
            df['%K_bin'] = np.where(df['%K'] > df['%K'].shift(),1,0)
            df['%D_bin'] = np.where(df['%D'] > df['%D'].shift(),1,0)
            df['%R_bin'] = np.where(df['%R'] > df['%R'].shift(),1,0)
            df['MACD_bin'] = np.where(df['MACD'] > 0 , 1,0)
            df['A/D_bin'] = np.where(df['A/D'] > df['A/D'].shift(),1,0)
            df['Momento_bin'] = np.where(df['Momento'] > 0 , 1, 0)
            colunas_bin = [x for x in list(df.columns) if '_bin' in x ]
            self.data = df[['Target'] + colunas_bin]
        elif categoria == 'continua':
            self.data = df[df.columns[6:]]  
        self.categoria = categoria
    
    def train_test_split(self,tamanho_treinamento = 0.8):
        df = self.data
        tamanho_train = int(len(df) * tamanho_treinamento)
        tipo = StandardScaler()
        # X
        X_train = df.iloc[:tamanho_train,1:]
        X_test = df.iloc[tamanho_train:,1:]

        # # Transformando
        if self.categoria == 'continua':
            X_train = tipo.fit_transform(X_train)
            X_test = tipo.transform(X_test)

        #  Y
        y_train = df.iloc[:tamanho_train,0]
        y_test = df.iloc[tamanho_train:,0]
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    def run_model_gridsearch(self,modelo,nome_modelo):
        modelo.fit(self.X_train,self.y_train)
        results = modelo.cv_results_['mean_test_score']
        params = modelo.cv_results_['params']
        df_result = pd.DataFrame({'params':params,
                                'mean roc auc':results})
        # Salvando Resultados
        df_result.to_csv(f'./resultados/{nome_modelo}.csv')
        # Salvando Modelo
        with open(f'./Modelos/{nome_modelo}.pkl','wb') as file:
            pickle.dump(modelo,file)
        self.modelo = modelo
        return 