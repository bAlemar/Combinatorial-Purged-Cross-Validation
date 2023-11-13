import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,ConfusionMatrixDisplay, confusion_matrix, classification_report
import pickle
from Data_and_Model import Dados


class check_results():
     def __init__(self,tipo_dado):
          # Carregando Base de Teste:
          funcao_modelo = Dados()
          funcao_modelo.Carregar_dados()
          funcao_modelo.Variaveis(categoria=tipo_dado)
          funcao_modelo.train_test_split()
          X_train = funcao_modelo.X_train
          y_train = funcao_modelo.y_train
          X_test = funcao_modelo.X_test
          y_test = funcao_modelo.y_test
          self.X_train = X_train
          self.X_test = X_test
          self.y_train = y_train
          self.y_test = y_test
     def matriz_confusao_geral(self,arquivo_modelo,tri=False):
          # Carregando Modelo:
          modelo = pickle.load(open(arquivo_modelo,'rb'))
          y_pred = modelo.predict(self.X_test)
          score_test = roc_auc_score(self.y_test,y_pred)
          cm = confusion_matrix(self.y_test,y_pred)
          labels = ['Baixa','Alta']
          cm_display = ConfusionMatrixDisplay(cm,display_labels=labels)
          cm_display.plot()
          cr = classification_report(self.y_test,y_pred,target_names=labels)
          print('roc_auc',round(roc_auc_score(self.y_test,y_pred),3))
          print('acuracy',round(accuracy_score(self.y_test,y_pred),3))
          print('Parametros',modelo.best_params_)
          print(cr)
          self.modelo = modelo
          return cr,cm_display