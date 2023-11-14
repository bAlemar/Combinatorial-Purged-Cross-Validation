# Combinatorial-Purged-Cross-Validation
#### Artigo: https://medium.com/@balemar/combinatorial-purged-cross-validation-48fc748d006e
Uma alternativa as diversas validações cruzadas usadas na literatura de predição do movimento de mercado.O projeto tem como objetivo disponibilizar a Combinatorial Purged CrossValidation e mostrar como ela pode ser crucial na busca pelos hiperparâmetros.


# Descrição do Projeto
Main.ipynb contém todas as funções citadas abaixo e como foi desenvolvido a comparação dos modelos. Sugiro começar lendo a Main.ipynb.
## Funções: 
### ./Functions/Data_and_Model.py  
Função de carregamento da base de dados do yfinance, criação das variáveis independentes binária e geração do modelo de GridSearch.
### ./Functions/Fold_of_CPCV.py
Função que printará a iteração da Validação Cruzada CPCV dado um rank específico de roc auc. Dessa maneira, a análise por trás da CV torna-se viável.
### ./Functions/cpcv.py
Função da Combinatorial Purged Cross Validation no sklearn.
### ./Functions/Confusion_Matrix.py
Roda relatório de classificação, score roc auc da base de teste e matriz de confusão dos modelos salvos.

# Como rodo o código no meu computador?
Para rodar o código no seu computador você pode: (i) clonar código via git; (ii) fazer download do zip. 
<p> (i) https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository. <p>
<p> Após essa etapa, você irá instalar todas as depedências(bibliotecas) necessárias digitando em seu terminal " pip install -r requirements.txt ".<p>

# Contato:
https://linktr.ee/bernardoalemar
https://www.linkedin.com/in/bernardo-alemar-9117a11a2/
