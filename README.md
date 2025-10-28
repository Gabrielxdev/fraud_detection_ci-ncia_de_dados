

#  Detecção de Fraude em Transações Financeiras

## 1. Objetivo

O projeto visou construir um modelo de machine learning para detectar transações financeiras fraudulentas a partir de um conjunto de dados sintéticos de mais de 6,3 milhões de registros. O foco principal era identificar padrões em transações de `CASH_OUT` e `TRANSFER`, que se mostraram os únicos tipos suscetíveis à fraude.

## 2. Análise Exploratória de Dados (EDA)

A análise inicial revelou insights cruciais que nortearam a modelagem:

* **Dataset Extremamente Desbalanceado:** Apenas 0,13% (8.213 transações) de todo o conjunto de dados eram fraudulentas. Isso tornou a acurácia uma métrica inútil e exigiu técnicas especiais de modelagem.
* **Tipos de Fraude:** A fraude ocorria **exclusivamente** em transações do tipo `TRANSFER` (maior taxa de fraude) e `CASH_OUT`. Os tipos `PAYMENT`, `CASH_IN` e `DEBIT` eram 100% legítimos.
* **Padrões de Valor (`amount`):**
    * A distribuição dos valores era fortemente assimétrica.
    * Após uma transformação logarítmica (`log1p`), a distribuição se mostrou **bimodal**, sugerindo dois grupos distintos de comportamento de transação (provavelmente legítimos vs. fraudulentos).
    * Boxplots confirmaram que transações fraudulentas tendem a ter valores medianos mais altos que as legítimas.
* **Padrões de Saldo:** Foram identificados comportamentos anômalos que são fortes indicadores de fraude, como o saldo do remetente (`oldbalanceOrg`) ser zerado após a transação ou, em casos mais estranhos, o saldo do destinatário diminuir após receber fundos.
* **Features Irrelevantes:**
    * `step` (tempo): A fraude parecia ocorrer aleatoriamente ao longo do tempo, sem sazonalidade. A coluna foi removida.
    * `nameOrig` e `nameDest`: Os fraudadores no dataset agiam apenas uma vez, tornando o ID do usuário inútil para prever reincidência. As colunas foram removidas.

## 3. Pré-processamento e Modelagem

Para preparar os dados para o modelo, foi construído um `Pipeline` robusto:

1.  **Divisão Estratificada:** Os dados foram divididos em treino e teste usando `stratify=y`, garantindo que a proporção de 0,13% de fraudes fosse mantida em ambos os conjuntos.
2.  **Transformação de Colunas (`ColumnTransformer`):**
    * **Colunas Numéricas** (`amount`, `oldbalanceOrg`, etc.): Foi aplicado o `StandardScaler` para padronizar as escalas.
    * **Coluna Categórica** (`type`): Foi aplicado o `OneHotEncoder` para transformar os tipos de transação (ex: `TRANSFER`) em formato numérico.
3.  **Modelo (Regressão Logística):**
    * Foi escolhido o `LogisticRegression` como classificador.
    * O parâmetro `class_weight="balanced"` foi utilizado. Isso foi **essencial** para instruir o modelo a dar um "peso" maior aos erros na classe minoritária (fraude), forçando-o a prestar atenção nela.
    * O `max_iter` foi aumentado para 1000 para garantir a convergência do modelo.

## 4. Resultados e Avaliação

A avaliação do modelo foi focada na precisão e no recall, pois a acurácia (95%) era enganosa.

### Avaliação 1: Limiar Padrão (0.5)

O modelo inicial, apesar de treinado com `class_weight="balanced"`, apresentou um desequilíbrio clássico:

* **Recall (Classe 1 - Fraude): 0.94 (Excelente)**
    * O modelo foi capaz de "encontrar" 94% de todas as fraudes reais.
* **Precision (Classe 1 - Fraude): 0.02 (Péssimo)**
    * Dos alertas de fraude gerados pelo modelo, 98% eram falsos positivos. Isso significa que, para pegar 2.315 fraudes, o modelo gerou mais de 103.000 alertas falsos.

**Diagnóstico:** O modelo era um "detetive" sensível que "gritava fraude" por qualquer motivo, tornando-o impraticável para uma equipe de análise.

### Avaliação 2: Ajuste de Limiar (Threshold) para 0.85

Para resolver o problema dos falsos positivos, foi feito um ajuste manual do limiar de decisão. Em vez de classificar como fraude qualquer coisa acima de 50% de probabilidade, o modelo passou a exigir 85% de certeza.

* **Recall (Classe 1 - Fraude): 0.79 (Bom)**
    * O modelo ainda encontrou 79% das fraudes reais (uma queda aceitável).
* **Precision (Classe 1 - Fraude): 0.08 (Melhoria de 4x)**
    * A precisão aumentou de 2% para 8%. O número de falsos positivos caiu drasticamente de ~103.000 para ~23.000.

**Diagnóstico:** O ajuste criou um modelo muito mais equilibrado. Embora ainda gere falsos positivos, ele reduziu o "ruído" em mais de 77%, tornando os alertas gerados muito mais confiáveis e acionáveis.

## 5. Deployment (Implantação)

1.  **Salvando o Modelo:** O `Pipeline` treinado (contendo o pré-processador e o classificador) foi salvo em um único arquivo, `fraud_detection_pipeline.pkl`, usando `joblib`.
2.  **Aplicação Web:** Um aplicativo web simples foi desenvolvido com o **Streamlit**, permitindo que um usuário insira manualmente os dados de uma transação e receba uma previsão (Fraude ou Não-Fraude) em tempo real.
