---

## **📌 Supervised vs. Unsupervised Learning**
### **Supervised Learning (Apprendimento Supervisionato)**
- I dati sono etichettati (ogni esempio ha un'uscita desiderata).
- Obiettivo: imparare una funzione che mappa input → output.
- Esempi di algoritmi:
  - **Regressione** (predizione di valori continui, es. prezzi di case)
  - **Classificazione** (es. riconoscimento di immagini, spam detection)
  - **Support Vector Machines (SVM)**
  - **Alberi decisionali & Random Forest**
  - **Neural Networks (MLP, CNN, LSTMs per sequenze)**

### **Unsupervised Learning (Apprendimento Non Supervisionato)**
- I dati **non** sono etichettati (non abbiamo un output esplicito).
- Obiettivo: trovare pattern nascosti nei dati.
- Esempi di algoritmi:
  - **Clustering** (K-Means, DBSCAN, Agglomerative)
  - **Dimensionality Reduction** (PCA, t-SNE)
  - **Anomaly Detection** (modelli che trovano anomalie nei dati)

---
## **📌 Clustering**
**Cos'è?**  
Una tecnica di Unsupervised Learning per raggruppare dati simili tra loro.

### **Algoritmi di Clustering**
1. **K-Means**
   - Divide i dati in **K gruppi** basati sulla distanza euclidea dai centri.
   - Necessita di scegliere il numero di cluster **K** a priori.
   - Vantaggi: semplice e veloce.
   - Svantaggi: sensibile agli outlier.

2. **DBSCAN**
   - Basato sulla densità: trova cluster di punti densi separati da regioni meno dense.
   - **Non richiede di specificare il numero di cluster**.
   - Ottimo per dati con forma irregolare.

3. **Clustering Gerarchico (Agglomerative)**
   - Costruisce una gerarchia di cluster (albero dendrogramma).
   - Non richiede un numero fisso di cluster.

### **Metriche di valutazione per clustering**
- **Silhouette Score**: misura quanto i punti siano ben assegnati ai cluster.
- **Davies-Bouldin Index**: misura la compattezza dei cluster.

---
## **📌 NLP (Natural Language Processing)**
### **Tecniche di Preprocessing**
1. **Tokenization** → separa il testo in parole o frasi.
2. **Stemming & Lemmatization** → riduce le parole alla radice.
3. **Stopwords Removal** → rimuove parole comuni ("il", "e", "di", etc.).
4. **TF-IDF** → rappresentazione dei testi basata sull’importanza delle parole.
5. **Word Embeddings**:
   - **Word2Vec** (CBOW e Skip-gram)
   - **FastText** (simile a Word2Vec ma considera sottoparole)
   - **BERT & GPT** (modelli basati su Transformer)

---
## **📌 Modelli di Machine Learning**
### **Modelli di base**
- **Regressione Lineare**: per predire valori numerici.
- **Regressione Logistica**: per problemi di classificazione binaria.

### **Modelli avanzati**
- **Random Forest**: combinazione di alberi decisionali per migliorare le performance.
- **Gradient Boosting (XGBoost, LightGBM, CatBoost)**: potenziamento dei modelli tramite boosting.
- **Reti Neurali (Deep Learning)**:
  - CNN (per immagini)
  - RNN/LSTM (per sequenze)
  - Transformer (per NLP)

---
## **📌 Feature Extraction**
### **Feature Selection vs Feature Extraction**
- **Feature Selection**: selezionare le caratteristiche più rilevanti eliminando le irrilevanti (LASSO, Random Forest Feature Importance).
- **Feature Extraction**: trasformare i dati in un nuovo spazio (PCA, Autoencoder).

### **Tecniche comuni**
- **PCA (Principal Component Analysis)**: riduce la dimensionalità mantenendo la varianza.
- **LDA (Linear Discriminant Analysis)**: riduce la dimensionalità massimizzando la separazione tra classi.

---