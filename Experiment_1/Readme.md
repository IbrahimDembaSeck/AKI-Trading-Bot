**Experiment 1: Dieses Experiment umfasst:**

### **Datenbeschaffung:**  
**Datei:** `data_preprocessing_1.py`  
Dieses Skript ist darauf ausgelegt, Zeitreihendaten für LSTM-Modelle vorzubereiten. Die Schritte umfassen:  

1. **Datenbeschaffung:**  
   - Historische Finanzdaten (z. B. Bitcoin, Gold) werden über Yahoo Finance heruntergeladen.

2. **Feature Engineering:**  
   - Kombination von Lookback-Features und Marktindikatoren.

3. **Datenverarbeitung:**  
   - Skalierung der Daten mithilfe eines `MinMaxScaler`.
   - Erstellung von Sequenzen und Zielwerten für das Modelltraining.

4. **Datenspeicherung:**  
   - Die verarbeiteten Daten (`X_train`, `y_train`, `X_test`, `y_test`) und der Scaler werden als `.pkl`-Dateien gespeichert.

5. **Ausgabe:**  
   - Vollständig vorbereitete Trainings- und Testdaten für die Modellentwicklung.  

### **Features:**

| **Feature**     | **Beschreibung**                          |
|------------------|------------------------------------------|
| `Close(t-1)` bis `Close(t-7)` | Historische Bitcoin-Preise (Lookback-Periode) |
| `Gold`           | Schlusskurs von Gold                     |
| `Silver`         | Schlusskurs von Silber                   |
| `Oil`            | Schlusskurs von Rohöl                   |
| `Gas`            | Schlusskurs von Erdgas                   |
| `FedFunds`       | Zinssatz der US-Notenbank (Fed)          |

Diese Features kombinieren spezifische historische Bitcoin-Daten mit wichtigen makroökonomischen Indikatoren und bilden die Grundlage für die Vorhersagen in diesem Experiment.  

### **Ziel:**  
Das Ziel dieses Experiments ist es, den Bitcoin-Preis für den 1. November 2024 vorherzusagen. Die Ergebnisse werden verwendet, um zu bewerten, wie genau der vorhergesagte Preis im Vergleich zum tatsächlichen Preis ist.  

Diese Evaluierung dient als Benchmark, um die Modellleistung zu bewerten. Basierend auf der Abweichung zwischen dem vorhergesagten und dem tatsächlichen Preis werden in den nachfolgenden Experimenten zusätzliche Features integriert, um die Vorhersagegenauigkeit versuchen zu verbessern.  

---

### **Modellarchitektur:**  
**Datei:** `model.py`  
Dieses Skript enthält die Implementierung eines Dual-Attention-LSTM-Modells für Zeitreihenvorhersagen. Es kombiniert die Stärken von LSTM-Schichten mit Aufmerksamkeitsmechanismen, um relevante Informationen aus sequentiellen Daten zu extrahieren.  

#### Hauptkomponenten:

1. **`DualAttentionLSTM` (Klasse):**  
   - Ein mehrschichtiges LSTM-Modell mit:
     - Zwei LSTM-Schichten.
     - Zwei Aufmerksamkeitsmechanismen zur Gewichtung signifikanter Teile der Sequenz.
     - Residual-Verbindungen und Layer-Normalisierung zur Stabilisierung des Trainings.
     - Dropout-Schichten zur Vermeidung von Overfitting.  
   - **Eingabe:** Zeitreihendaten mit der Form `(batch_size, sequence_length, input_size)`.
   - **Ausgabe:** Ein skaliertes Vorhersageergebnis pro Batch.

2. **`ensure_correct_dimensions` (Hilfsfunktion):**  
   - Stellt sicher, dass die Eingabedaten den erwarteten Dimensionen entsprechen und passt sie bei Bedarf an.

---

### **Leistungskriterien :**  

| **Kriterium**      | **Beschreibung**                                     | **Ziel**                                   |
|--------------------|------------------------------------------------------|-------------------------------------------|
| **RMSE**           | Root Mean Square Error: Durchschnitt der quadrierten Abweichungen der Vorhersagen. | Bewertung der Vorhersagegenauigkeit.      |
| **RAE**            | Relative Absolute Error: Verhältnis der Abweichung zum Durchschnitt der Vorhersage. | Vergleich mit einfachen Prognosemethoden. |
| **Profit/Verlust** |  Abweichung der Vorhersage vom tatsächlichen Wert.   | Bewertung der Investitionsstrategie.      |
| **Preisdifferenz** | Differenz zwischen vorhergesagtem Preis und Einstiegspreis. | Bewertung der Vorhersagegenauigkeit.      |
| **Grafische Darstellung** | Visualisiert tatsächliche und vorhergesagte Werte.   | Erkennung von Trends und Mustern.         |

Diese Kriterien werden verwendet, um die Modellleistung sowohl in Bezug auf die Vorhersagegenauigkeit als auch auf die potenzielle Anwendbarkeit für Investitionsstrategien zu bewerten.


**Baseline**

Hier wird als Basline der historische Durchnitt vom Bitcoin Preis ermittelt und als Baseline gesetzt.

**Ergebnisse**

Die Ergebnisse vom Training und vom Experiment sind bitte von den jeweiligen Log-Dateien zu entnehmen.

Man erhofft im nächsten Experiment eine klare Verbesserung zu sehen
