**Experiment 5:**

**Datenbeschaffung:**

Wie Experiment 4 nur:  + Feature Engineering

**Features**

Wie Experiment 4 nur: + Feature Engineering : Rollende Mittelwerte, Standard Abweichungen, Preisveränderungen, 
prozentuale Preisveränderungen, Momentum und Zeitbasiertes Feature

**Ziel**

Observieren, was das Hinzufügen der neuen Features  für eine Auswirkung auf die Vorhersage, die davor gemacht wurde hat.

**Modellarchitektur + Hauptkomponente**

Wie im Experiment 1

**Leistungskriterium und Baseline**

Wie im Experiment 1

**Ergebnisse**

Kaufpreis 2024-10-01: 60837.0078125
Tatsächlicher Wert 2024-11-01: 69482.46875
Preisunterschied: -7647.200651167717
RMSE: 7647.20
RAE: 2.02

Konvergenztabelle:
            Metric         Value
0  Predicted Value  61835.268099
1     Actual Value  69482.468750
2  Price Deviation  -7647.200651
3             RMSE   7647.200651
4              RAE      2.021376
Baseline (Durchschnitt historischer Preise): 65699.30
Baseline-Deviation (Abweichung vom tatsächlichen Wert): -3783.17

Dieses Experiment stellt sich eher als Rückschritt dar, weil die Prognosen deutlich schlechter wurden..