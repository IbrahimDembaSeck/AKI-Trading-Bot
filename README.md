# AKI-Trading-Bot

TDieses Projekt wurde im Rahmen des Moduls "Anwendungen künstlicher Intelligenz" (AKI) erstellt. Es handelt sich um ein Deep-Learning-Projekt, das darauf abzielt, den zukünftigen Preis von Bitcoin (BTC) vorherzusagen.

Die Aufgabe in diesem Projekt besteht darin, zu bestimmen, ob es profitabel wäre, Bitcoin an einem bestimmten Tag zu kaufen, und nach 30 Tagen zu überprüfen, ob basierend auf der Vorhersage ein Gewinn erzielt werden konnte.

In diesem Projekt ist das Kaufdatum auf den 1. Oktober 2024 festgelegt, und das Vorhersagedatum ist der 1. November 2024. Der Trainingszeitraum erstreckt sich vom 1. Oktober 2010 bis zum 30. September 2024.

Jedes Experiment verfügt über eine eigene Readme.md-Datei, in der das Experiment im Detail erklärt wird.

Dieses Projekt umfasst insgesamt fünf Experimente, wobei das Modell in allen Experimenten gleich bleibt.

**Zusammenfassung und Eigenfeedback nach den Experimenten**

Nach Abschluss der fünf Experimente lässt sich feststellen, dass in allen Durchläufen ein starkes Overfitting aufgetreten ist. Dennoch konnten einige wertvolle Erkenntnisse gewonnen werden. Betrachtet man den Test Loss in allen Experimenten (dokumentiert in den jeweiligen Logdateien), zeigt sich, dass das vierte Experiment die besten Test-Loss-Werte erzielt hat. Neben dem Test Loss hatte das vierte Experiment auch die insgesamt überzeugendsten Ergebnisse, wie in den Logdateien festgehalten wurde.

Das Projekt bietet noch erhebliches Entwicklungspotenzial, insbesondere im Hinblick auf die Lösung des Overfitting-Problems, um robustere Ergebnisse zu erzielen. Aufgrund der begrenzten Zeit war es jedoch nicht möglich, diese Problematik im aktuellen Rahmen zu beheben. Wir schlagen daher folgende Lösungsansätze vor:

---

**1. Verbesserte Datenverarbeitung**
Die Datenverarbeitung könnte eine der Hauptursachen für das beobachtete Overfitting sein. Insbesondere das Kombinieren verschiedener Indikatoren in einer einzigen Eingabematrix könnte zu einem erhöhten Rauschen geführt haben, was die Trainingsergebnisse negativ beeinflusst. 

**Vorschlag:**  
Die Indikatoren bzw. Features sollten in einzelne Gruppen aufgeteilt und separat in das Modell eingespeist werden. Unsere Hypothese ist, dass die derzeitige Kombination der Features die Modellleistung stark beeinträchtigt. Diese Anpassung sollte in allen Experimenten umgesetzt werden.

---

**2. Vereinfachung und schrittweise Optimierung des Modells**
Das aktuelle Modell ist grundsätzlich geeignet, jedoch wäre es sinnvoll, nach der Anpassung der Datenverarbeitung zunächst mit einem einfacheren LSTM-Modell zu beginnen. 

**Vorschlag:**  
Nach jedem erfolgreichen Experiment kann die Komplexität des Modells schrittweise erhöht werden. Auf diese Weise lässt sich eine bessere Abstimmung erzielen, und unnötige Fehlerquellen können frühzeitig minimiert werden. Dieses iterative Vorgehen fördert zudem den systematischen Ergebnisfortschritt.

---

**3. Optimierung des Trainingsskripts**
Das Trainingsskript ist funktional und erfüllt seinen Zweck. Dennoch besteht Potenzial für Verbesserungen, um tiefere Einblicke in die Trainingsdynamik zu gewinnen.

**Optionale Maßnahmen:**  
Erweiterungen könnten vorgenommen werden, um zusätzliche Analysen und detailliertere Einblicke in den Trainingsprozess zu ermöglichen. 

---

**4. Verbesserungen des Experimentskripts**
Das Experimentskript ist gut strukturiert und bietet eine umfassende Darstellung der Ergebnisse, einschließlich technischer Details. 

**Vorschlag:**  
Ergänzend könnten zusätzliche Visualisierungen erstellt werden, um noch mehr technische Details grafisch aufzubereiten und die Ergebnisse weiter zu verdeutlichen.

---

**Fazit:**  
Die bisherigen Experimente haben wertvolle Erkenntnisse geliefert, auch wenn Overfitting ein dominierendes Problem darstellt. Mit den vorgeschlagenen Maßnahmen, insbesondere der Anpassung der Datenverarbeitung und einer schrittweisen Modelloptimierung, lässt sich das Projekt gezielt weiterentwickeln, um solidere und verlässlichere Ergebnisse zu erzielen.



Credits: Ibrahim Demba Seck and Atakan Yalcin
