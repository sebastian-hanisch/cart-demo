# CART – der Entscheidungsbaum – Streamlit-Demo

Erstes Stück der **Baumbasierten Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning" und die **erste Demo des Portfolios mit überwachtem Lernen** (bisher: Kombinatorik, Graphen, Clustering, Anomalieerkennung):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **CART**, den Entscheidungsbaum (Breiman, Friedman, Olshen, Stone 1984) – an einem wachsenden Beispiel.
Vehikel: **Lieferungen** mit acht echten Merkmalen (Distanz, Ladegewicht, Stopps, Verkehr, Wetter, Wochentag, Zeitfenster-Enge, Fahrerjahre) und einstellbar vielen **Rauschmerkmalen**; Klassifikation ("kommt die Lieferung zu spät?") und Regression ("wie lange dauert sie?") auf denselben Merkmalen.
Alle Daten sind erzeugt, alle Zahlen gemessen und in `tests/test_claims.py` festgehalten – keine echten Daten. Der Baum ist von Grund auf in numpy geschrieben; scikit-learn kommt nur in den Tests als Gegenprobe vor.

**Bezug zu OR:** die vorhergesagte Lieferdauer ist Eingabe der Planung (Zeitfenster, Touren mit Zeitfenstern); ein Baum liefert dazu Regeln, die man Disponenten zeigen kann. Lernen und Optimieren ergänzen sich – diese Demo zeigt den Lernteil für sich.

**Einordnung in die Reihe (die Kanten des Graphen):** CART ist die **Wurzel** der Linie. Er wählt Schnitte **gierig** und baut den Baum in einem Zug; daraus folgen die beiden Schwächen, die die übrigen Stücke antreiben – **Überanpassung** (ein weitergewachsener Baum lernt das Rauschen) und **Instabilität** (andere Stichprobe, anderer Baum).

```
CART  (dieses Stück)
 ├─ Bagging → Random Forest → Extra Trees        (Varianz senken)
 └─ AdaBoost → Gradient Boosting → { XGBoost, LightGBM, CatBoost }   (Fehler nacheinander korrigieren)
```

| Frage | Ergebnis (1200 Lieferungen, 3 Rauschmerkmale, 70 % Training / 30 % Test, Seed 7; Klassifikation "zu spät") |
|---|---|
| Flacher Baum (Tiefe 3, Blatt ≥ 5) | ✅ 8 Blätter, Trainingsfehler **15.8 %**, Testfehler **17.2 %** (Raten: 46.4 %), AUC 0.882, Log-Loss 0.42. Der erste Schnitt ist das **Ladegewicht** (798 kg), danach Verkehr. |
| Voll gewachsen (Tiefe 16, Blatt ≥ 1) | ❌ 100 Blätter, Trainingsfehler **0 %**, Testfehler **16.7 %** – kaum anders als der flache Baum. **Die Genauigkeit täuscht:** AUC **0.830** statt 0.882, Log-Loss **5.76** statt 0.42 (Blätter mit 0 % oder 100 %, die sich sicher irren). |
| Beschneiden (Kosten-Komplexität, alpha per Kreuzvalidierung **nur auf dem Training**) | ✅ 100 → **17 Blätter** (alpha 0.004), Testfehler 16.4 %, AUC 0.890, Log-Loss 0.58. Auf fünf weiteren Datensätzen ist der so beschnittene Baum im Mittel um mehr als 2 Punkte besser als der volle (siehe Tests); die Testkurve über den Pfad hat ihr Minimum in der Mitte (34 Blätter, 13.9 % – nur als Diagnose, gewählt wird nie mit den Testdaten). |
| Falsche Etiketten im Training | ❌ Testfehler des vollen Baums bei 0 / 10 / 20 % vertauschten Etiketten (Mittel über sechs Datensätze): **20.4 / 25.4 / 35.0 %**; ein flacher Baum (Tiefe 4, Blatt ≥ 5) **17.1 / 17.1 / 21.8 %**. Der Test bleibt sauber. |
| Rauschmerkmale | ⚠️ Der volle Baum benutzt sie: **16 %** seiner Schnitte (3 Rauschmerkmale) bzw. **29 %** (8) liegen auf Rauschen – fast alle tief im Baum (in den ersten vier Ebenen zusammen nur 5 bzw. 8 Schnitte über sechs Datensätze). |
| **Instabilität** (30 Bäume auf Bootstrap-Stichproben bzw. auf 95 % der Trainingszeilen) | ❌ Ein voll gewachsener Baum ändert die Vorhersage auf mehr als **16 %** der Testlieferungen (Bootstrap) bzw. mehr als **8 %** (5 % der Zeilen weg); ein Baum der Tiefe 3 ist in allen sechs Datensätzen stabiler, wechselt aber in den oberen zwei Ebenen den Aufbau. Die **Wurzel** ist stabil: alle 30 Bäume beginnen mit demselben Merkmal (Klassifikation und Regression). |
| Gini gegen Entropie | ➖ Sie sind fast austauschbar: in **2 von 6** Datensätzen dieselben Merkmale in den ersten sieben Schnitten, die Vorhersagen (Tiefe 4, Blatt ≥ 5) stimmen zu **92 bis 98 %** überein. |
| Regression (Tiefe 4, Blatt ≥ 5) | ✅ 16 Blätter, Trainings-RMSE 12.3 min, Test-RMSE **14.4 min** gegen 27.2 min beim Mittelwert (R² 0.719); wichtigster Schnitt Distanz (65.5 %). Der Baum liefert eine **Treppe**, keine Gerade – eine lineare Steigung braucht viele Blätter. |

## Was die Demo zeigt

- **CART in Aktion:** der Baum wächst Schnitt für Schnitt (Ebene für Ebene, wie CART ihn baut) mit Schritt-Regler und Abspielen: Baumdiagramm (Blätter nach Wert gefärbt, oranger Rand bei Schnitten auf Rauschmerkmalen, schwarzer Ring um den nächsten Schnitt), die Vorhersage über zwei wählbare Merkmale (übrige auf dem Median), der **Weg einer Testlieferung** durch den Baum und die **Schnittsuche** des nächsten Knotens (bester Gewinn je Merkmal, Gewinn über alle Schwellen des gewählten Merkmals).
- **Was der Baum gelernt hat:** Blätter, Trainings- und Testfehler gegen das Raten, AUC und Log-Loss (bzw. R² und MAE), ein Urteil (Überanpassung / zu einfach / ausgewogen), Wichtigkeit der Merkmale, Fehler gegen Tiefe und der **Beschneidungspfad** (Blätter gegen Fehler).
- **Regler:** Aufgabe (Klassifikation | Regression), Kriterium (Gini | Entropie; bei Regression fest Varianz), Tiefe, Mindestblattgröße, Beschneiden (aus | Kreuzvalidierung | von Hand), Lieferungen, Rauschmerkmale, falsche Etiketten, Seed.
- **Experimente auf Knopfdruck:** Stabilität (Bootstrap und 5 % weniger Zeilen, drei Baumgrößen) und Beschneiden auf fünf Datensätzen (voll, flach, beschnitten).

## Modell und Verfahren

- **Schnittsuche:** je Merkmal sortieren, kumulative Summen (Klassenzahl bzw. Summe und Quadratsumme) liefern die Unreinheit beider Kinder für **alle** Schwellen in einem Zug; Kandidaten sind Mitten zwischen verschiedenen benachbarten Werten, zulässig nur mit Mindestblattgröße. Bei Gleichstand gewinnt das kleinste Merkmal, dann die kleinste Schwelle.
- **Unreinheit:** Gini 2p(1−p), Entropie (bit), Varianz; Blattwert = Anteil bzw. Mittelwert. Wachsen in **Breitenreihenfolge**, damit "der Baum nach k Schnitten" einfach der Baum ist, dessen Knoten ab dem k-ten inneren wieder Blätter sind.
- **Beschneiden:** Kosten-Komplexität (kleinstes effektives alpha zuerst, Pfad und Beschneiden wie `ccp_alpha` in scikit-learn); alpha per fünffacher **Kreuzvalidierung auf dem Training** (höchstens 40 Kandidaten), bei Gleichstand das größere.
- **Gütemaße von Hand:** Fehlerquote, AUC über die Rangsumme, Log-Loss, RMSE, MAE, R².

## Was nicht funktioniert hat / Grenzen

- **Erste Fassung der Daten:** die Klassenaussage war ein Unterschied zweier großer, fast gerader Terme plus Rauschen – kaum ein Baum kam unter etwa 17 % Fehler, und der Testfehler stieg nie mit der Tiefe. Erst eine Dauer aus **Schwellen und Stufen** (Stau ab einem Verkehrsindex, der mit der Distanz wächst; schwere Ladung verlängert jeden Stopp; Regen; Wochenende) mit Rauschen 7 min und einem Puffer für die versprochene Zeit gab ein Beispiel, an dem ein Baum lernen kann und Überanpassung sichtbar wird.
- **Der Testfehler steigt mit der Tiefe nur mäßig:** Mittel über sechs Datensätze, voller Baum (Tiefe 16, Blatt ≥ 1) gegen die – im Nachhinein am Test gewählte – beste Tiefe: Klassifikation **20.4 % gegen 15.5 %** (beste Tiefe 3 bis 6), Regression **14.6 gegen 13.3 min** (beste Tiefe 5 bis 8). Der deutliche Befund ist die **Lücke zwischen Training und Test**, der **Log-Loss** und die **Instabilität**.
- **Wurzel ist stabil, der Rest nicht:** ein dominantes Merkmal (das Ladegewicht bei der Klassifikation, die Distanz bei der Regression) macht den ersten Schnitt eindeutig; die Instabilität sitzt in den unteren Ebenen.
- **Gleichstände sind echt:** in kleinen Knoten (zwei Lieferungen, oder wenige mit großer Mindestblattgröße) sind mehrere Schnitte gleich gut; scikit-learn wählt zufällig, diese Umsetzung das kleinste Merkmal. Deshalb sind die Bäume nur dort **exakt gleich** mit scikit-learn, wo es keine Gleichstände gibt (siehe Verifikation).
- **Treppen statt Geraden, Schnitte nur an einer Achse, keine Extrapolation:** siehe die Tabelle "Wo die Annahmen enden" in der App.

## Verifikation

`tests/test_algorithm.py` (37 Tests): Schnittsuche gegen eine **Brute-Force-Suche** über alle Schwellen (alle drei Kriterien, mit Gleichständen und Mindestblattgröße); **scikit-learn** (`DecisionTreeClassifier/Regressor`) liefert bei stetigen Merkmalen und ohne Gleichstände **dieselben Bäume**, Vorhersagen und Blattwerte; die **Wichtigkeit** stimmt auf scikit-learns eigenem Baum überein; **Pfad der Kosten-Komplexität und beschnittene Bäume** stimmen mit `cost_complexity_pruning_path` und `ccp_alpha` überein (auf scikit-learns eigenen Bäumen, damit Gleichstände beim Wachsen nichts verdecken); Grenzfälle (Tiefe 0, `min_samples_leaf` = n, reine Knoten, konstantes Merkmal, voller Baum passt das Training); Breitenreihenfolge und Wachsen Schnitt für Schnitt; Gütemaße gegen `sklearn.metrics`.
`tests/test_claims.py` hält **jede Zahl** aus App und README fest. `tests/test_app.py` prüft die Oberfläche per AppTest (jedes Preset, Aufgabenwechsel, ausgeblendete Regler, Abspielen mit mehreren Bildern und schrittspezifischen Diagramm-Schlüsseln, Permalink, Experimente).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `cart_algorithm.py` | Schnittsuche, Wachsen, Vorhersage, Wichtigkeit, Beschneiden, Gütemaße |
| `cart_scenario.py` | Lieferdaten (Merkmale, Dauer, Etiketten, Aufteilung) |
| `cart_evaluation.py` | Analyse, Kreuzvalidierung, Instabilität, Experimente |
| `cart_visualization.py` | Baumdiagramm, Karte, Schnittsuche, Kurven |
| `cart_presets.py`, `cart_constants.py` | Regler, Permalink, Schnellstart-Beispiele, Grenzen |
| `tests/` | Algorithmus-, Claims- und App-Tests |

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -q
```

---

Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
