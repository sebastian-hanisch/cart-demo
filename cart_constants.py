"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

# Merkmale der Lieferungen: (Name, Einheit)
FEATURES = [("Distanz", "km"), ("Ladegewicht", "kg"), ("Stopps", ""), ("Verkehr", "0-1"), ("Wetter", "0-1"), ("Wochentag", "0 = Mo"), ("Zeitfenster-Enge", "0-1"), ("Fahrerjahre", "Jahre")]
N_BASE = len(FEATURES)

TASKS = ("class", "reg")
TASK_LABELS = {"class": "Klassifikation: kommt die Lieferung zu spät?", "reg": "Regression: wie lange dauert die Lieferung?"}
DEFAULT_TASK = "class"
CRITERIA = {"class": ("gini", "entropy"), "reg": ("variance",)}
CRITERION_LABELS = {"gini": "Gini-Unreinheit", "entropy": "Entropie", "variance": "Varianz (Fehlerquadrate)"}
DEFAULT_CRITERION = {"class": "gini", "reg": "variance"}

N_MIN, N_MAX, DEFAULT_N = 400, 3000, 1200
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0, 8, 3               # Rauschmerkmale (zufällig, ohne Bezug zum Ziel)
LABEL_NOISE_MIN, LABEL_NOISE_MAX, DEFAULT_LABEL_NOISE = 0, 20, 0     # Prozent falsche Etiketten (nur Klassifikation)
DEPTH_MIN, DEPTH_MAX, DEFAULT_DEPTH = 1, 16, 4
LEAF_MIN, LEAF_MAX, DEFAULT_LEAF = 1, 50, 5
TEST_SHARE = 0.3
DEFAULT_SEED = 7
DEFAULT_TASK_SHOWN = "class"

SWEEP_SEEDS = tuple(range(100000, 100005))
BOOTSTRAPS = 30
OVERFIT_GAP_CLASS = 0.08      # Testfehler minus Trainingsfehler (Klassifikation) ab hier: Überanpassung
OVERFIT_RATIO_REG = 1.6        # Testfehler / Trainingsfehler (Regression) ab hier: Überanpassung
UNDERFIT_SHARE = 0.75          # Testfehler über diesem Anteil des Rate-Fehlers: der Baum ist zu einfach
CV_GRID = 40                 # höchstens so viele alpha-Kandidaten in der Kreuzvalidierung

COLORS = {"train": "#1f77b4", "test": "#d62728", "late": "#d62728", "ontime": "#2ca02c", "split": "#111111", "noise": "#ff7f0e"}

PRUNE_MODES = ("off", "cv", "manual")
PRUNE_LABELS = {"off": "kein Beschneiden", "cv": "alpha per Kreuzvalidierung", "manual": "von Hand (Blätterzahl)"}
DEFAULT_PRUNE = "off"
DEFAULT_PRUNE_LEAVES = 12
DEFAULT_MAP = (0, 3)          # Kartenausschnitt: Distanz x Verkehr
DEFAULT_LABEL_NOISE_SHOWN = 0

PRESETS = {
    "🌳 Flacher Baum": dict(task="class", criterion="gini", depth=3, leaf=5, prune="off", prune_leaves=DEFAULT_PRUNE_LEAVES, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🌲 Voll gewachsen": dict(task="class", criterion="gini", depth=DEPTH_MAX, leaf=1, prune="off", prune_leaves=DEFAULT_PRUNE_LEAVES, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "✂️ Beschneiden": dict(task="class", criterion="gini", depth=DEPTH_MAX, leaf=1, prune="cv", prune_leaves=DEFAULT_PRUNE_LEAVES, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🏷️ Falsche Etiketten": dict(task="class", criterion="gini", depth=DEPTH_MAX, leaf=1, prune="off", prune_leaves=DEFAULT_PRUNE_LEAVES, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=10, seed=DEFAULT_SEED, fx=0, fy=3),
    "📈 Regression": dict(task="reg", criterion="variance", depth=4, leaf=5, prune="off", prune_leaves=DEFAULT_PRUNE_LEAVES, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
}
PRESET_HELP = {
    "🌳 Flacher Baum": "Klassifikation, Tiefe 3, Blätter mit mindestens 5 Lieferungen: 8 Blätter, Trainingsfehler 15.8 %, Testfehler 17.2 % (Raten: 46.4 %). Der erste Schnitt ist das Ladegewicht (ab 798 kg wird es eng), danach Verkehr und Distanz - Training und Test liegen dicht beieinander.",
    "🌲 Voll gewachsen": "Tiefe 16, Blätter ab einer Lieferung: der Baum wächst, bis jedes Blatt rein ist - 100 Blätter, Trainingsfehler 0 %, Testfehler 16.7 %, kaum anders als beim flachen Baum. Die Genauigkeit täuscht: AUC 0.830 statt 0.882 und Log-Loss 5.76 statt 0.42 - die Blätter sind überzuversichtlich. Überanpassung.",
    "✂️ Beschneiden": "Derselbe volle Baum, mit Kosten-Komplexität auf 17 Blätter beschnitten (alpha 0.004, gewählt per Kreuzvalidierung auf dem Training): Trainingsfehler 10.4 %, Testfehler 16.4 %, AUC 0.890, Log-Loss 0.58 - der Testfehler ist nicht schlechter als beim vollen Baum, der Log-Loss viel besser.",
    "🏷️ Falsche Etiketten": "10 % der Trainingsetiketten sind vertauscht, der Baum wächst voll (125 Blätter, Trainingsfehler 0 %): er lernt die falschen Etiketten mit, der Testfehler steigt auf 23.6 % (ohne falsche Etiketten 16.7 %). Ein flacher Baum wäre kaum betroffen.",
    "📈 Regression": "Regression: Dauer in Minuten, Tiefe 4, Blätter ab 5 Lieferungen: 16 Blätter, Trainingsfehler (RMSE) 12.3 min, Testfehler 14.4 min gegen 27.2 min beim Raten des Mittelwerts (R² 0.719). Der wichtigste Schnitt ist die Distanz (65.5 % der Wichtigkeit); der Baum liefert eine Treppe, keine Gerade.",
}
