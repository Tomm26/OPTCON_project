## README.md
```markdown
# Optimal Control of a Flexible Robotic Arm

Questo progetto implementa strategie di controllo ottimo per un braccio robotico flessibile, modellato come un robot planare a due link con coppia applicata sul primo giunto.

## Struttura del Progetto
```
OPTCON_project-1/
├── src/
│   ├── parameters.py     # Parametri di sistema e costanti
│   ├── dynamics.py       # Implementazione della dinamica
│   ├── derivatives.py    # Derivate simboliche per il controllo ottimo
│   ├── visualization/
│   │   └── animate.py    # Utility per l'animazione
│   └── controllers/      # (Future implementazioni dei controllori)
├── tests/               # (Test futuri)
└── examples/           # (Esempi futuri)
```

## Requisiti
- Python 3.8+
- NumPy
- SciPy
- SymPy
- Matplotlib

## Installazione
```bash
# Crea e attiva un ambiente virtuale (opzionale ma raccomandato)
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
```

## Descrizione dei Task

### Task 0 - Setup del Problema ✓
- [x] Discretizzazione della dinamica
- [x] Implementazione equazioni stato-spazio
- [x] Implementazione funzione dinamica
- [x] Visualizzazione base del sistema

### Task 1 - Generazione Traiettoria (I)
- [ ] Calcolo equilibri del sistema
- [ ] Definizione curva di riferimento tra equilibri
- [ ] Implementazione algoritmo di Newton per controllo ottimo

### Task 2 - Generazione Traiettoria (II)
- [ ] Generazione curva stato-input smooth
- [ ] Implementazione generazione traiettoria
- [ ] Calcolo traiettoria quasi-statica

### Task 3 - Tracking tramite LQR
- [ ] Linearizzazione dinamica robot
- [ ] Implementazione algoritmo LQR
- [ ] Test con condizioni iniziali perturbate

### Task 4 - Tracking tramite MPC
- [ ] Implementazione algoritmo MPC
- [ ] Test performance tracking
- [ ] Confronto con risultati LQR

### Task 5 - Animazione
- [x] Implementazione animazione base
- [ ] Aggiunta visualizzazione traiettoria
- [ ] Aggiunta visualizzazione spazio delle fasi

## Uso
Per eseguire l'animazione base:
```bash
python src/visualization/animate.py
```
```

## requirements.txt
```
numpy>=1.21.0
scipy>=1.7.0
sympy>=1.9
matplotlib>=3.4.0
```