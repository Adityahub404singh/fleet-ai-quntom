# 🚢 QuantumFleet AI

**AI + Optimization Decision-Support System for Green Fleet Management**
(SIH Prototype — Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization)

QuantumFleet AI helps a fleet manager decide **which vessel, at what speed,
using which fuel**, to move cargo on schedule while minimizing fuel use,
cost, and GHG emissions.

```
Vessels + Cargo + Voyage + Speed + Fuel + Weather
                    │
                    ▼
         🤖 AI Fuel Prediction (XGBoost)
                    │
                    ▼
        ⛽ Cost + 🌱 GHG Emissions
                    │
                    ▼
      🧠 Optimization Engine
      Classical baseline  vs  Quantum-Inspired (simulated annealing / QUBO-style)
                    │
                    ▼
         ⭐ Best Feasible Recommended Plan
                    │
                    ▼
              📊 Dashboard
```

## ⚠️ Important, honest disclaimers (say this to judges up front)

- **Data is simulated**, generated from a physics-inspired formula (fuel ∝ speed³,
  the standard cubic law of ship propulsion) plus load/weather/vessel-profile
  effects and noise. It is clearly labelled as synthetic, not live telemetry.
- **"Quantum-inspired" does not mean real quantum hardware.** The optimizer runs
  on a normal CPU. It formulates the fleet decision (which speed, which fuel)
  as a QUBO-style binary/discrete search and solves it with **simulated
  annealing** — a well-known classical technique that mimics quantum annealing
  behaviour. We benchmark it against a classical greedy baseline rather than
  just asserting it's better.

## 🏗️ Project structure

```
quantumfleet-ai/
├── backend/
│   ├── main.py                     # FastAPI app — all API endpoints
│   ├── database.py                 # SQLAlchemy models (SQLite for demo)
│   ├── config.py                   # Fuel price / emission / lifecycle factors
│   ├── seed_data.py                # Populates sample vessels & voyages
│   ├── ml/
│   │   ├── generate_data.py        # Builds synthetic training dataset
│   │   ├── train_model.py          # Trains XGBoost fuel-prediction model
│   │   └── predict.py              # Loads model, predicts + confidence band
│   ├── optimization/
│   │   ├── classical.py            # Greedy baseline optimizer
│   │   └── quantum_inspired.py     # Simulated-annealing / QUBO-style optimizer
│   ├── data/                       # Generated CSV + trained model + SQLite DB
│   └── requirements.txt
└── frontend/
    └── index.html                  # Single-page dashboard (no build step)
```

## ▶️ Run it (today, for the demo)

### 1. Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python3 ml/generate_data.py     # build synthetic dataset  (~2s)
python3 ml/train_model.py       # train the XGBoost model  (~5s)
python3 seed_data.py            # seed sample vessels/voyages

uvicorn main:app --reload --port 8000
```

Backend is now live at `http://127.0.0.1:8000` (interactive API docs at `/docs`).

### 2. Frontend

No build step needed — just open the file:

```bash
cd frontend
# double-click index.html, or serve it:
python3 -m http.server 5500
```

Then open `http://127.0.0.1:5500` (or double-click `index.html` directly).
The dashboard talks to the backend at `http://127.0.0.1:8000`.

## 🖥️ API endpoints

| Method | Endpoint              | Purpose                                              |
|--------|-----------------------|-------------------------------------------------------|
| GET    | `/api/vessels`         | List fleet vessels                                    |
| POST   | `/api/vessels`         | Add a vessel                                           |
| GET    | `/api/voyages`         | List voyages                                           |
| POST   | `/api/predict-fuel`    | AI fuel prediction + confidence band + explainability  |
| POST   | `/api/optimize`        | Current vs Classical vs Quantum-Inspired comparison    |
| POST   | `/api/scenario`        | What-if simulator (fuel price / cargo / speed changes) |
| GET    | `/api/explain`         | Global feature-importance breakdown                    |

## 🎤 Demo script (for judges)

1. **Overview** tab — show the fleet (4 vessels, mixed status).
2. **Predict Fuel** tab — pick Vessel A, 14 knots, 4000t cargo → click
   **Predict Fuel**. Point out the confidence range and the "Why this
   prediction?" explainability bars (speed dominates — cubic law).
3. **Optimize** tab — set current speed 18 knots, deadline 3 days → click
   **Optimize Fleet**. Show the **Current vs Classical vs Quantum-Inspired**
   table and the recommended plan with reasons.
4. **Scenario Simulator** tab — bump fuel price +20% → **Recalculate**, show
   cost impact instantly.
5. Close with the disclaimer above: simulated data, CPU-based quantum-inspired
   heuristic, benchmarked (not just asserted) against a classical baseline.

## 📤 Push to GitHub

```bash
cd quantumfleet-ai
git add -A
git commit -m "QuantumFleet AI prototype: fuel prediction + classical vs quantum-inspired fleet optimization"
git branch -M main
git remote add origin https://github.com/<your-username>/quantumfleet-ai.git
git push -u origin main
```

## 🔭 Future scope (mentioned, not built for MVP)

- Live weather API integration
- Real vessel telemetry / GPS / AIS feed instead of simulated data
- PostgreSQL in place of SQLite (same SQLAlchemy code, just change `DATABASE_URL`)
- Feedback loop: actual voyage fuel vs predicted → periodic model retraining
