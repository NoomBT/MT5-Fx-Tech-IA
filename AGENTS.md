# AGENTS.md - MT5-Fx-Tech-IA

## Project Overview
Python trading bot with Flask web API + ML-based trading signals for MetaTrader 5.

## Run
```bash
python flasktradebot.py  # Start Flask server on port 5000
pip install -r requirements.txt  # Install dependencies
python ml_model.py  # Train ML model
```

## Architecture
- `flasktradebot.py`: Flask API + trading loop + order execution (1300+ lines)
- `ml_model.py`: ML model training (Random Forest, 3-class: BUY/NEUTRAL/SELL)
- `ml_model_gpu.py`: GPU-accelerated variant
- `templates/index.html`: Web dashboard
- `ml_model.pkl`: Trained model

## Key Settings (in flasktradebot.py)
- Default symbol: XAUUSDm
- Default timeframe: M15
- AI model: ml_model.pkl
- Trend filter: EMA 200 on H1 timeframe

## Triple Barrier Labeling
- TP: 1.0% (~100 pips for XAUUSD)
- SL: 0.5% (~50 pips)
- Max holding: 10 bars (RR 2:1)

## Current Model Performance
- Accuracy: ~41%
- Return: +18% in simulation
- Best features: volatility, ema slopes, ny_volatility, m15_slope

## Session Features
- London: 8-16 UTC
- NY: 13-21 UTC  
- ny_volatility = ny_session * volatility

## Important Notes
- Requires MT5 Terminal with API enabled
- TA-Lib needs separate binary on Windows
- Session features are most predictive
- Model tends to predict Sell better than Buy (precision 0.58 vs 0.27)

## Quick Commands
```bash
del ml_model.pkl  # Delete old model before retraining
python ml_model.py
```