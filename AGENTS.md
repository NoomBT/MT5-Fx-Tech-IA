# MT5-Fx-Tech-IA

## Project Overview
Python trading bot with Flask web API + ML-based trading signals for MetaTrader 5.

## Run
```bash
python flasktradebot.py
```
Flask server starts on port 5000. Open http://localhost:5000 for web dashboard.

## Dependencies
Install via `pip install -r requirements.txt`. Key packages:
- `MetaTrader5` - MT5 terminal API
- `TA-Lib` - Technical indicators (requires installation)
- `pandas`, `numpy`, `scikit-learn` - ML/data processing
- `Flask` - Web server

## Architecture
- `flasktradebot.py`: Main entry - Flask API + trading loop + order execution (1300+ lines)
- `ml_model.py`: ML model training script (Random Forest, 3-class: BUY/NEUTRAL/SELL)
- `ml_model_gpu.py`: GPU-accelerated model variant
- `ml_model.pkl`, `ml_model_gpu.pkl`: Trained model files

## Key Trading Settings (in flasktradebot.py)
- Default symbol: `XAUUSDm`
- Default timeframe: `M15`
- AI signal models: `ml_model.pkl` (CPU) or `ml_model_gpu.pkl` (GPU)
- Trend filter: EMA 200 on H1 timeframe
- Strategies: AI/Random Forest, BB+MA+MACD, EMA+MACD+RSI crossover

## Testing
No formal test framework. Manual testing via:
1. Start Flask app
2. Use web UI or API endpoints to control bot
3. Check `logs/` directory for activity

## Notes
- Requires MT5 Terminal running with API enabled
- TA-Lib may need separate binary installation on Windows
- GPU model requires CUDA-compatible environment