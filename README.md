# MT5-Fx-Tech-IA

Python trading bot with Flask web API + Machine Learning signals for MetaTrader 5.

## Features

- **AI Trading**: Random Forest ML model for BUY/NEUTRAL/SELL signals
- **Multiple Strategies**: AI, BB+MA+MACD, EMA+MACD+RSI crossover
- **Trend Filter**: EMA 200 on H1 timeframe
- **Risk Management**: SL/TP, Trailing Stop, Martingale
- **Grid Trading**: Automated grid orders
- **Web Dashboard**: Real-time control via browser

## Requirements

- MetaTrader 5 Terminal (with API enabled)
- Python 3.8+
- TA-Lib (requires separate binary installation on Windows)

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python flasktradebot.py
```

Access dashboard at http://localhost:5000

## Default Settings

- Symbol: XAUUSDm
- Timeframe: M15
- Model: ml_model.pkl (CPU) or ml_model_gpu.pkl (GPU)

## Project Structure

| File | Description |
|------|-------------|
| `flasktradebot.py` | Main entry - Flask API + trading loop (1300+ lines) |
| `ml_model.py` | ML model training (Random Forest, 3-class) |
| `ml_model_gpu.py` | GPU-accelerated model variant |
| `ml_model.pkl` | Trained CPU model |
| `ml_model_gpu.pkl` | Trained GPU model |
| `templates/index.html` | Web dashboard |
| `logs/` | Activity logs |

## Strategies

1. **AI/RF**: ML-based signals using Random Forest classifier
2. **BB+MA+MACD**: Bollinger Bands + Moving Average + MACD
3. **EMA+MACD+RSI**: EMA crossover + MACD + RSI confirmation

## Dependencies

- MetaTrader5 - MT5 terminal API
- TA-Lib - Technical indicators
- Flask - Web server
- pandas, numpy, scikit-learn - ML/data processing