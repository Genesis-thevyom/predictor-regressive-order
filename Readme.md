# Predictor Regressive Order

A modular machine learning project for experimenting with different regression models and techniques. This project provides a framework for linear, multi-variable, and polynomial regression, along with utilities for feature scaling, cost function calculation, and gradient descent optimization.

## Project Structure

```
main.py                  # Entry point for running-experiments
requirements.txt         # Python dependencies
Readme.md                # Project documentation

data/
    generate.py          # Synthetic data generator
model/
    linear_regression.py     # Linear regression implementation
    multi_regression.py      # Multi-variable regression implementation
    polynomial_regression.py # Polynomial regression implementation
utils/
    cost_function.py         # Cost function calculations
    feature_scaling.py       # Feature scaling utilities
    gradient_descent.py      # Gradient descent optimizer
    learning_rate.py         # Learning rate scheduler
```

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib
  - jupyter

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. **Generate Data:**
   - Use scripts in `data/` to generate or preprocess datasets.
2. **Select Model:**
   - Implement or modify regression models in `model/`.
3. **Run Experiments:**
   - Use `main.py` to run training, evaluation, and visualization.
4. **Utilities:**
   - Utilities in `utils/` help with scaling, optimization, and cost calculations.

## Extending

- Add new regression models in the `model/` directory.
- Add new data generators or preprocessors in `data/`.
- Add or modify utility functions in `utils/`.

## License

MIT License
