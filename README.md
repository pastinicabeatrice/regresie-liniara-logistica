# Regresie liniară și logistică

Regresia liniară și regresia logistică implementate de la zero în NumPy, antrenate cu gradient descent.

## Modele

| Model | Ieșire | Funcție de cost | Exemplu din `main.py` |
|---|---|---|---|
| `LinearRegression` | valoare reală | eroarea pătratică medie | prețul unui apartament după suprafață |
| `LogisticRegression` | probabilitate (sigmoid) | entropia încrucișată | promovarea unui examen după orele de studiu |

Ambele modele moștenesc aceeași clasă, `GradientDescentModel`. Pentru amândouă, gradientul
are aceeași formă, `Xᵀ(ŷ - y) / n`; diferă doar funcția de ieșire și costul.

`standardize` aduce trăsăturile la medie 0 și deviație 1, ca gradient descent să conveargă repede.

## Rulare

```
pip install -r requirements.txt
python main.py
python -m unittest
```
