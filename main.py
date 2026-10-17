import numpy as np

from models import LinearRegression, LogisticRegression

rng = np.random.default_rng(42)


def house_prices():
    area = rng.uniform(30, 120, 200)
    price = 1500 * area + 20000 + rng.normal(0, 10000, 200)
    return area.reshape(-1, 1), price


def exam_results():
    hours = rng.uniform(0, 10, 200)
    passed = (hours + rng.normal(0, 1.5, 200) > 5).astype(int)
    return hours.reshape(-1, 1), passed


def main():
    area, price = house_prices()
    mean, std = area.mean(), area.std()
    linear = LinearRegression(learning_rate=0.1, epochs=500).fit((area - mean) / std, price)
    slope = linear.weights[0] / std
    intercept = linear.bias - slope * mean
    print("Regresie liniara: pretul unui apartament dupa suprafata")
    print(f"  pret = {slope:.0f} * suprafata + {intercept:.0f}   (date generate cu 1500 * suprafata + 20000)")
    print(f"  eroare medie: {np.sqrt(linear.losses[-1]):.0f} EUR")
    print(f"  apartament de 75 m2: {linear.predict([[(75 - mean) / std]])[0]:.0f} EUR\n")

    hours, passed = exam_results()
    logistic = LogisticRegression(learning_rate=0.5, epochs=2000).fit(hours, passed)
    accuracy = np.mean(logistic.predict(hours) == passed)
    print("Regresie logistica: promovarea examenului dupa orele de studiu")
    print(f"  acuratete: {accuracy:.0%}")
    for h in (2, 5, 8):
        print(f"  {h} ore de studiu -> probabilitate de promovare {logistic.output([[h]])[0]:.0%}")


if __name__ == "__main__":
    main()
