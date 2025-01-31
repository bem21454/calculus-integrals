import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, Exponentiation, WhiteKernel
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

def collect_data(row: int):
    with (open('data/data.csv', 'r', newline='') as file):
        reader = csv.reader(file, delimiter=',')
        raw_data = list(reader)
        years = np.array(list(map(lambda _: [int(_)], raw_data[0][1:])))
        emissions = np.array(list(map(lambda _: round(float(_), 2), raw_data[row][1:])))
        source = raw_data[row][0]
    return years, emissions, source

def generate_gpr_model(X, Y, source):
    # Create Kernel
    kernel = (1 * Matern(length_scale_bounds=(1e-10, 1e10)))*(1 * Exponentiation(DotProduct(), 2)) + (1 * WhiteKernel())

    # Initialize GPR Model
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=1000,
    )

    # Split data
    train_input, test_input, train_output, test_output = train_test_split(
        X,
        Y,
        test_size=0.20,
        shuffle=True,
        random_state=1_000
    )

    # Fit training data
    gpr.fit(train_input, train_output)

    # Create test data
    x = np.linspace(1950, 2050, 12_801)[:, np.newaxis]
    y, sigma = gpr.predict(x, return_std=True)

    print(f'{source}: {gpr.score(X, Y)}')
    return x, y, sigma

def plot(X, Y, x, y, sigma, source):
    plt.plot(X, Y, '.', label=r'Emissions')
    plt.plot(x, y, label=f'{source} Mean Prediction')
    plt.fill_between(
        x.ravel(),
        y - 1.96 * sigma,
        y + 1.96 * sigma,
        alpha=0.2,
        label=r'95% Confidence Interval'
    )
    plt.legend()
    plt.xlabel('Years')
    plt.ylabel('Emissions (MMT)')

def main():
    for _ in range(1, 8):  # 1-8
        years, emissions, source = collect_data(_)
        x, y, sigma = generate_gpr_model(years, emissions, source)
        plot(years, emissions, x, y, sigma, source)

        data = zip(x.ravel(), y, sigma)
        with open(f'data/{source}.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['year', 'emission', 'std_dev'])
            writer.writeheader()
            writer.writerows({'year': y, 'emission': e, 'std_dev': s} for y, e, s in data)

        with open(f'data/{source}_raw.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['year', 'emission'])
            writer.writeheader()
            writer.writerows({'year': y, 'emission': e} for y, e in zip(years.ravel(), emissions))
    plt.show()

if __name__ == "__main__":
    main()
