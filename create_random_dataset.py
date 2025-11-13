import numpy as np
import pandas as pd
from sklearn.utils import shuffle

np.random.seed(21)

biomas = ["Amazônia", "Cerrado", "Mata Atlântica", "Caatinga", "Pampa", "Pantanal"]
climas = ["Equatorial", "Tropical", "Semiárido", "Subtropical"]

n_samples = 200
data = {
    "bioma": np.random.choice(biomas, n_samples),
    "precipitacao": np.random.uniform(500, 3000, n_samples),  # mm/ano
    "clima": np.random.choice(climas, n_samples),
    "temperatura": np.random.uniform(15, 35, n_samples),  # °C
    "tempo_abandono": np.random.uniform(1, 20, n_samples),  # anos
}

altura = (
    0.005 * data["precipitacao"]
    + 0.3 * data["temperatura"]
    + 0.8 * data["tempo_abandono"]
    + np.random.normal(0, 2, n_samples)
)

data["altura_arvore"] = altura

df = pd.DataFrame(data)
df = shuffle(df).reset_index(drop=True)

df.to_csv("dataset_trees.csv", index=False)
print("Dataset created.")