# Загрузка библиотек

import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# Загрузка данных

df = pd.read_csv("heart_failure_clinical_dataset.csv")

# Создание списков переменных

features = list(df.columns)[:-1]
target = df.columns[-1]

# Разделение на тренировочную и тестовую выборки, сохранение выборок

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=17)

X_test.to_csv("X_test.csv", index=None)
y_test.to_csv("y_test.csv", index=None)

X_train.to_csv("X_train.csv", index=None)
y_train.to_csv("y_train.csv", index=None)


# Сборка пайплайна

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


continuous_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
                   'serum_creatinine', 'serum_sodium', 'time']
base_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

continuous_transformers = []
base_transformers = []

for cont_col in continuous_cols:
    transformer = Pipeline([
        ('selector', NumberSelector(key=cont_col)),
        ('standard', StandardScaler())])
    continuous_transformers.append((cont_col, transformer))

for base_col in base_cols:
    base_transformer = Pipeline([
        ('selector', NumberSelector(key=base_col))])
    base_transformers.append((base_col, base_transformer))

# Объединение трансформеров

feats = FeatureUnion(continuous_transformers + base_transformers)
feature_processing = Pipeline([('feats', feats)])

feature_processing.fit_transform(X_train)

# Классификатор

pipeline = Pipeline([
    ('features', feats),
    ('classifier', RandomForestClassifier(random_state=17))])

# Обучение пайплайна

pipeline.fit(X_train, y_train)

# Сохранение пайплайна

with open("classifier_pipeline.dill", "wb") as f:
    dill.dump(pipeline, f)

