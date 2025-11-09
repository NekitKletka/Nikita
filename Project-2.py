import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("testi/Salary_Data.csv")

print(df.info())
print(df.isna().sum())

df.dropna(inplace=True)

numerical_data = ['Age', 'Years of Experience']
category_data = ['Gender', 'Education Level', 'Job Title']

transform = ColumnTransformer(
    transformers=[
        ('text', StandardScaler(), numerical_data),
        ('code', OneHotEncoder(drop='first', handle_unknown='ignore'), category_data)
    ]
)

X = df.drop('Salary', axis=1)
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

predicts = []
fitted_models = {}
models = {
    'Linear': LinearRegression(),
    'Ridge': RidgeCV(),
    'Lasso': LassoCV()
}

for name, model in models.items():
    model = Pipeline([
        ('preprocessor', transform),
        ('regressor', model)
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predicts.append(y_pred)
    fitted_models[name] = model
    print(name)
    print('R2', r2_score(y_test, y_pred))
    print('MAE', mean_absolute_error(y_test, y_pred))
    print('MSE', mean_squared_error(y_test, y_pred))



models = {
    'Linear': (predicts[0], 'Blue'),
    'Ridge': (predicts[1], 'Red'),
    'Lasso': (predicts[2], 'Yellow')
}

fig, axes = plt.subplots(3, 1, figsize=(10, 9))

for ax, (name, (y_hat, color)) in zip(axes, models.items()):
    ax.scatter(y_test, y_hat, color=color, alpha=0.6, label=name)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Идеальное совпадение')
    ax.legend()
    ax.grid(True)
fig.suptitle('Сравнение моделей', fontsize = 20)
fig.text(0.5, 0.04, 'Фактическая зарплата', ha='center')
fig.text(0.04, 0.5,'Предсказанная зарплата', va='center', rotation='vertical')
plt.show()

results = []
for name, y_pred in zip(models.keys(), predicts):
    results.append({
        'Model': name,
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print(results_df)
sns.barplot(x='Model', y='R2', data=results_df)
plt.title('Сравнение моделей по R2')
plt.show()

corr = df[['Age', 'Years of Experience', 'Salary']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Корреляция числовых признаков')
plt.show()
print(corr)


new_employee = pd.DataFrame({
    'Age': [20],
    'Gender': ['Male'],
    'Education Level': ['Master’s'],
    'Job Title': ['Director of Data Science'],
    'Years of Experience': [5]
})

best_model = fitted_models['Ridge']
pred_salary = best_model.predict(new_employee)
print('Прогнозная зарплата нового сотрудника:', int(pred_salary))

"""
    Итоговый отчёт

Лучшая модель: Ridge (R2 0.880484239887088; MAE 13073.368740544041; MSE 331806495.59396744)
Главный фактор, влияющий на зарплату: Years of Experience
Вывод: Модель подходит для базового прогнозирования зарплат,
но её можно улучшить, если добавить больше данных

"""

