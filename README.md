import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
# Загружаем датасет (предположим, что данные находятся в файле 'credit_data.csv')
data = pd.read_csv('credit_data.csv')
# Разделяем данные на признаки (X) и целевую переменную (y)
X = data.drop('Дефолт', axis=1)
y = data['Дефолт']
# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Инициализируем модель XGBoost и задаем параметры
model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
# Обучаем модель на обучающих данных
model.fit(X_train, y_train)
# Делаем предсказания на тестовых данных
predictions = model.predict(X_test)
# Вычисляем точность модели
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Выводим отчет о классификации
print("Classification Report:")
print(classification_report(y_test, predictions))
