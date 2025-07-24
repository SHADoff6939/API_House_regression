import pandas as pd
from sklearn.model_selection import train_test_split

# Завантажте оригінальний файл
data = pd.read_csv('original_train.csv')

# Розділіть дані у співвідношенні 80/20
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Збережіть результати
train_data.to_csv('train.csv', index=False)
val_data.to_csv('val.csv', index=False)

print(f"Розділено успішно! Навчальний набір: {len(train_data)} записів, Валідаційний: {len(val_data)} записів")