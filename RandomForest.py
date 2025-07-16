import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt

df=pd.read_csv('dataset_sintetico.csv')
X=df.drop(columns=['GBS'])
y=df['GBS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
modelo=RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
#Probabilidad de GBS
#Sospecha
y_proba = modelo.predict_proba(X_test)[:, 1]
#Binarización de acuerdo a umbral (certeza)
#Decisión
y_pred=(y_proba>=0.6).astype(int)
#Reporte sin calibración
print(classification_report(y_test, y_pred))
calibrated_model=CalibratedClassifierCV(modelo, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)
y_proba_calibrado=calibrated_model.predict_proba(X_test)[:, 1]
y_pred_calibrado=(y_proba_calibrado>=0.6).astype(int)
#reporte calibrado
print(classification_report(y_test, y_pred_calibrado))
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(modelo, X, y, cv=cv, scoring='f1_weighted')
print("F1 promedio en CV:", scores.mean())
fraction_pos, mean_pred=calibration_curve(y_test, y_proba, n_bins=10)
fraction_pos_cal, mean_pred_cal=calibration_curve(y_test, y_proba_calibrado, n_bins=10)
plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado')
plt.plot(mean_pred, fraction_pos, 's-', label='Modelo sin calibrar')
plt.plot(mean_pred_cal, fraction_pos_cal, 'o-', label='Modelo calibrado')
plt.xlabel('Probabilidad predicha')
plt.ylabel('Fracción de positivos')
plt.legend()
plt.title('Curvas de calibración')
plt.show()