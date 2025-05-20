import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("lalonde_data.csv")
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])
for col in ['re74', 're75', 're78']:
    if col in df.columns:
        df[col] = df[col].fillna(0)
T = df['treat']          
Y = df['re78']            
X = df.drop(columns=['treat', 're78'])  
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("X_scaled shape:", X_scaled.shape)
print("T (treatment) shape:", T.shape)
print("Y (outcome) shape:", Y.shape)
