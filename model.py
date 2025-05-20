import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
df = pd.read_csv("lalonde_data.csv")
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])
df[['re74', 're75', 're78']] = df[['re74', 're75', 're78']].fillna(0)
T = df['treat']                  
Y = df['re78']                     
X = df.drop(columns=['treat', 're78']) 
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
log_model = LogisticRegression()
log_model.fit(X_scaled, T)
propensity_scores = log_model.predict_proba(X_scaled)[:, 1] 

treated_idx = T == 1
control_idx = T == 0

X_treated = X_scaled[treated_idx]
X_control = X_scaled[control_idx]
Y_treated = Y[treated_idx].values
Y_control = Y[control_idx].values
propensity_control = propensity_scores[control_idx]
nn = NearestNeighbors(n_neighbors=1)
nn.fit(propensity_control.reshape(-1, 1))
_, indices = nn.kneighbors(propensity_scores[treated_idx].reshape(-1, 1))
matched_Y_control = Y_control[indices.flatten()]
ate_psm = np.mean(Y_treated - matched_Y_control)
print("Estimated ATE using PSM:", round(ate_psm, 2))
