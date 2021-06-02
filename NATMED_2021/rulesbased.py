import numpy as np
import pandas as pd

# Data Parameters
# Cut-off values found during CV of binary per-view classifiers
predicted_probabilities_file = "data/AbNL_probpreds_121019.csv"
cols = [0,1,2,3,5,6,7,8]
cut_off_dict = {0:0.7337093, 1:0.67976165, 2:0.4500074}
best_threshold = -0.736294143732291

# Vector Data
# Col 1 identifies patient
# Cols 2-4 identify image within patient study
# Col 5 identifies the cardiac view (true or predicted, depending on cohort) {0:3VT, 1:3VV, 2:LVOT, 3:A4C, 4:ABDO}
# Col 6 is the predicted probability that the view is NL
# Col 7 is the predicted probability that the view is AbNL
# Cols 6 & 7 sum to 1, and are the results of the per-view binary predictions
DF = pd.read_csv(predicted_probabilities_file, names=["id",	"clip", "frame", "total_frame", "view", "NL_prob", "Abnormal_prob"])

# Apply per-view cutoffs to [3VT, 3VV, LVOT] views
DF.loc[(DF["view"] == 0) & (DF["Abnormal_prob"] < cut_off_dict[0]), "Abnormal_prob"] = 0
DF.loc[(DF["view"] == 1) & (DF["Abnormal_prob"] < cut_off_dict[1]), "Abnormal_prob"] = 0
DF.loc[(DF["view"] == 2) & (DF["Abnormal_prob"] < cut_off_dict[2]), "Abnormal_prob"] = 0

# Group by id/view, calculate sum pf probabilities per-view
DF = DF.set_index(["id", "view"], inplace=False)
DF.sort_index(inplace=True)
DF = DF.groupby(["id", "view"])["NL_prob", "Abnormal_prob"].sum().reset_index()
DF = pd.pivot_table(DF, index=["id"], columns="view", values=["NL_prob", "Abnormal_prob"], fill_value=0)

# Scale per-view sum for each id
DF.iloc[:,[0,5]]=DF.iloc[:,[0,5]].div(np.sum(DF.iloc[:,[0,5]].values,axis=1),axis=0).fillna(0)
DF.iloc[:,[1,6]]=DF.iloc[:,[1,6]].div(np.sum(DF.iloc[:,[1,6]].values,axis=1),axis=0).fillna(0)
DF.iloc[:,[2,7]]=DF.iloc[:,[2,7]].div(np.sum(DF.iloc[:,[2,7]].values,axis=1),axis=0).fillna(0)
DF.iloc[:,[3,8]]=DF.iloc[:,[3,8]].div(np.sum(DF.iloc[:,[3,8]].values,axis=1),axis=0).fillna(0)
DF.iloc[:,[4,9]]=DF.iloc[:,[4,9]].div(np.sum(DF.iloc[:,[4,9]].values,axis=1),axis=0).fillna(0)

# Remove vectors with only Abdomen views
DF = DF[np.sum(DF.astype(bool).iloc[:,cols], axis=1) != 0]

# Use only [3VT, 3VV, LVOT, A4C] views
X = DF.iloc[:, cols]

# Apply rule-based classifier
# CHD labels {0:NL, 1: AbNL}
y_pred_labels = np.zeros(len(X))
AbNL_sum = np.sum(X.iloc[:,0:4].values, axis=1)
NL_sum = np.sum(X.iloc[:,4:].values, axis=1)
y_pred_labels[(AbNL_sum - NL_sum) > best_threshold] = 1

print(y_pred_labels)