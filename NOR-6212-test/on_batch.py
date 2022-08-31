from sklearn.ensemble import IsolationForest

columns = attributes['columns2']
df = df.dropna()

if params['strictness2'] != 0:
    strictness = 1/(params['strictness2']+2)
    df["outlier"] = IsolationForest(contamination=strictness, random_state=0).fit_predict(df[columns].values)
else:
    df["outlier"] = IsolationForest(random_state=0).fit_predict(df[columns].values) # auto

df["outlier"] = df["outlier"].replace(to_replace={-1: "yes", 1: "no"})

return df
