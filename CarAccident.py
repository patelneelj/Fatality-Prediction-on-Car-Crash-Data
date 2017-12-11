
# coding: utf-8

# In[148]:


from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[149]:


def get_iris_data():
    """Get the Car accident, from local csv or pandas repo."""
    if os.path.exists("C:\\Users\\patel.DESKTOP-SE1G8AK\\Downloads\\733 DataMining\\Project\\nassCDS.csv"):
        print("--nassCDS.csv found locally")
        df = pd.read_csv("C:\\Users\\patel.DESKTOP-SE1G8AK\\Downloads\\733 DataMining\\Project\\nassCDS.csv", index_col=0)
    else:
        print("-- trying to download from database")
        fn = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/DAAG/nassCDS.csv"
        try:
            df = pd.read_csv(fn)
        except:
            exit("-- Unable to download nassCDS.csv")

        with open("nassCDS.csv", 'w') as f:
            print("-- writing to local nassCDS.csv file")
            df.to_csv(f)

    return df


# In[150]:


df = get_iris_data()


# In[151]:


print("* df.head()", df.head(), sep="\n", end="\n\n")
print("* df.tail()", df.tail(), sep="\n", end="\n\n")


# In[152]:


print("* Injury Levels:", df["injSeverity"].unique(), sep="\n")
df["injSeverity"].unique().shape


# In[153]:


print("* Injury Levels:", df["dead"].unique(), sep="\n")
df["dead"].unique().shape


# In[154]:


def encode_target(df, target_column, newName):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[newName] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


# In[155]:


df2, deadN = encode_target(df, "dead","deadN")
print("* df2.head()", df2[["deadN", "dead"]].head(),
      sep="\n", end="\n\n")
print("* df2.tail()", df2[["deadN", "dead"]].tail(),
      sep="\n", end="\n\n")
print("* deadN", deadN, sep="\n", end="\n\n")
df2["deadN"]


# In[156]:


df3, injSeverityN = encode_target(df, "injSeverity","injSeverityN")
print("* df2.head()", df3[["injSeverityN", "injSeverity"]].head(),
      sep="\n", end="\n\n")
print("* df2.tail()", df3[["injSeverityN", "injSeverity"]].tail(),
      sep="\n", end="\n\n")
print("* injSeverityN", injSeverityN, sep="\n", end="\n\n")
df3["injSeverityN"]


# In[157]:


df4, airbagN = encode_target(df, "airbag","airbagN")
print("* df4.head()", df4[["airbagN", "airbag"]].head(),
      sep="\n", end="\n\n")
print("* df4.tail()", df4[["airbagN", "airbag"]].tail(),
      sep="\n", end="\n\n")
print("* airbagN", airbagN, sep="\n", end="\n\n")


# In[158]:


df5, seatbeltN = encode_target(df, "seatbelt","seatbeltN")
print("* df5.head()", df5[["seatbeltN", "seatbelt"]].head(),
      sep="\n", end="\n\n")
print("* df5.tail()", df5[["seatbeltN", "seatbelt"]].tail(),
      sep="\n", end="\n\n")
print("* seatbeltN", seatbeltN, sep="\n", end="\n\n")


# In[159]:


df6, sexN = encode_target(df, "sex","sexN")
print("* df6.head()", df6[["sexN", "sex"]].head(),
      sep="\n", end="\n\n")
print("* df6.tail()", df6[["sexN", "sex"]].tail(),
      sep="\n", end="\n\n")
print("* sexN", sexN, sep="\n", end="\n\n")


# In[160]:


df7, abcatN = encode_target(df, "abcat","abcatN")
print("* df2.head()", df7[["abcatN", "abcat"]].head(),
      sep="\n", end="\n\n")
print("* df2.tail()", df7[["abcatN", "abcat"]].tail(),
      sep="\n", end="\n\n")
print("* abcatN", abcatN, sep="\n", end="\n\n")


# In[161]:


df8, occRoleN = encode_target(df, "occRole","occRoleN")
print("* df8.head()", df8[["occRoleN", "occRole"]].head(),
      sep="\n", end="\n\n")
print("* df8.tail()", df8[["occRoleN", "occRole"]].tail(),
      sep="\n", end="\n\n")
print("* occRoleN", occRoleN, sep="\n", end="\n\n")


# In[169]:


df5


# In[186]:



import pandas as pd 
frames = pd.merge(frames,df8)
frames


# In[191]:


features = list(frames.columns[15:])
print("* features:", features, sep="\n")


# In[193]:


y = df3["injSeverityN"]
X = df2[features]
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)

