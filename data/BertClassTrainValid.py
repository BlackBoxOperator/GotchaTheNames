import pandas as pd
from sklearn.model_selection import train_test_split

"""
4700(4651) default aml, 300 (372) default non-aml,
1500 our aml, dayN (1534 aml/non-aml)

total: 4700 aml, 1800 non-aml (valid not included)

valid dayN.csv
train: 3700 non-aml, 1400 aml
test: 1000 non-aml, 400 (372) aml

"""
aml = "aml.csv"
content = "content.csv"
newcont = "newamlcont.csv"

df1 = pd.read_csv(aml)
df2 = pd.read_csv(content)
df3 = pd.read_csv(newcont)

del df3['query']
df3.insert(1, 'title', [''] * len(df3))
df3['label'] = [1] * len(df3)

our_aml = df3

df2['label'] = [int(idx in df1['index'].values) for idx in df2['index']]

dft_aml = df2.loc[df2['label'] == 1]
dft_non_aml = df2.loc[df2['label'] == 0]

dft_aml = dft_aml.sample(frac=1).reset_index(drop=True)

non_aml_train, non_aml_valid = train_test_split( \
       dft_non_aml, test_size=10/47, random_state = 44)

aml_train, aml_valid = train_test_split( \
       dft_aml, test_size=72, random_state = 44)

"""
setup 1.

train: 3700 (3661) non-aml, 1100 our aml + 372 default aml
test: 1000 (990) non-aml, 400 our aml
"""
our_aml_train, our_aml_valid = train_test_split( \
       our_aml, test_size=400, random_state = 44)

setup1_train = pd.concat([non_aml_train, our_aml_train, dft_aml], axis=0)
setup1_valid = pd.concat([non_aml_valid, our_aml_valid], axis=0)
setup1_train = setup1_train.sample(frac=1)
setup1_valid = setup1_valid.sample(frac=1)
setup1_train.to_csv("TrainClassifySetup1.csv", index=False)
setup1_valid.to_csv("ValidClassifySetup1.csv", index=False)

print('setup1:', len(setup1_train), len(setup1_valid))

"""
setup 2.

train: 3700 (3661) non-aml, 1500 our aml
test: 1000 (990) non-aml, 372 default aml
"""
#setup2_train = pd.concat([non_aml_train, our_aml], axis=0)
#setup2_valid = pd.concat([non_aml_valid, dft_aml], axis=0)
#setup2_train = setup2_train.sample(frac=1)
#setup2_valid = setup2_valid.sample(frac=1)
#setup2_train.to_csv("TrainClassifySetup2.csv", index=False)
#setup2_valid.to_csv("ValidClassifySetup2.csv", index=False)
#
#print('setup2:', len(setup2_train), len(setup2_valid))

"""
setup 3.

train: 3700 (3661) non-aml, 1100 our aml + 300 default aml
test: 1000 (990) non-aml, 400 our aml + 72 default aml
"""

#setup3_train = pd.concat([non_aml_train, our_aml_train, aml_train], axis=0)
#setup3_valid = pd.concat([non_aml_valid, our_aml_valid, aml_valid], axis=0)
#setup3_train = setup3_train.sample(frac=1)
#setup3_valid = setup3_valid.sample(frac=1)
#setup3_train.to_csv("TrainClassifySetup3.csv", index=False)
#setup3_valid.to_csv("ValidClassifySetup3.csv", index=False)
#
#print('setup3:', len(setup3_train), len(setup3_valid))

"""
check set
"""

#checks = [pd.read_csv('day{}.csv'.format(i)) for i in range(1, 5)]
#check = pd.concat(checks, axis=0)
#del check['predict']
#check.insert(1, 'title', [''] * len(check))
#check['label'] = [int(l) for l in check['label']]
#check.to_csv("CheckClassify.csv", index=False)
#print('check:', len(check))
#
#weakness = pd.read_csv('weak.csv')
#del weakness['predict']
#weakness.insert(1, 'title', [''] * len(weakness))
#weakness['label'] = [int(l) for l in weakness['label']]
#weakness.to_csv("WeakClassify.csv", index=False)
#print('weakness:', len(weakness))

#old setup
#train, valid = train_test_split(df2, test_size=0.2, stratify=df2['label'], random_state = 44)
#df2.to_csv("all.csv", index = False)
#valid.to_csv("ValidClassify.csv", index=False)
#train.to_csv("TrainClassify.csv", index=False)
