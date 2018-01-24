import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


def replaceNotIrr(x):
    if x != 'irregular':
        return 'other'
    return x

np.random.seed(42)
#
# Получение данных
#
dataset = pd.read_csv('features.csv')

#
# Разделяем (тестовый и тренировочный наборы)
#
train = dataset[dataset['type'] == 'train']
test = dataset[dataset['type'] == 'test']


classes = list(map(replaceNotIrr, train.pop('class').values))
features = train.ix[:, 'local_clust[pp.im_sq(70)]':]

testClasses = list(map(replaceNotIrr, test.pop('class').values))
testFeatures = test.ix[:, 'local_clust[pp.im_sq(70)]':]

#
# Тренировка модели

#0.972947147467 1000

model = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=3000, n_jobs=2,
            oob_score=False, random_state=None, verbose=2, warm_start=False)
model.fit(features, classes)
#
# Отчёты о точности работы
#
print(classification_report(testClasses, model.predict(testFeatures)))
print(accuracy_score(testClasses, model.predict(testFeatures)))
