import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time

import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

st.title('タイタニック号 生存予測アプリ')
img = Image.open('titanic_ticket.jpg')
st.image(img, use_column_width=True)

#########################################
#年齢
age = st.number_input('年齢', 0, 100, 20)

#性別
sex_male = 0
sex_female = 0
gender = st.selectbox(
    '性別',
    ('男', '女'))
if gender == '男':
    sex_male = 1
    sex_female = 0
else:
    sex_male = 0
    sex_female = 1

#配偶者
sibsp = 0
partner = st.selectbox(
    '配偶者の有無',
    ('なし', 'あり'))
if partner =='あり':
    sibsp = 1
else:
    sibsp = 0

#子供
parch = 0
child = st.selectbox(
    '子供の有無',
    ('なし', 'あり'))
if child =='あり':
    parch = 1
else:
    parch = 0

#客室等級
pclass = 3
pclass_st = st.selectbox(
    '客室等級',
    ('1', '2', '3'))
if pclass_st =='1':
    pclass = 1
elif pclass_st =='2':
    pclass = 2
else:
    pclass = 3

#DataFrame作成
test_dict = {
    "Pclass":pclass,
    "Age":age,
    "SibSp":sibsp,
    "Parch":parch,
    "Sex_female":sex_female,
    "Sex_male":sex_male
}
test = pd.DataFrame(test_dict, index=['0'])
# テストデータの予測を格納する、418行5列のnumpy行列を作成
test_pred = np.zeros((len(test), 5))
###########################################

button = st.button('Predict')

if button:
    train = pd.read_csv('train.csv')
    sample_submission = pd.read_csv('gender_submission.csv')
    # SexとEmbarkedのOne-Hotエンコーディング
    train = pd.get_dummies(train, columns=['Sex'])
    # 不要な列の削除
    train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Fare', 'Embarked'], axis=1, inplace=True)
    X_train = train.drop(['Survived'], axis=1)  # X_trainはtrainのSurvived列以外
    y_train = train['Survived']  # Y_trainはtrainのSurvived列
    # 5分割交差検証を指定し、インスタンス化
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # スコアとモデルを格納するリスト
    score_list = []
    models = []
    for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):    
        print(f'fold{fold_ + 1} start')
        train_x = X_train.iloc[train_index]
        valid_x = X_train.iloc[valid_index]
        train_y = y_train[train_index]
        valid_y = y_train[valid_index]
    
        # lab.Datasetを使って、trainとvalidを作っておく
        lgb_train= lgb.Dataset(train_x, train_y)
        lgb_valid = lgb.Dataset(valid_x, valid_y)
        # パラメータを定義
        lgbm_params = {'objective': 'binary'}

        # lgb.trainで学習
        gbm = lgb.train(params=lgbm_params,
                        train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_valid],
                        early_stopping_rounds=20,
                        verbose_eval=-1
                        )
        oof = (gbm.predict(valid_x) > 0.5).astype(int)
        score_list.append(round(accuracy_score(valid_y, oof)*100,2))
        models.append(gbm)  # 学習が終わったモデルをリストに入れておく

    for fold_, gbm in enumerate(models):
        test_pred[:, fold_] = gbm.predict(test) # testを予測
        pred = np.mean(test_pred, axis=1) * 100
    st.title('あなたの生存確率は ' + str(round(pred[0],2)) + '%です!')
    st.write('※予想モデルの正答率は76.32%')
