# ============================================================
# REALMLP FAST BLEND FIXED VERSION
# Airline Satisfaction Kaggle
#
# Fix:
#   1. traveler_segment / arrival_missing KeyError
#   2. TargetEncoder uses generated features correctly
#   3. safer category + numeric processing
#
# ============================================================


import os
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore")


import torch

from pytabkit import RealMLP_TD_Classifier



# ============================================================
# PATH
# ============================================================

TRAIN_PATH = (
    r"D:\kaggle\airline\data\train.csv"
)

TEST_PATH = (
    r"D:\kaggle\airline\data\test.csv"
)


LGBM_OOF_PATH = (
    r"D:\kaggle\airline\lgbm_realmlp_blend"
    r"\lgbm_realmlp_blend_oof.csv"
)


OUTPUT_DIR = Path(
    r"D:\kaggle\airline\realmlp_fast"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)



# ============================================================
# SETTINGS
# ============================================================


TARGET = "satisfaction"

ID_COL = "id"

SEED = 42

N_FOLD = 5



DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("="*80)
print("DEVICE:", DEVICE)

if DEVICE == "cuda":
    print(
        torch.cuda.get_device_name(0)
    )

print("="*80)



# ============================================================
# COLUMN SET
# ============================================================


CATS = [

    "Gender",

    "Customer Type",

    "Type of Travel",

    "Class",

]


SERVICES = [

    "Inflight wifi service",

    "Departure/Arrival time convenient",

    "Ease of Online booking",

    "Gate location",

    "Food and drink",

    "Online boarding",

    "Seat comfort",

    "Inflight entertainment",

    "On-board service",

    "Leg room service",

    "Baggage handling",

    "Checkin service",

    "Cleanliness",

]


NUMERIC = [

    "Age",

    "Flight Distance",

    "Departure Delay in Minutes",

    "Arrival Delay in Minutes",

]



# ============================================================
# LOAD DATA
# ============================================================


def load_data():

    train = pd.read_csv(
        TRAIN_PATH
    )


    test = pd.read_csv(
        TEST_PATH
    )


    y = (
        train[TARGET]
        .astype("int8")
        .values
    )


    return train,test,y




# ============================================================
# BASIC FEATURES
# ============================================================


def basic_features(df):


    X = df.drop(
        columns=[
            ID_COL,
            TARGET
        ],
        errors="ignore"
    ).copy()



    X["traveler_segment"] = (

        df["Type of Travel"].astype(str)

        +"|"

        +df["Customer Type"].astype(str)

        +"|"

        +df["Class"].astype(str)

    )



    X["arrival_missing"] = (

        df["Arrival Delay in Minutes"]
        .isna()
        .astype("int8")

    )


    return X




# ============================================================
# CONDITIONAL FEATURES
# ============================================================


def conditional_features(df):


    out = pd.DataFrame(
        index=df.index
    )


    seg=(

        df["Type of Travel"].astype(str)

        +"|"

        +df["Customer Type"].astype(str)

        +"|"

        +df["Class"].astype(str)

    )



    for c in SERVICES:


        out[c+"_segment"]=(

            df[c].astype(str)

            +"|"

            +seg

        )



    pairs=[

        (
            "Inflight wifi service",
            "Online boarding"
        ),

        (
            "Seat comfort",
            "Inflight entertainment"
        ),

        (
            "Inflight wifi service",
            "Ease of Online booking"
        )

    ]



    group=(

        df["Type of Travel"].astype(str)

        +"|"

        +df["Customer Type"].astype(str)

    )



    for a,b in pairs:


        out[a+"_"+b]=(

            df[a].astype(str)

            +"|"

            +df[b].astype(str)

            +"|"

            +group

        )



    return out





# ============================================================
# BUILD FOLD FEATURES FIXED
# ============================================================


def build_fold_features(
    train,
    test,
    tr_idx,
    va_idx,
    y
):


    X_train = basic_features(
        train
    )


    X_test = basic_features(
        test
    )



    original=list(
        X_train.columns
    )



    # -----------------------------
    # categorical
    # -----------------------------


    for c in CATS + ["traveler_segment"]:


        cats=pd.Index(

            pd.concat(

                [
                    X_train[c],
                    X_test[c]
                ]

            )
            .astype(str)
            .unique()

        )


        dtype=pd.api.types.CategoricalDtype(
            cats
        )


        X_train[c]=X_train[c].astype(
            dtype
        )


        X_test[c]=X_test[c].astype(
            dtype
        )



    # -----------------------------
    # numeric fill
    # -----------------------------


    for c in X_train.columns:


        if c not in CATS + ["traveler_segment"]:


            if X_train[c].dtype!="category":


                med=X_train[c].median()


                X_train[c]=X_train[c].fillna(
                    med
                )


                X_test[c]=X_test[c].fillna(
                    med
                )



    Xtr=X_train.iloc[tr_idx].copy()

    Xva=X_train.iloc[va_idx].copy()

    Xte=X_test.copy()



    # ========================================================
    # TARGET ENCODE
    # FIXED HERE
    # ========================================================


    Z=X_train[
        original
    ].astype(str)


    U=X_test[
        original
    ].astype(str)



    encoder=TargetEncoder(

        target_type="binary",

        smooth="auto",

        cv=5,

        shuffle=True,

        random_state=SEED

    )



    tr_te=encoder.fit_transform(

        Z.iloc[tr_idx],

        y[tr_idx]

    )


    va_te=encoder.transform(
        Z.iloc[va_idx]
    )


    te_te=encoder.transform(
        U
    )



    for i,c in enumerate(original):


        name="te_"+c.replace(
            " ",
            "_"
        )


        Xtr[name]=tr_te[:,i]

        Xva[name]=va_te[:,i]

        Xte[name]=te_te[:,i]



    # ========================================================
    # CONDITIONAL TARGET ENCODE
    # ========================================================


    C1=conditional_features(
        train
    )

    C2=conditional_features(
        test
    )


    encoder2=TargetEncoder(

        target_type="binary",

        smooth=20,

        cv=5,

        random_state=SEED

    )



    a=encoder2.fit_transform(

        C1.iloc[tr_idx],

        y[tr_idx]

    )


    b=encoder2.transform(
        C1.iloc[va_idx]
    )


    c=encoder2.transform(
        C2
    )



    for i in range(a.shape[1]):

        Xtr[f"cond_{i}"]=a[:,i]

        Xva[f"cond_{i}"]=b[:,i]

        Xte[f"cond_{i}"]=c[:,i]



    # ========================================================
    # CLEAN
    # ========================================================


    for df in [
        Xtr,
        Xva,
        Xte
    ]:


        df.replace(
            [np.inf,-np.inf],
            np.nan,
            inplace=True
        )


        for col in df.columns:


            if str(df[col].dtype)!="category":


                df[col]=(
                    df[col]
                    .fillna(0)
                    .astype("float32")
                )


    return Xtr,Xva,Xte
# ============================================================
# LOAD LGBM OOF
# ============================================================


def load_lgbm_oof():


    df=pd.read_csv(
        LGBM_OOF_PATH
    )


    train=pd.read_csv(
        TRAIN_PATH,
        usecols=[
            ID_COL,
            TARGET
        ]
    )


    pred=(

        df["lgbm"]
        .values
        .astype("float64")

    )


    y=(

        train[TARGET]
        .astype("int8")
        .values

    )


    auc=roc_auc_score(
        y,
        pred
    )


    print(
        "Existing LGBM OOF:",
        auc
    )


    return y,pred




# ============================================================
# REALMLP TRAIN
# ============================================================


def train_realmlp(
    Xtr,
    ytr,
    Xva,
    yva,
    Xtest,
    config,
    seed
):


    model=RealMLP_TD_Classifier(

        device=DEVICE,

        random_state=seed,

        n_cv=1,

        n_refit=0,

        n_epochs=config["epochs"],

        batch_size=config["batch"],

        hidden_sizes=config["hidden"],

        lr=config["lr"],

        val_metric_name="cross_entropy",

        use_ls=False,

        verbosity=0

    )



    model.fit(

        Xtr,

        ytr,

        Xva,

        yva

    )



    val_pred=(

        model.predict_proba(
            Xva
        )[:,1]

    )



    test_pred=(

        model.predict_proba(
            Xtest
        )[:,1]

    )


    del model

    gc.collect()


    if DEVICE=="cuda":

        torch.cuda.empty_cache()



    return val_pred,test_pred




# ============================================================
# CONFIGS
# ============================================================


CONFIGS=[


    {

        "name":"mlp_small",

        "batch":1024,

        "epochs":160,

        "lr":0.04,

        "hidden":[256,256,256]

    },


    {

        "name":"mlp_large",

        "batch":1024,

        "epochs":192,

        "lr":0.035,

        "hidden":[320,320,320]

    },


    {

        "name":"mlp_deep",

        "batch":1024,

        "epochs":160,

        "lr":0.04,

        "hidden":[256,256,256,256]

    }

]





# ============================================================
# BLEND SEARCH
# ============================================================


def find_best_weight(
    y,
    lgbm,
    mlp
):


    best_auc=0

    best_w=0.5



    for w in np.arange(
        0,
        1.001,
        0.005
    ):


        pred=(

            w*lgbm

            +

            (1-w)*mlp

        )


        auc=roc_auc_score(
            y,
            pred
        )


        if auc>best_auc:


            best_auc=auc

            best_w=w



    return (
        best_w,
        1-best_w,
        best_auc
    )





# ============================================================
# MAIN
# ============================================================


def main():


    start=time.time()


    train,test,y=load_data()



    lgbm_y,lgbm_oof=load_lgbm_oof()



    assert np.all(
        y==lgbm_y
    )



    skf=StratifiedKFold(

        n_splits=N_FOLD,

        shuffle=True,

        random_state=SEED

    )



    results=[]

    predictions=[]



    for cfg in CONFIGS:



        print(
            "\n"+"="*80
        )

        print(
            "MODEL:",
            cfg["name"]
        )

        print(
            "="*80
        )



        oof=np.zeros(
            len(train)
        )


        test_pred=np.zeros(
            len(test)
        )



        for fold,(tr,va) in enumerate(

            skf.split(train,y),

            1

        ):



            print(
                "\nFold",
                fold
            )


            t=time.time()



            Xtr,Xva,Xtest=build_fold_features(

                train,

                test,

                tr,

                va,

                y

            )


            print(
                "Feature shape:",
                Xtr.shape
            )



            val_pred,pred_test=train_realmlp(

                Xtr,

                y[tr],

                Xva,

                y[va],

                Xtest,

                cfg,

                SEED+fold

            )



            oof[va]=val_pred


            test_pred += (
                pred_test/N_FOLD
            )



            auc=roc_auc_score(

                y[va],

                val_pred

            )


            print(
                "Fold AUC:",
                auc
            )


            print(
                "Time:",
                round(
                    (time.time()-t)/60,
                    2
                ),
                "min"
            )



            del Xtr,Xva,Xtest

            gc.collect()



        mlp_auc=roc_auc_score(
            y,
            oof
        )


        corr=np.corrcoef(

            lgbm_oof,

            oof

        )[0,1]



        lw,mw,blend_auc=find_best_weight(

            y,

            lgbm_oof,

            oof

        )



        print("\nRESULT")

        print(
            "RealMLP AUC:",
            mlp_auc
        )

        print(
            "Correlation:",
            corr
        )

        print(
            "Blend AUC:",
            blend_auc
        )

        print(
            "LGBM weight:",
            lw
        )

        print(
            "RealMLP weight:",
            mw
        )



        results.append(

            {

                "name":cfg["name"],

                "mlp_auc":mlp_auc,

                "corr":corr,

                "lgbm_weight":lw,

                "mlp_weight":mw,

                "blend_auc":blend_auc

            }

        )



        predictions.append(

            {

                "name":cfg["name"],

                "test":test_pred

            }

        )





    results=pd.DataFrame(
        results
    )


    results=results.sort_values(

        "blend_auc",

        ascending=False

    )



    print("\nFINAL RANKING")

    print(
        results.to_string(
            index=False
        )
    )



    best_name=results.iloc[0]["name"]



    print(
        "\nBEST MODEL:",
        best_name
    )



    best_pred=[

        x["test"]

        for x in predictions

        if x["name"]==best_name

    ][0]



    # ========================================================
    # SAVE REALMLP TEST PRED
    # ========================================================


    pd.DataFrame(

        {

            ID_COL:
            test[ID_COL],

            "realmlp":
            best_pred

        }

    ).to_csv(

        OUTPUT_DIR
        /
        "realmlp_test_prediction.csv",

        index=False

    )



    results.to_csv(

        OUTPUT_DIR
        /
        "results.csv",

        index=False

    )



    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
    )


    print(

        "Runtime:",

        round(
            (time.time()-start)/60,
            2
        ),

        "min"

    )





if __name__=="__main__":

    main()