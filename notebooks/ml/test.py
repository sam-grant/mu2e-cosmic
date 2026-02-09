run = "j"
cutset_name = "MLProcess"

import sys
sys.path.append("../../src/ml")
from assemble import AssembleDataset
from train import Train

assembler = AssembleDataset(run=run, cutset_name=cutset_name, verbosity=2)
# assembler.draw_hists()
# cut_flows = assembler.get_cut_flows()
# for name, cut_flow in cut_flows.items():
#     print(f"{name}:\n{cut_flow}") 

print(assembler.check_dT_window_results())

data = assembler.assemble_dataset()
trainer = Train(data)
training_results = trainer.train(
    tag="xgb_baseline",
    save_output=True,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

# from tensorflow import keras
# 
# # Define Keras model
# model = keras.Sequential([
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 
# # Train with Keras
# trainer = Train(data, model=model, scale_features=True)
# results = trainer.train(
#     tag="keras_baseline",
#     save_output=True,
#     epochs=50,
#     batch_size=32,
#     validation_split=0.2
# )

# # load results
# import joblib 
# with open(f"../../output/ml/{run}/results/xgb_baseline/results.pkl", "rb") as f:
#     training_results = joblib.load(f)

print(training_results)

# Now test/validate the model 
from analyse_model import AnaModel
ana = AnaModel(training_results)
# ana.print_summary()
# ana.plot_roc(show=False)
# ana.plot_score_distribution()
# ana.plot_feature_importance()
threshold_results = ana.find_threshold(show=False)

# ana.plot_score_distribution(
#     vlines=threshold_results["threshold"],
#     xmax=0.001
# )

ana.plot_low_score_physics(
    df_full=data["df_full"],
    threshold=threshold_results["threshold"]
)

ana.money_table(
    X=data["X_test"],
    y=data["y_test"],
    metadata=data["metadata_test"],
    threshold=threshold_results["threshold"]
)
ana.money_table(
    X=data["df_full"][ana.feature_names],
    y=data["df_full"]["label"],
    metadata=data["df_full"][["subrun", "event"]],
    threshold=threshold_results["threshold"]
)
# # from load import LoadML
# # data = LoadML(run=run, verbosity=2).get_ml_data()

# from assemble import AssembleDataset
# ass = AssembleDataset(run=run)

# # ass.draw_hists()
# a, b = ass.get_cut_flows()
# print(a)
# print(b)

# print(ass.check_dT_window_results())

# Check false negatives — are they NaN dT (no CRV coincidence)?
threshold = threshold_results["threshold"]
df_check = data["metadata_test"].copy()
df_check["label"] = data["y_test"].values
df_check["dT"] = data["X_test"]["dT"].values
df_check["score"] = ana.y_proba

# CRY events not vetoed by ML (event-level: max score per event)
event_check = df_check.groupby(["subrun", "event"]).agg(
    max_score=("score", "max"),
    label=("label", "first"),
    dT_values=("dT", list)
).reset_index()

ml_fn = event_check[(event_check["label"] == 1) & (event_check["max_score"] < threshold)]
print(f"\nML false negatives ({len(ml_fn)} events, threshold={threshold:.4f}):")
print(ml_fn[["subrun", "event", "max_score", "dT_values"]].to_string(index=False))

# dT cut false negatives — CRY events where no coincidence has dT in [0, 150]
import numpy as np
dT_min, dT_max = 0, 150
event_check["dT_vetoed"] = event_check["dT_values"].apply(
    lambda dts: any((not np.isnan(dt)) and dT_min <= dt <= dT_max for dt in dts)
)
dt_fn = event_check[(event_check["label"] == 1) & (~event_check["dT_vetoed"])]
print(f"\ndT cut false negatives ({len(dt_fn)} events, window=[{dT_min}, {dT_max}] ns):")
print(dt_fn[["subrun", "event", "max_score", "dT_values"]].to_string(index=False))

# Events missed by one method but caught by the other
ml_only_fn = ml_fn[~ml_fn.set_index(["subrun", "event"]).index.isin(
    dt_fn.set_index(["subrun", "event"]).index)]
dt_only_fn = dt_fn[~dt_fn.set_index(["subrun", "event"]).index.isin(
    ml_fn.set_index(["subrun", "event"]).index)]
print(f"\nMissed by ML only (dT catches): {len(ml_only_fn)}")
print(ml_only_fn[["subrun", "event", "max_score", "dT_values"]].to_string(index=False))
print(f"\nMissed by dT only (ML catches): {len(dt_only_fn)}")
print(dt_only_fn[["subrun", "event", "max_score", "dT_values"]].to_string(index=False))