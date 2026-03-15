import shap
import numpy as np
import pandas as pd

def model_insights(model, input_df):

    # ─────────────────────────────────────────
    # Transform pipeline data
    # ─────────────────────────────────────────
    X_fe = model.named_steps['feature_engineering'].transform(input_df)
    X_pre = model.named_steps['preprocess'].transform(X_fe)

    feature_names = model.named_steps['preprocess'].get_feature_names_out()

    # ─────────────────────────────────────────
    # Feature groups
    # ─────────────────────────────────────────
    numeric_features = ['amt','zip','lat','long','city_pop','unix_time',
                        'merch_lat','merch_long','Year','Month','Day','Hour','IsWeekend','age']

    low_card_cat = ['category','state','gender']
    high_card_cat = ['merchant','city','job']

    groupable_features = numeric_features + low_card_cat + high_card_cat

    # ─────────────────────────────────────────
    # Get stack models
    # ─────────────────────────────────────────
    stack_clf = model.named_steps['stack']

    model_info = [
        ("RandomForest", stack_clf.estimators_[0]),
        ("XGBoost",      stack_clf.estimators_[1]),
        ("CatBoost",     stack_clf.estimators_[2]),
        ("LightGBM",     stack_clf.estimators_[3]),
        ("Keras",        stack_clf.estimators_[4]),
    ]

    all_contributions = {f: [] for f in groupable_features}
    successful_models = []

    # ─────────────────────────────────────────
    # SHAP per model
    # ─────────────────────────────────────────
    for name, base_model in model_info:

        print(f"→ {name:12}", end=" ")

        try:

            if name == "Keras":

                explainer = shap.KernelExplainer(base_model.predict_proba, X_pre)
                shap_all = explainer.shap_values(X_pre)

                shap_vals = shap_all[1][0] if isinstance(shap_all, list) else shap_all[0]

            else:

                explainer = shap.TreeExplainer(
                    base_model,
                    model_output="raw",
                    feature_perturbation="tree_path_dependent"
                )

                shap_all = explainer.shap_values(X_pre)
                shap_vals = shap_all[1][0] if isinstance(shap_all, list) else shap_all[0]

            # Group contributions
            for feat in groupable_features:

                idx = [i for i, col in enumerate(feature_names) if feat in col]

                if idx:
                    contrib = shap_vals[idx].sum()
                    all_contributions[feat].append(contrib)

            successful_models.append(name)
            print("OK")

        except Exception as e:

            print(f"FAILED ({type(e).__name__})")

    # ─────────────────────────────────────────
    # Aggregate contributions
    # ─────────────────────────────────────────
    rows = []
    
    for feat, vals in all_contributions.items():

        if len(vals) > 0:
                
            rows.append({
                "Feature": feat,
                "Mean_SHAP": np.mean(vals),
                "Std_SHAP": np.std(vals),
                "Num_Models": len(vals),
                "Direction": "Positive" if np.mean(vals) > 0 else "Negative"
            })

    result_df = pd.DataFrame(rows)

    if not result_df.empty:

        total = np.abs(result_df["Mean_SHAP"]).sum()

        if total > 0:
            result_df["Abs. Scale(%)"] = (
                result_df["Mean_SHAP"] / total * 100
            )

        result_df = result_df.sort_values(
            "Mean_SHAP",
            key=abs,
            ascending=False
        ).reset_index(drop=True)
        
    return {
        "successful_models": successful_models,
        "feature_contributions": result_df
    }