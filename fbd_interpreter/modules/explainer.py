import os

import shap

from fbd_interpreter.icecream import icecream


def apply_ale_plot(model, data, feature_names, target_col, classif=False):
    ale_plots = icecream.IceCream(
        data=data[feature_names],
        feature_names=feature_names,
        bins=10,
        model=model,
        targets=data[target_col],
        aggfunc="mean",
        use_classif_proba=classif,
        use_ale=True,
    )
    figs = ale_plots.draw(kind="ale", show=False)
    return figs


def compute_pdp(model, data, feature_names, target_col, classif=False):
    pdp_plots = icecream.IceCream(
        data=data.drop([target_col], axis=1),
        feature_names=feature_names,
        bins=10,
        model=model,
        targets=data[target_col],
        aggfunc="mean",
        use_classif_proba=classif,
    )
    return pdp_plots


def apply_pdp_ice_plot(model, data, feature_names, target_col, classif=False):
    pdp_plots = compute_pdp(
        model=model,
        data=data,
        feature_names=feature_names,
        target_col=target_col,
        classif=classif,
    )
    figs_pdp = pdp_plots.draw(kind="pdp", show=False)
    figs_ice = pdp_plots.draw(kind="ice", show=False)
    return figs_pdp, figs_ice


def apply_shap_plot(model, data, feature_names):
    shap_values = shap.KernelExplainer(model).shap_values(data[feature_names])
    fig_1 = shap.summary_plot(shap_values, data[feature_names], plot_type="bar")
    fig_2 = shap.summary_plot(shap_values, data[feature_names])
    return fig_1, fig_2


def shap_plot(obs_num, model, X_test):
    explainerModel = shap.KernelExplainer(model)
    shap_values_Model = explainerModel.shap_values(X_test)
    shap_local_fig = shap.force_plot(
        explainerModel.expected_value,
        shap_values_Model[obs_num],
        X_test.iloc[[obs_num]],
        link="logit",
    )
    return shap_local_fig
