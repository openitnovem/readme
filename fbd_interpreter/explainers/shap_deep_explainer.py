import shap


class ShapDeepExplainer(object):
    def __init__(self, model):
        self.model = model

    def apply_shap_plot(self, train_data, feature_names):
        shap_values = shap.KernelExplainer(self.model).shap_values(
            train_data[feature_names]
        )
        fig_1 = shap.summary_plot(
            shap_values, train_data[feature_names], plot_type="bar"
        )
        fig_2 = shap.summary_plot(shap_values, train_data[feature_names])
        return fig_1, fig_2

    def shap_plot(self, obs_num, data_test):
        explainerModel = shap.KernelExplainer(self.model)
        shap_values_Model = explainerModel.shap_values(data_test)
        shap_local_fig = shap.force_plot(
            explainerModel.expected_value,
            shap_values_Model[obs_num],
            data_test.iloc[[obs_num]],
            link="logit",
        )
        return shap_local_fig
