"""
**************************************************************************************
AutoMoL: Pipeline for automated machine learning for drug design.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
**************************************************************************************

This file includes the main functionality for showing the results from the automol generated models in streamlit.

This modules contains the methods:

validate_properties_in_data
figure_lay_out
show_figures_in_columns

and the classes

ModelVisualiser
ClassificationVisualiser
RegressionVisualiser
"""
import streamlit as st
import pandas as pd
import numpy as np

from app_model_training import *

from automol.plotly_util import PlotlyDesigner, create_figure_captions, figure_details


def validate_properties_in_data(stacked_model):
    """
    verifies that the properties for which the stacking model has estimators are present in the validation set of the stacking model
    """
    for p in stacked_model.models:
        if p not in stacked_model.Validation.columns:
            return False
    return True


def figure_lay_out():
    """
    Creates streamlit boxes for the layout of the figures.
    
    Returns
    user_col: the ordered list of colors for the different graphs in a plot
    nb_cols: the number of columns in which streamlit structures the different figures
    use_fixed_width: boolean to set if the figures should have fixed width and not use the width of the container and thus the width of the column
    x_width: width of the figures, only used if the value above is set
    y_height: height of the figures
    """
    user_col = split_in_strings(
        st.text_input(
            'Provide list of colors for graphs separated by ,',
            "darkorange,darkred,darkgreen,deepskyblue,darkviolet,orangered,yellowgreen",
            help=
            "This list of python colors is used for the different graphs in a figure. The colors are used in the given order. Separate each color by a comma (,)."
        ))
    nb_cols = int(
        st.number_input(
            'Number of columns for grid of figures',
            1,
            5,
            2,
            step=1,
            help="Set the number of column to structure all the figures."))
    #y_height=st.slider('figure width', 300,1200, 600)
    use_fixed_width = st.checkbox(
        'Use fixed width',
        help=
        "If this is set, the figure are shown using the width given below, if not, the width depends on the number of given columns and equals the column width."
    )
    x_width = int(
        st.number_input(
            'Figure width',
            300,
            1200,
            600,
            step=10,
            help=
            "The width of the figure, only used when the checkbox -Use fixed width- above is set"
        ))
    y_height = int(
        st.number_input('Figure height',
                        300,
                        1200,
                        600,
                        step=10,
                        help="The height of the figure"))

    return user_col, nb_cols, use_fixed_width, x_width, y_height


def show_figures_in_columns(plotly_dictionary,
                            fig_list,
                            selected_figures,
                            nb_cols=3,
                            use_container_width=True,
                            start_index=1):
    captions, types, _ = create_figure_captions(selected_figures,
                                                start_index=start_index)
    captions = [
        capti.replace('\33[1m', '').replace('\33[0m', '') for capti in captions
    ]
    markdown_str = from_list_to_markdown(string_list=types)
    cols = st.columns(nb_cols)
    for index, fig in enumerate(selected_figures):
        plotly_dictionary[fig] = fig_list[fig]
        cols[int(index % nb_cols)].plotly_chart(
            fig_list[fig], use_container_width=use_container_width)
        cols[int(index % nb_cols)].caption(captions[index])
    st.markdown(markdown_str)
    return plotly_dictionary, start_index + len(selected_figures)


class ModelVisualiser:
    """
    A parent (abstract) class for visualization of the classification and regression figures

    List of methods:
    :method __init__: constructor
    :method generate_general_figures (virtual) : creates the list of figures 
    :method show_general_figures: shows the figures 
    :method generate_bokeh (virtual) :  generate bokeh figures (regression figures with chemical structure)
    :method generate_additional_figures:  checkbox to generate advanced figures
    :method generate_task_figures  (virtual): generate task specific advanced figures
    """
    def __init__(self, user_col, nb_cols, use_fixed_width, x_width, y_height):
        """
        Constructor
        
        :param user_col: list of colors
        :param nb_cols: number of columns to show figures
        :param use_fixed_width: boolean to indicate use of container width in streamlit columns
        :param x_width: width of the figures, unused if use_fixed_width is True
        :param y_height: height of the figures
        """
        self.user_col = user_col
        self.nb_cols = nb_cols
        self.use_fixed_width = use_fixed_width
        self.x_width = x_width
        self.y_height = y_height
        self.generate_bokeh_reg = False
        self.advanced_figures = False
        self.revert_log10 = False
        self.apply_tick_transformation = False
        self.nb_figures = 1
        self.good_class=None
        self.cutoff=None

    def generate_general_figures(self, class_properties, out, stacked_model):
        """
        Generate a list of general plotly figures, intended for inheritance.
        
        :param class_properties: list of properties that have been fitted in the model 
        :param out: the output of the model
        :param stacked_model: the stacking model
        """
        pass

    def show_general_figures(self, plotly_dictionary):
        """
        Shows the created figures, captions and figure type explanation. If generate_bokeh_reg is set, the method generate_bokeh() is called. If not the generated plotly figures are shown.
        
        :param plotly_dictionary: a dictionary of plotly figures that collects the figures over all stages. 
        """
        if self.generate_bokeh_reg:
            for index, fig in enumerate(self.fig_l):
                plotly_dictionary[fig] = self.fig_l[fig]
            self.generate_bokeh()
        else:
            cols = st.columns(self.nb_cols)
            for index, fig in enumerate(self.selected_figures):
                plotly_dictionary[fig] = self.fig_l[fig]
                cols[int(index % self.nb_cols)].plotly_chart(
                    self.fig_l[fig],
                    use_container_width=not self.use_fixed_width)
                cols[int(index % self.nb_cols)].caption(self.captions[index])
        markdown_str = from_list_to_markdown(string_list=self.types)
        st.markdown(markdown_str)
        return plotly_dictionary

    def generate_bokeh(self):
        """
        Virtual method to generate and show bokeh figure (chemical structure figures)
        """
        pass

    def generate_additional_figures(self, plotly_dictionary):
        """
        Check if the user wants advanced figures and then calls generate_task_figures
        
        :param plotly_dictionary: a dictionary of plotly figures that collects the figures over all stages. 
        """
        self.advanced_figures = st.checkbox("Generate advanced figures?",
                                            value=True)
        if self.advanced_figures:
            plotly_dictionary = self.generate_task_figures(plotly_dictionary)
        return plotly_dictionary

    def generate_task_figures(self, plotly_dictionary):
        """
        Virtual method to generate and show task specific advanced figures.
        
        :param plotly_dictionary: a dictionary of plotly figures that collects the figures over all stages. 
        """
        pass


class ClassificationVisualiser(ModelVisualiser):
    def __init__(self, user_col, nb_cols, use_fixed_width, x_width, y_height):
        """
        Constructor that displays two boxes for the choice of colormaps used in the classification figures
        """
        super().__init__(user_col, nb_cols, use_fixed_width, x_width, y_height)
        with st.expander("Select colormaps for classification reports"):
            cmap1 = st.selectbox('Colormap for classification report', [
                'PiYG', 'viridis', 'YlGnBu', 'Blues', 'aggrnyl', 'agsunset',
                'blackbody', 'bluered', 'blugrn', 'bluyl', 'brwnyl', 'bugn',
                'bupu', 'burg', 'burgyl', 'cividis', 'darkmint', 'electric',
                'emrld', 'gnbu', 'greens', 'greys', 'hot', 'inferno', 'jet',
                'magenta', 'magma', 'mint', 'orrd', 'oranges', 'oryel',
                'peach', 'pinkyl', 'plasma', 'plotly3', 'pubu', 'pubugn',
                'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdpu',
                'redor', 'reds', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
                'turbo', 'ylgn', 'ylorbr', 'ylorrd', 'algae', 'amp', 'deep',
                'dense', 'gray', 'haline', 'ice', 'matter', 'solar', 'speed',
                'tempo', 'thermal', 'turbid', 'armyrose', 'brbg', 'earth',
                'fall', 'geyser', 'prgn', 'picnic', 'portland', 'puor', 'rdgy',
                'rdylbu', 'rdylgn', 'spectral', 'tealrose', 'temps', 'tropic',
                'balance', 'curl', 'delta', 'oxy', 'edge', 'hsv', 'icefire',
                'phase', 'twilight', 'mrybm', 'mygbm'
            ])
            cmap2 = st.selectbox('Colormap for confusion matrix', [
                'Blues', 'PiYG', 'viridis', 'YlGnBu', 'aggrnyl', 'agsunset',
                'blackbody', 'bluered', 'blugrn', 'bluyl', 'brwnyl', 'bugn',
                'bupu', 'burg', 'burgyl', 'cividis', 'darkmint', 'electric',
                'emrld', 'gnbu', 'greens', 'greys', 'hot', 'inferno', 'jet',
                'magenta', 'magma', 'mint', 'orrd', 'oranges', 'oryel',
                'peach', 'pinkyl', 'plasma', 'plotly3', 'pubu', 'pubugn',
                'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdpu',
                'redor', 'reds', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
                'turbo', 'ylgn', 'ylorbr', 'ylorrd', 'algae', 'amp', 'deep',
                'dense', 'gray', 'haline', 'ice', 'matter', 'solar', 'speed',
                'tempo', 'thermal', 'turbid', 'armyrose', 'brbg', 'earth',
                'fall', 'geyser', 'prgn', 'picnic', 'portland', 'puor', 'rdgy',
                'rdylbu', 'rdylgn', 'spectral', 'tealrose', 'temps', 'tropic',
                'balance', 'curl', 'delta', 'oxy', 'edge', 'hsv', 'icefire',
                'phase', 'twilight', 'mrybm', 'mygbm'
            ])
        with st.expander("Information on the Classification metrics", expanded=True):
            st.info(
                ' Classification metrics:\n * **Accuracy**: is the ratio of correctly classified samples. Misleading for unbalanced data sets. \n * **AUC-ROC**: see for example [TowardsDataScience](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5). \n * **Precision**: precision looks at the predicted samples of class and is the ratio of the correct predicted samples w.r.t. the total amount of predicted samples of that class. It gives the ratio of how much of that specific predicted class is correctly labeled (True positive / (True positive + False positive). It does not say anything about the true samples of that class that were missed (and thus falsely mislabeled). \n * **Recall**: Is the ratio of true predictions of that class that were found by the classifier. It is the number of correctly classified samples of that class w.r.t. the total amount of samples of that class (True positive / (True positive + False negative)). Note that this does not say anything about how many samples of that class are falsely predicted as this class, e.g. the false positive. \n * **F1-score**: f1-score is  is the harmonic mean of the precision and recall (2 * (precision*recall)/(precision+recall)). \n \n More details on precision, recall and F1-score can be found on [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall) '
            )
        self.cmap = [cmap1, cmap2]

    def generate_general_figures(self, class_properties, out, stacked_model):
        """
        Creates the general figures for classification results and stores all the relevant parameters. 
        """
        self.labelnames = stacked_model.labelnames
        self.stacked_model = stacked_model
        self.out = out
        self.class_properties = class_properties
        youden_dict, fig_l = PlotlyDesigner(
            colors=self.user_col).show_classification_report(
                self.class_properties,
                self.out, [
                    self.stacked_model.Validation[f'{p}'].values
                    for p in self.class_properties
                ],
                labelnames=self.labelnames,
                cmap=self.cmap,
                fig_size=(self.x_width, self.y_height))
        self.youden_dict = youden_dict
        self.fig_l = fig_l
        self.selected_figures = list(self.fig_l.keys())
        #st.multiselect(
        #    "Select Figures to be shown",
        #    self.fig_l,
        #    default=self.fig_l,
        #    help="Select figures from the list",
        #    )
        self.captions, self.types, _ = create_figure_captions(
            self.selected_figures, start_index=self.nb_figures)
        self.captions = [
            capti.replace('\33[1m', '').replace('\33[0m', '')
            for capti in self.captions
        ]
        self.types = [
            capti.replace('\33[1m', '**').replace('\33[0m', '**')
            for capti in self.types
        ]
        self.nb_figures = self.nb_figures + len(self.selected_figures)

    def generate_task_figures(self, plotly_dictionary):
        """
        generates and displays the advanced classification figures
        """
        with st.expander("Threshold tuning for classification", expanded=True):
            F1_dict, fig_list = PlotlyDesigner(
                colors=self.user_col).show_clf_threshold_report(
                    self.class_properties,
                    self.out, [
                        self.stacked_model.Validation[f'{p}'].values
                        for p in self.class_properties
                    ],
                    youden_dict=self.youden_dict,
                    labelnames=self.labelnames,
                    fig_size=(self.x_width, self.y_height))
            selected_th_figures = st.multiselect(
                "Select threshold tuning figures to be shown",
                fig_list,
                default=fig_list,
                help="Select figures from the list",
            )

            plotly_dictionary, self.nb_figures = show_figures_in_columns(
                plotly_dictionary,
                fig_list,
                selected_th_figures,
                nb_cols=self.nb_cols,
                use_container_width=not self.use_fixed_width,
                start_index=self.nb_figures)

        with st.expander("Advanced classification figures", expanded=True):
            fig_l = PlotlyDesigner(
                colors=self.user_col).show_additional_classification_report(
                    self.class_properties,
                    self.out, [
                        self.stacked_model.Validation[f'{p}'].values
                        for p in self.class_properties
                    ],
                    labelnames=self.labelnames,
                    cmap=self.cmap,
                    fig_size=(self.x_width, self.y_height))
            selected_figures = st.multiselect(
                "Select advanced classification figures to be shown",
                fig_l,
                default=fig_l,
                help="Select figures from the list",
            )

            plotly_dictionary, self.nb_figures = show_figures_in_columns(
                plotly_dictionary,
                fig_l,
                selected_figures,
                nb_cols=self.nb_cols,
                use_container_width=not self.use_fixed_width,
                start_index=self.nb_figures)

        return plotly_dictionary


class RegressionVisualiser(ModelVisualiser):
    def __init__(self, user_col, nb_cols, use_fixed_width, x_width, y_height,
                 smiles_list):
        """
        Constructor to generate and display results in streamlit. An additional parameter is added containing the smiles of the samples. 
        
        :param smiles_list: the list of smiles 
        """
        super().__init__(user_col, nb_cols, use_fixed_width, x_width, y_height)
        self.smiles_list = smiles_list
        self.revert_log10 = st.checkbox(
            'Revert transformed values to the original values')
        self.apply_tick_transformation = st.checkbox(
            'Plot transformed values but with original values on axis')
        if self.revert_log10 and self.apply_tick_transformation:
            self.apply_tick_transformation = False
        if self.apply_tick_transformation:
            st.warning(
                'This only transforms the ticks on the axis, not the values themselves. Setting cutoffs still needs to be done using a transformed value'
            )
        self.generate_bokeh_reg = st.checkbox(
            "Generate scatter plot with chemical structure",
            value=False)
        if self.generate_bokeh_reg:
            st.warning(
                'It can happen that streamlit gets stuck in a loop of generating the scatter plots with chemical structures (latency issues). Uncheck the box for scatterplots with chemical structures if this happens.'
            )
            self.legend_pos = st.selectbox('Legend position for scatterplot', [
                'bottom_right', 'bottom_left', 'top_right', 'top_left',
                "top_center", "center_right", "center_left", "bottom_center",
                "center"
            ])

    def generate_general_figures(self, class_properties, out, stacked_model):
        """
        Generates and saves the general figures to display the regression results and locally saves the required parameters from the given stacking model. 
        """
        self.leave_group_out = None
        self.prop_cliff_dict = None
        if hasattr(stacked_model, 'leave_group_out'):
            self.leave_group_out = stacked_model.leave_group_out
        if hasattr(stacked_model, 'prop_cliff_dict'):
            self.prop_cliff_dict = stacked_model.prop_cliff_dict

        self.stacked_model = stacked_model
        self.out = out
        self.class_properties = class_properties
        self.log10_props = class_properties

        if self.revert_log10:
            self.class_properties = [
                '_'.join(p.split('_')[1:]) for p in self.class_properties
            ]
            if self.prop_cliff_dict is not None:
                self.prop_cliff_dict = {
                    '_'.join(key.split('_')[1:]): val
                    for key, val in self.prop_cliff_dict.items()
                }

        self.fig_l = PlotlyDesigner(
            colors=self.user_col).show_regression_report(
                self.class_properties,
                self.out,
                y_true=[
                    self.stacked_model.Validation[f'{p}'].values
                    for p in self.class_properties
                ],
                prop_cliffs=self.prop_cliff_dict,
                leave_grp_out=self.leave_group_out,
                fig_size=(self.x_width, self.y_height),
                smiles=list(self.smiles_list.values),
                apply_tick_transformation=self.apply_tick_transformation)

        self.selected_figures = list(self.fig_l.keys())
        print(self.selected_figures)
        self.captions, self.types, _ = create_figure_captions(
            self.selected_figures, self.nb_figures)
        self.nb_figures = self.nb_figures + len(self.selected_figures)
        self.captions = [
            capti.replace('\33[1m', '').replace('\33[0m', '')
            for capti in self.captions
        ]
        self.types = [
            capti.replace('\33[1m', '**').replace('\33[0m', '**')
            for capti in self.types
        ]

        if self.generate_bokeh_reg:
            self.fig_bokeh = PlotlyDesigner(
                colors=self.user_col).show_bokeh_scatter(
                    self.class_properties,
                    self.out,
                    y_true=[
                        self.stacked_model.Validation[f'{p}'].values
                        for p in self.class_properties
                    ],
                    fig_size=(self.x_width, self.y_height),
                    smiles=list(self.smiles_list.values),
                    legend_pos=self.legend_pos,
                    apply_tick_transformation=self.apply_tick_transformation)
            self.selected_bokeh_reg_figures = list(self.fig_bokeh.keys())

    def generate_task_figures(self, plotly_dictionary):
        """
        Generates and displays the advanced figures for regression results. 
        """

        with st.expander("Advanced regression figures", expanded=True):
            fig_adv = PlotlyDesigner(
                colors=self.user_col).show_additional_regression_report(
                    self.class_properties,
                    self.out,
                    y_true=[
                        self.stacked_model.Validation[f'{p}'].values
                        for p in self.class_properties
                    ],
                    prop_cliffs=self.prop_cliff_dict,
                    leave_grp_out=self.leave_group_out,
                    fig_size=(self.x_width, self.y_height),
                    smiles=list(self.smiles_list.values),
                    apply_tick_transformation=self.apply_tick_transformation)
            selected_figures = st.multiselect(
                "Select Advanced regression figures to be shown",
                fig_adv,
                default=fig_adv,
                help="Select figures from the list",
            )

            plotly_dictionary, self.nb_figures = show_figures_in_columns(
                plotly_dictionary,
                fig_adv,
                selected_figures,
                nb_cols=self.nb_cols,
                use_container_width=not self.use_fixed_width,
                start_index=self.nb_figures)

        with st.expander("Turning a regressor into a classifier",
                         expanded=True):
            with st.form("Regression cutoff(s)"):
                min_ytrue = [
                    float(
                        np.nanmin(list(
                            self.stacked_model.Validation[p].values)))
                    for p in self.class_properties
                ]
                max_ytrue = [
                    float(
                        np.nanmax(list(
                            self.stacked_model.Validation[p].values)))
                    for p in self.class_properties
                ]
                if len(self.class_properties) == 1:
                    cutoff1 = st.number_input(
                        f'Cutoff for property {self.class_properties[0]}',
                        min_ytrue[0], max_ytrue[0],
                        float(
                            np.round(
                                min_ytrue[0] +
                                (max_ytrue[0] - min_ytrue[0]) / 2, 2)))
                    cutoff = [cutoff1]
                elif len(self.class_properties) == 2:
                    cutoff1 = st.number_input(
                        f'Cutoff for property {self.class_properties[0]}',
                        min_ytrue[0], max_ytrue[0],
                        float(
                            np.round(
                                min_ytrue[0] +
                                (max_ytrue[0] - min_ytrue[0]) / 2, 2)))
                    cutoff2 = st.number_input(
                        f'Cutoff for property {self.class_properties[1]}',
                        min_ytrue[1], max_ytrue[1],
                        float(
                            np.round(
                                min_ytrue[1] +
                                (max_ytrue[1] - min_ytrue[1]) / 2, 2)))
                    cutoff = [cutoff1, cutoff2]
                elif len(self.class_properties) == 3:
                    cutoff1 = st.number_input(
                        f'Cutoff for property {self.class_properties[0]}',
                        min_ytrue[0], max_ytrue[0],
                        float(
                            np.round(
                                min_ytrue[0] +
                                (max_ytrue[0] - min_ytrue[0]) / 2, 2)))
                    cutoff2 = st.number_input(
                        f'Cutoff for property {self.class_properties[1]}',
                        min_ytrue[1], max_ytrue[1],
                        float(
                            np.round(
                                min_ytrue[1] +
                                (max_ytrue[1] - min_ytrue[1]) / 2, 2)))
                    cutoff3 = st.number_input(
                        f'Cutoff for property {self.class_properties[2]}',
                        min_ytrue[2], max_ytrue[2],
                        float(
                            np.round(
                                min_ytrue[2] +
                                (max_ytrue[2] - min_ytrue[2]) / 2, 2)))
                    cutoff = [cutoff1, cutoff2, cutoff3]
                else:
                    auto_cutoffs = [
                        float(
                            np.round(
                                min_ytrue[ip] +
                                (max_ytrue[ip] - min_ytrue[ip]) / 2, 2))
                        for ip in range(len(self.class_properties))
                    ]
                    string_cutoffs = ','.join(
                        [str(val) for val in auto_cutoffs])
                    cutoff = split_in_floats(
                        st.text_input(
                            'provide (transformed) cutoff(s) for each property as list separated by ,  ',
                            string_cutoffs))

                if not isinstance(cutoff, list):
                    cutoff = [cutoff]
                good_class = st.selectbox(
                    'Positive class is less or more than cutoff?', ['>', '<'])
                Regenerate = st.form_submit_button("Update cutoff(s)")
            classification_figures = st.checkbox(
                "Generate classification figures for regression?", value=True)
            if classification_figures:
                fig_l = PlotlyDesigner(
                    colors=self.user_col).show_reg_cutoff_report(
                        self.class_properties,
                        self.out,
                        y_true=[
                            self.stacked_model.Validation[f'{p}'].values
                            for p in self.class_properties
                        ],
                        fig_size=(self.x_width, self.y_height),
                        cutoff=cutoff,
                        good_class=good_class,
                        smiles=list(self.smiles_list.values),
                        apply_tick_transformation=self.
                        apply_tick_transformation)
                clf_types = np.unique(
                    np.array([str(key)[0] for key in fig_l.keys()])).tolist()
                delete_keys = [
                    key for key in plotly_dictionary.keys()
                    if str(key)[0] in clf_types
                ]
                for key in delete_keys:
                    del plotly_dictionary[key]

                selected_cutoff_figures = st.multiselect(
                    "Select figures to be shown for classification from regression",
                    fig_l,
                    default=fig_l,
                    help="Select figures from the list",
                )

                plotly_dictionary, self.nb_figures = show_figures_in_columns(
                    plotly_dictionary,
                    fig_l,
                    selected_cutoff_figures,
                    nb_cols=self.nb_cols,
                    use_container_width=not self.use_fixed_width,
                    start_index=self.nb_figures)

                generate_bokeh = st.checkbox(
                    "Generate cutoff scatter plot with chemical structure",
                    value=True)
                if generate_bokeh:
                    st.warning(
                        'It can happen that streamlit gets stuck in a loop of generating the scatter plots with chemical structures. Uncheck the box for scatterplots with chemical structures if this happens.'
                    )
                    legend_pos = st.selectbox(
                        'Legend position for classification scatterplot', [
                            'bottom_right', 'bottom_left', 'top_right',
                            'top_left', "top_center", "center_right",
                            "center_left", "bottom_center", "center"
                        ])

                    fig_l = PlotlyDesigner(
                        colors=self.user_col).show_bokeh_cutoff_report(
                            self.class_properties,
                            self.out,
                            y_true=[
                                self.stacked_model.Validation[f'{p}'].values
                                for p in self.class_properties
                            ],
                            fig_size=(self.x_width, self.y_height),
                            cutoff=cutoff,
                            good_class=good_class,
                            smiles=list(self.smiles_list.values),
                            legend_pos=legend_pos,
                            apply_tick_transformation=self.
                            apply_tick_transformation)
                    selected_bokeh_cutoff_figures = st.multiselect(
                        "Select Figures to be shown for cutoff scatterplots with chemical structure",
                        fig_l,
                        default=fig_l,
                        help="Select figures from the list",
                    )
                    cols = st.columns(self.nb_cols)
                    for index, fig in enumerate(selected_bokeh_cutoff_figures):
                        cols[int(index % self.nb_cols)].bokeh_chart(
                            fig_l[fig],
                            use_container_width=not self.use_fixed_width)

        if self.prop_cliff_dict is not None:
            with st.expander("Show only property cliff results",
                             expanded=True):
                prop_cliffs_trimmed = {}
                prop_cliff_figures = {}
                for index, p in enumerate(self.class_properties):
                    property_smiles = self.stacked_model.Validation.stereo_SMILES.values[
                        self.prop_cliff_dict[p]]
                    if len(property_smiles) > 0:
                        out_prop_cliff = self.stacked_model.predict(
                            props=self.log10_props[index],
                            smiles=property_smiles,
                            compute_SD=True,
                            convert_log10=self.revert_log10)

                        prop_cliffs_trimmed[p] = np.arange(
                            len(self.prop_cliff_dict[p]))
                        fig_l = PlotlyDesigner(
                            colors=self.user_col).show_regression_report(
                                p,
                                out_prop_cliff,
                                y_true=[
                                    self.stacked_model.Validation[f'{p}'].
                                    values[self.prop_cliff_dict[p]]
                                ],
                                prop_cliffs=prop_cliffs_trimmed,
                                leave_grp_out=None,
                                bins=10,
                                fig_size=(self.x_width, self.y_height),
                                smiles=property_smiles,
                                apply_tick_transformation=self.
                                apply_tick_transformation)
                        prop_cliff_figures = {**prop_cliff_figures, **fig_l}

                    else:
                        st.warning(f'Property {p} has no activity cliffs')

                if prop_cliff_figures:
                    selected_figures = st.multiselect(
                        "Select Figures to be shown for property cliffs",
                        prop_cliff_figures,
                        default=prop_cliff_figures,
                        help="Select figures from the list",
                    )

                    _, self.nb_figures = show_figures_in_columns(
                        {},
                        prop_cliff_figures,
                        selected_figures,
                        nb_cols=self.nb_cols,
                        use_container_width=not self.use_fixed_width,
                        start_index=self.nb_figures)

        if self.leave_group_out is not None:
            with st.expander("Show only leave group out results",
                             expanded=True):
                ood_smiles = self.stacked_model.Validation.stereo_SMILES.values[
                    self.leave_group_out]

                out_ood = self.stacked_model.predict(
                    props=self.log10_props,
                    smiles=ood_smiles,
                    compute_SD=True,
                    convert_log10=self.revert_log10)

                leave_group_out_trim = np.arange(len(self.leave_group_out))
                fig_l = PlotlyDesigner(
                    colors=self.user_col).show_regression_report(
                        self.class_properties,
                        out_ood,
                        y_true=[
                            self.stacked_model.Validation[f'{p}'].values[
                                self.leave_group_out]
                            for p in self.class_properties
                        ],
                        prop_cliffs=None,
                        leave_grp_out=leave_group_out_trim,
                        bins=10,
                        fig_size=(self.x_width, self.y_height),
                        smiles=ood_smiles,
                        apply_tick_transformation=self.
                        apply_tick_transformation)

                selected_figures = st.multiselect(
                    "Select Figures to be shown for leave group out",
                    fig_l,
                    default=fig_l,
                    help="Select figures from the list",
                )

                _, self.nb_figures = show_figures_in_columns(
                    {},
                    fig_l,
                    selected_figures,
                    nb_cols=self.nb_cols,
                    use_container_width=not self.use_fixed_width,
                    start_index=self.nb_figures)
                
        self.good_class=good_class
        self.cutoff=cutoff
        return plotly_dictionary

    def generate_bokeh(self):
        """
        shows the generated bokeh figure. 
        """
        cols = st.columns(self.nb_cols)
        for index, fig in enumerate(self.selected_bokeh_reg_figures):
            cols[int(index % self.nb_cols)].bokeh_chart(
                self.fig_bokeh[fig],
                use_container_width=not self.use_fixed_width)
            cols[int(index % self.nb_cols)].caption(self.captions[index])