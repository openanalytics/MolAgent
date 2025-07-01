"""
**************************************************************************************
AutoMoL: Pipeline for automated machine learning for drug design.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
**************************************************************************************
Module that contains the functionality to retrieve user input for the training of JnJ AutoML stacking models

includes the methods:

load_data: reads the data as dataframe 
from_list_to_markdown: creates a string message from a given python list which when rendered in markdown translates to list of bullets 
from_list_to_enum_markdown: creates a string message from a given python list which when rendered in markdown translates to list of ordered points 
split_in_floats: splits a string into a list of floats with the outside list separated by ; and inside loop by , 
split_in_strings: splits a string into a list of strings with the outside list separated by ; and inside loop by , 
split_in_ints: splits a string into a list of ints with the outside list separated by ; and inside loop by , 
gen_sb_model: streamlit functionality to retrieve general model parameters in the sidebar (sb)
gen_sb_general_tasks: streamlit functionality to retrieve general task parameters in the sidebar (sb)
gen_sb_single_prop_classification: streamlit functionality to retrieve classification parameters in the case of one prop in the sidebar (sb)
gen_sb_adv_classification: streamlit functionality to retrieve advanced classification parameters in the sidebar (sb)
gen_sb_classification: streamlit functionality to retrieve classification parameters in the sidebar (sb)
gen_sb_regression: streamlit functionality to retrieve regression parameters in the sidebar (sb)
gen_sb_adv_regression: streamlit functionality to retrieve advanced regression parameters in the sidebar (sb)
gen_sb_validation: streamlit functionality to retrieve validation set parameters in the sidebar (sb)
gen_sb_ncv: streamlit functionality to retrieve nested cross-validation parameters in the sidebar (sb)
gen_sb_sw: streamlit functionality to retrieve sample_weights parameters in the sidebar (sb)
standardize_data: function to apply standardization
gen_mp_validation: function to create the validation set

and the classes:

ModelTrainer: generic wrapper of PropertyBuilder from property_prep.py in the automol package to print warnings in streamlit
ClassificationTrainer: classification specialization of ModelTrainer
RegressionTrainer: regression specialization of ModelTrainer
"""
from multiprocessing import cpu_count
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
from datetime import date
import os
from os import path

from automol.property_prep import add_stereo_smiles, make_category, add_rdkit_standardized_smiles
from automol.property_prep import ClassBuilder, PropertyTransformer
from automol.validation import stratified_validation, leave_grp_out_validation, mixed_validation
from automol.feature_generators import retrieve_default_offline_generators


# @st.cache(allow_output_mutation=True)
def load_data(df_file, sep=','):
    """reads the given files as dataframe

    :param df_file: file with the data
    :param sep: the separator used in the file
    """
    df = pd.read_csv(df_file, sep=sep, na_values=['NAN', '?', 'NaN'])
    return df


def from_list_to_markdown(string_list=None, header="Figure legends"):
    """Turn a given list of strings into a string. This string when parsed by MarkDown results in a list of bullets.

    :param string_list: the given list of strings
    :param header: list title, explanation, etc . Beneath this small sentence the list is shown in Markdown
    """
    assert len(string_list) > 0, 'length of given string list is less than 1'
    string_list = [
        t.replace('\33[1m', '**').replace('\33[0m', '**') for t in string_list]
    return f'**{header}:**  \n * '+" * ".join([f'{t}  \n' for t in string_list])


def from_list_to_enum_markdown(string_list=None, header="Figure legends"):
    """Turn a given list of strings into a string. This string when parsed by MarkDown results in a list of ordered points.

    :param string_list: the given list of strings
    :param header: list title, explanation, etc . Beneath this small sentence the list is shown in Markdown
    """
    assert len(string_list) > 0, 'length of given string list is less than 1'
    string_list = [
        t.replace('\33[1m', '**').replace('\33[0m', '**') for t in string_list]
    return f'{header}:  \n'+" ".join([f'{i+1}. {t}  \n' for i, t in enumerate(string_list)])


def split_in_floats(string_list):
    """Splits given string into list of list of floats with ; used for the outer splits and , for the inner splits
    """
    assert string_list is not None and len(
        string_list) > 0, "Invalid list of strings"
    if string_list[-1] == ";":
        string_list = string_list[:-1]
    outer_splits = string_list.split(";")
    if len(outer_splits) == 1:
        return [float(val) for val in outer_splits[0].split(",")]
    elif len(outer_splits) > 1:
        return [[float(val) for val in l[:-1].split(",")] if l[-1] == "," else [float(val) for val in l.split(",")] for l in outer_splits]
    else:
        return []


def split_in_strings(string_list):
    """Splits given string into list of list of strings with ; used for the outer splits and , for the inner splits
    """
    assert string_list is not None and len(
        string_list) > 0, "Invalid list of strings"
    if string_list[-1] == ";":
        string_list = string_list[:-1]
    outer_splits = string_list.split(";")
    if len(outer_splits) == 1:
        return [val for val in outer_splits[0].split(",")]
    elif len(outer_splits) > 1:
        return [[val for val in l[:-1].split(",")] if l[-1] == "," else [val for val in l.split(",")] for l in outer_splits]
    else:
        return []


def split_in_ints(string_list):
    """Splits given string into list of list of integers with ; used for the outer splits and , for the inner splits
    """
    assert string_list is not None and len(
        string_list) > 0, "Invalid list of strings"
    if string_list[-1] == ";":
        string_list = string_list[:-1]
    outer_splits = string_list.split(";")
    if len(outer_splits) == 1:
        return [int(val) for val in outer_splits[0].split(",")]
    elif len(outer_splits) > 1:
        return [[int(val) for val in l[:-1].split(",")] if l[-1] == "," else [int(val) for val in l.split(",")] for l in outer_splits]
    else:
        return []


def gen_sb_model():
    """
    Function that retrieves the data file and the required file names where to store the model and the output summary. 
    """
    st.sidebar.title("Parameters")
    st.sidebar.subheader("File parameters")
    save_model_to_file = st.sidebar.text_input(
        'Provide model name', f'model.pt', help="After training the model is saved in the direcotry of the app in the project environment using the given name. **String must include .pt**.")
    st.sidebar.subheader("Report generation")
    generate_pdf = st.sidebar.checkbox(f'Generate pdf summary?', value=True)
    save_output_to_file='output.pdf'
    title=''
    authors=''
    mails=''
    summary=''
    send_mail=False
    sender=''
    recipients=''
    if generate_pdf:
        save_output_to_file = st.sidebar.text_input('Provide output file name', f'output.pdf',
                                                    help="It is possible to save a pdf summary containing authors data, project summary, essential parameters and chosen output figures. The file is given this name. **String must include .pdf**")
        title = st.sidebar.text_input("Title","AutoMoL report")
        authors = st.sidebar.text_input("Authors","Joris Tavernier")
        mails = st.sidebar.text_input("Contact","joris.tavernier@openanalytics.eu")
        summary = st.sidebar.text_input("Summary","AutoMoL Prop1")
        
    st.sidebar.subheader("Data file")
    sep = st.sidebar.selectbox(
        'Datafile separator',
        [',', 'tab', ';'], help="Provide the separator of the data file, the columns for each line in the file are separated by this symbol. ")
    if sep == 'tab':
        sep = "\t"
    df_file = st.sidebar.file_uploader("Upload data file", type={
                                       "csv", "txt"}, help="drag or select your data file here. Only csv and txt files are accepted, the file is read using the separator given above. ")
    
    return save_model_to_file, save_output_to_file, sep, df_file, generate_pdf, title, authors, mails, summary,False, sender, recipients


def gen_sb_general_tasks(df):
    """
    Function that retrieves user input for the general parameters for the task at hand. Based on the given data file, the properties are selected, the column containing the SMILES, task, computation power, validation strategy and standardization strategy.

    :param df: Dataframe containing the data. 
    """
    smiles_column = st.selectbox(
        'Set the column containing the smiles',
        df.columns, help="Select from the columns in the data file, the column containing the smiles")
    properties = st.multiselect(
        "Select properties to be fitted from data file",
        df.columns,
        default=[],
        help="Select from the columns in the data file, the properties to be fitted",
    )
    task = st.selectbox(
        'Select the task',
        ['Regression', 'Classification'])
    computional_load = st.selectbox(
        'Select the Computational budget',
        ['cheap', 'moderate', 'expensive'], help="* **Cheap**: Blender optimized by gridsearch using [svr, lasso, kernelridge, pls ] for regression and [lr, SVC, knn] for classification **rough estimate 3000 samples: 1-5 min**  \n* **Moderate**: Blender optimized by randomized search with 60 iterations using [pls, svr, lasso, kernelridge, sgdr, rfr, lgbm] for regression and [lr, knn, SVC, sgdc, lgbm] for classification  **rough estimate 3000 samples: 30-45 min** \n* **Expensive**: Sklearn stacking optimized by randomized search with 100 iterations using [svr, lasso, kernelridge, sgdr, dtr, lgbm(4)] for regression and [lr, knn, SVC, sgdc, dtc, lgbm (4)] for classification **rough estimate 3000 samples: 10-12 hours** ")
        
    radius = st.slider('Provide radius for morgan fingerprints', 1, 5, 2)
    nbits = st.slider('Provide number of bits for morgan fingerprints', 1024, 8192, 2048,1024)
    
    feature_generators=retrieve_default_offline_generators(radius=radius,nbits=nbits)
    
    used_features = st.multiselect(
        "Select features to fit properties",
        [*feature_generators],
        default=[],
        help="Select used features",
    )
    
    strategy = st.selectbox(
        'Select the validation set strategy',
        ['stratified','leave group out','mixed'], help="Validation set strategy:\n * **mixed**: 30% property cliffs (similar compounds with large property difference), 30% complete clusters (leave group out) and rest stratified \n * **stratified**: validation set is evenly taken from generated clusters \n * **leave group out**: validation set consists of complete clusters")
    
    return smiles_column, properties, task, computional_load, strategy, 'RDKIT', feature_generators, used_features


def gen_sb_single_prop_classification(properties, df):
    """
    Function that display and retrieve the required user input for classification in the case of one property fit. This function will show a histrogram when non categorical data is used and lets the user decide the cutoff values more intuitively. 

    :param properties: list of properties, in this case of length 1
    :param df: dataframe with the data
    """
    assert len(
        properties) == 1, 'number of properties is not 1, wrong function is called'
    categorical = st.checkbox('Categorical property', value=True,
                              help="If the property is already placed in classes, leave this box checked. If the property is continuous uncheck the box. ")
    if not categorical:
        mask = np.isnan(df[properties[0]])
        nb_bins = int(st.number_input(f'Number of bins histogram', 10, 200, 15,
                      1, help='Select the number of bins to be used in the histogram. '))
        chart_data, bin_edges = np.histogram(
            df[properties[0]][~mask], bins=nb_bins)
        label_step = 1
        if nb_bins > 20:
            label_step = 2
        if nb_bins > 100:
            label_step = 5
        fig = go.Figure(data=[
            go.Bar(name='Prop. count',
                   x=np.round(bin_edges[1:], 2),
                   y=chart_data,
                   text=chart_data,
                   textposition='inside',
                   textfont_size=10,
                   textfont_color="black"
                   )])
        fig.update_layout(title=f'Histogram with {nb_bins} bins<br><sup>{properties[0]}</sup>',
                          title_font_size=18, height=600, width=600,
                          xaxis_title='Property bins',
                          yaxis_title=f'Counts',
                          xaxis_tickangle=-45,
                          xaxis=dict(
                              tickmode='array',
                              tickvals=np.round(bin_edges[1::label_step], 2),
                              ticktext=np.round(bin_edges[1::label_step], 2)
                          ),
                          legend=dict(
                              title_text='',
                              traceorder="normal",
                              font=dict(
                                  size=11,
                                  color="black"
                              )
                          )
                          )
        st.plotly_chart(fig, use_container_width=True)
        # st.bar_chart(pd.DataFrame({'count':chart_data,'x':np.round(bin_edges[1:],2)}).set_index('x'),use_container_width=True)

        nb_classes = [st.selectbox('provide number of classes', range(
            10)[2:], 0, help="Provide the number of classes for this property")]

        use_quantiles = st.checkbox('Use quantiles instead of absolute values', value=False,
                                    help="Check this box if instead of specific cutoffs, you want to provided relative quantiles to determine the classes. ")
        if use_quantiles:
            val_step = 0.8/(nb_classes[0])
            suggested_values = [float(np.round(0.1+(i+1)*val_step, 3))
                                for i in range(nb_classes[0]-1)]

            if nb_classes[0] == 2:
                v1 = st.number_input(
                    f'Provide class quantile for split between classes 0 and 1', 0.05, 0.95, suggested_values[0], 0.01)
                class_values = [v1]
            elif nb_classes[0] == 3:
                v1 = st.number_input(
                    f'Provide class quantile for split between classes 0 and 1', 0.05, 0.95, suggested_values[0], 0.01)
                v2 = st.number_input(f'Provide class quantile for split between classes 1 and 2',
                                     v1+0.05, 0.95, max(v1+0.05, suggested_values[1]), 0.01)
                class_values = [v1, v2]
            elif nb_classes[0] == 4:
                v1 = st.number_input(
                    f'Provide class quantile for split between classes 0 and 1', 0.05, 0.95, suggested_values[0], 0.05)
                v2 = st.number_input(f'Provide class quantile for split between classes 1 and 2',
                                     v1+0.05, 0.95, max(v1+0.05, suggested_values[1]), 0.01)
                v3 = st.number_input(f'Provide class quantile for split between classes 2 and 3',
                                     v2+0.05, 0.95, max(v2+0.05, suggested_values[2]), 0.01)
                class_values = [v1, v2, v3]
            elif nb_classes[0] == 5:
                v1 = st.number_input(
                    f'Provide class quantile for split between classes 0 and 1', 0.05, 0.95, suggested_values[0], 0.01)
                v2 = st.number_input(f'Provide class quantile for split between classes 1 and 2',
                                     v1+0.05, 0.95, max(v1+0.05, suggested_values[1]), 0.01)
                v3 = st.number_input(f'Provide class quantile for split between classes 2 and 3',
                                     v2+0.05, 0.95, max(v2+0.05, suggested_values[2]), 0.01)
                v4 = st.number_input(f'Provide class quantile for split between classes 3 and 4',
                                     v3+0.05, 0.95, max(v3+0.05, suggested_values[3]), 0.01)
                class_values = [v1, v2, v3, v4]
            else:
                class_values = split_in_floats(st.text_input(
                    'provide class quantiles as a list with values separated by ,', f'0.15,0.3,0.45,0.6,0.75', help="quantile determining the class"))
            #
        else:

            precision = float(st.selectbox('Input precision', [10 ** -i for i in range(
                8)], 2, help="this determines the precision of the values for the class cutoffs set below and the labelnames."))
            number_after = int(-np.log10(precision))
            max_val = float(
                np.round(np.max(df[properties[0]][~mask]), number_after))
            min_val = float(
                np.round(np.min(df[properties[0]][~mask]), number_after))
            val_step = 0.8/(nb_classes[0])
            suggested_values = [float(np.round(
                (0.1+(i+1)*val_step)*(max_val-min_val), 3)) for i in range(nb_classes[0]-1)]
            print(suggested_values)
            step = float(
                (np.max(df[properties[0]][~mask])-np.min(df[properties[0]][~mask]))/100)
            if nb_classes[0] == 2:
                v1 = st.number_input(f'Provide class value for split between classes 0 and 1',
                                     min_val, max_val, suggested_values[0], precision, format=f'%.{number_after}f')
                class_values = [v1]
            elif nb_classes[0] == 3:
                v1 = st.number_input(f'Provide class value for split between classes 0 and 1',
                                     min_val, max_val, suggested_values[0], precision, format=f'%.{number_after}f')
                v2 = st.number_input(f'Provide class value for split between classes 1 and 2', v1+precision,
                                     max_val, max(v1+0.05, suggested_values[1]), precision, format=f'%.{number_after}f')
                class_values = [v1, v2]
            elif nb_classes[0] == 4:
                v1 = st.number_input(f'Provide class value for split between classes 0 and 1',
                                     min_val, max_val, suggested_values[0], precision, format=f'%.{number_after}f')
                v2 = st.number_input(f'Provide class value for split between classes 1 and 2', v1+precision,
                                     max_val, max(v1+0.05, suggested_values[1]), precision, format=f'%.{number_after}f')
                v3 = st.number_input(f'Provide class value for split between classes 2 and 3', v2+precision,
                                     max_val, max(v2+0.05, suggested_values[2]), precision, format=f'%.{number_after}f')
                class_values = [v1, v2, v3]
            elif nb_classes[0] == 5:
                v1 = st.number_input(f'Provide class value for split between classes 0 and 1',
                                     min_val, max_val, suggested_values[0], precision, format=f'%.{number_after}f')
                v2 = st.number_input(f'Provide class value for split between classes 1 and 2', v1+precision,
                                     max_val, max(v1+0.05, suggested_values[1]), precision, format=f'%.{number_after}f')
                v3 = st.number_input(f'Provide class value for split between classes 2 and 3', v2+precision,
                                     max_val, max(v2+0.05, suggested_values[2]), precision, format=f'%.{number_after}f')
                v4 = st.number_input(f'Provide class value for split between classes 3 and 4', v3+precision,
                                     max_val, max(v3+0.05, suggested_values[3]), precision, format=f'%.{number_after}f')
                class_values = [v1, v2, v3, v4]
            else:
                class_values = split_in_floats(st.text_input(f'provide {nb_classes[0]-1} class cutoffs as a list separated by ,',
                                               f'{np.round(np.median(df[properties[0]][~mask])-6*step,number_after)},{np.round(np.median(df[properties[0]][~mask])-3*step,number_after)},{np.round(np.median(df[properties[0]][~mask]),number_after)},{np.round(np.median(df[properties[0]][~mask])+3*step,number_after)},{np.round(np.median(df[properties[0]][~mask])+6*step,number_after)}', help=f'list of values determining the classes. Provide {nb_classes[0]-1} values separated by a ,'))
            df_col = pd.DataFrame({"a": df[properties[0]][~mask]})
            ranges = [np.min(df[properties[0]][~mask])] + \
                class_values+[np.max(df[properties[0]][~mask])]
            grouped_df = df_col.groupby(
                pd.cut(df_col.a, ranges, include_lowest=True, precision=number_after)).count()
            samples_list = [f'**({np.format_float_positional(i.left,precision=number_after)},{np.format_float_positional(i.right,precision=number_after)}]**: {v}' for i,
                            v in zip(grouped_df.index, grouped_df.a)]
            samples_list[0] = samples_list[0].replace("(", "[")
            st.markdown(from_list_to_enum_markdown(
                string_list=samples_list, header="Samples within classes"))

    else:
        mask = df[properties[0]].isna()
        nb_classes = [len(df[properties[0]][~mask].unique())]
        class_values = None
        use_quantiles = False

    return categorical, nb_classes, use_quantiles, class_values


def gen_sb_adv_classification():
    """
    retrieves the classification score function
    """
    clf_scorer = st.selectbox(
        'Which scoring function',
        ['balanced_accuracy', 'accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples',
         'neg_log_loss', 'precision', 'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples', 'recall', 'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples', 'jaccard', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted'],
        help='Select from the given list the scikit-learn score function. This function is used to select the optimal parameters for each method in a nested cross-validation manner.')
    return clf_scorer


def gen_sb_adv_regressionclassification():
    """
    retrieves the regression score function for regressionclassification
    """
    regclf_scorer = st.selectbox(
        'Which scoring function',
        ['r2', 'explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error',
         'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance',
         'neg_mean_absolute_percentage_error'],
        help='Select from the given list the scikit-learn score function. This function is used to select the optimal parameters for each method in a nested cross-validation manner.')
    return regclf_scorer


def gen_sb_classification(properties, df):
    """
    retrieves general user input for classification, includes use of categorical properties, number of classes, use of quantiles, the values/quantiles to split the data in classes. 
    """
    categorical = st.checkbox(
        'Categorical properties', help="If the properties already contain the classes, check this box. If the properties are continuous leave unchecked. ")
    if not categorical:
        nb_classes = split_in_ints(st.text_input('provide number of classes as a list separated by , for each selected property', f'2,3',
                                   help="For each selected property set the number of classes, e.g. 2,3 means first property has 2 classes, second property 3."))
        use_quantiles = st.checkbox('Use quantiles instead of actual values',
                                    help="Check this box if instead of specific cutoffs, relative quantiles are used to determine the classes. ")
        class_values = split_in_floats(st.text_input('provide class thresholds/quantiles as a list of lists separated by ; for each selected property with the property values separated by , ', f'1;1,5',
                                       help="The number of values for each property = number of classes -1 for that property, each value separated by , and each property list separated by ;. For 2 properties with respectively two classes and three classes, provide 1 threshold for the first and 2 for the second, eg 1;1,5."))
    else:
        nb_classes = [len(df[p][~df[p].isna()].unique()) for p in properties]
        class_values = None
        use_quantiles = False

    return categorical, nb_classes, use_quantiles, class_values


def gen_sb_regression():
    """
    retrieves general user input for regression, includes setting log10 transformation, logit transformation and divide by 100. Additionally removal of outliers is indicated.  
    """
    use_log10 = st.checkbox('Use log10 transformed values',
                            help="transform all selected properties using log10 before training")
    use_logit = st.checkbox('Use logit transformed values',
                            help="transform all selected properties using logit function before training")
    percentages = st.checkbox('Divide property by 100 for logit',
                              help="Divide all selected properties by 100 before training")
    return use_log10, use_logit, percentages, False


def gen_sb_adv_regression():
    """
    Retrieves additional advanced regression user input, includes the scorer and the confidence for outlier removal. 
    """
    reg_scorer = st.selectbox(
        'Which scoring function',
        ['r2', 'explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error',
         'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance',
         'neg_mean_absolute_percentage_error'],
        help='Select from the given list the scikit-learn score function. This function is used to select the optimal parameters for each method in a nested cross-validation manner.')
    return reg_scorer


def gen_sb_validation(strategy='stratified',feature_generators={}):
    """
    Advanced user input for creating the validation set. Includes the choice of clustering algorithm and parameters and option to generate silhouette plots. 
    """
    val_clustering = st.selectbox(
        'Clustering algorithm for validation',
        ['Bottleneck', 'Butina', 'HierarchicalButina', 'Scaffold'], help="Choose between:  \n * **Bottleneck**: k-means++ on Bottleneck generated features  \n * **Butina**: Butina clustering with one addional loop through the data to assign samples to the closest leader  \n * **Hierarchical butina**: applies butina hierarchically with increasing threshold on the leaders  \n * **Scaffold**: applies MurckoScaffold clustering")
    val_include_chirality = None
    val_butina_cutoff = None
    val_km_groups = None
    val_km_features=None
    minority_nb = None
    test_size = st.slider('Ratio of test size ', 0.05, 0.95, 0.25,
                          help="Size of validation set divided by total data set size will approximately be this ratio")
    if val_clustering == "Scaffold":
        val_include_chirality = st.checkbox('include chirality for validation')
    if val_clustering == "HierarchicalButina":
        val_butina_cutoff = split_in_floats(st.text_input(
            'provide Butina cutoff(s) as list separated by ,  for validation', f'0.4,0.6'))
    if val_clustering == "Butina":
        val_butina_cutoff = [float(st.number_input(
            'provide Butina cutoff for validation', 0.05, 0.95, 0.6, 0.01))]
    if val_clustering == "Bottleneck":
        val_km_groups = st.slider(
            'Number of clusters for KMeans for validation', 5, 100, 30)
        val_km_features = st.multiselect(
            "Select features to to be used in kmeans++ in validation",
            [*feature_generators],
            default=['Bottleneck'],
            help="Select used features",
        )
    if strategy == 'mixed' or strategy == 'stratified':
        minority_nb = st.slider('Threshold of minority cluster samples', 1, 20, 5,
                                help='A cluster with an amount of samples less than this value is added to the collection of minority cluster and is merged into one Minorities cluster for stratified and mixed validation strategies.')
    plot_sihl = st.checkbox('Plot silhouette clustering score',value=True)
    sihl_map = 'jet'
    if plot_sihl:
        sihl_map = st.selectbox(
            'Colormap silhouette clustering score',
            ['jet', 'PiYG', 'viridis', 'YlGnBu', 'Blues', 'aggrnyl', 'agsunset', 'blackbody', 'bluered', 'blugrn', 'bluyl', 'brwnyl',
             'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'darkmint', 'electric', 'emrld', 'gnbu', 'greens', 'greys', 'hot',
             'inferno',  'magenta', 'magma', 'mint',  'orrd',  'oranges', 'oryel', 'peach', 'pinkyl', 'plasma',
             'plotly3', 'pubu',        'pubugn',      'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdpu',
             'redor', 'reds', 'sunset',  'sunsetdark', 'teal', 'tealgrn', 'turbo', 'ylgn',      'ylorbr',
             'ylorrd',     'algae',       'amp',         'deep',    'dense',       'gray',        'haline', 'ice',
             'matter',      'solar',       'speed',       'tempo',    'thermal',     'turbid',      'armyrose',
             'brbg',        'earth',       'fall',        'geyser',      'prgn',  'picnic',      'portland',    'puor',
             'rdgy',        'rdylbu',      'rdylgn',      'spectral',    'tealrose',    'temps',       'tropic',
             'balance',     'curl',        'delta',       'oxy',         'edge',    'hsv',         'icefire',
             'phase',       'twilight',    'mrybm',       'mygbm'])

    return val_clustering, test_size, val_include_chirality, val_butina_cutoff, val_km_groups, val_km_features, plot_sihl, sihl_map, minority_nb


def gen_sb_ncv(feature_generators={}):
    """
    retrieves user input for nested cross-validation using streamlit functions
    """
    cv_clustering = st.selectbox(
        'Clustering algorithm for cross-validation',
        ['Bottleneck', 'Butina', 'HierarchicalButina', 'Scaffold'], help="Choose between:  \n * **Bottleneck**: k-means++ on Bottleneck generated features  \n * **Butina**: Butina clustering with one addional loop through the data to assign samples to the closest leader  \n * **Hierarchical butina**: applies butina hierarchically with increasing threshold on the leaders  \n * **Scaffold**: applies MurckoScaffold clustering")
    include_chirality = None
    butina_cutoff = None
    km_groups = None
    if cv_clustering == "Scaffold":
        include_chirality = st.checkbox(
            'include chirality for cross-validation')
    if cv_clustering == "HierarchicalButina":
        butina_cutoff = split_in_floats(st.text_input(
            'provide Butina_cutoff(s) as list separated by ,  for cross-validation', f'0.4,0.6'))
    if cv_clustering == "Butina":
        butina_cutoff = [float(st.number_input(
            'provide Butina cutoff for nested cross-validation', 0.05, 0.95, 0.6, 0.01))]
    if cv_clustering == "Bottleneck":
        km_groups = st.slider(
            'Number of clusters for KMeans  for cross-validation', 5, 100, 25)
        cv_km_features = st.multiselect(
            "Select features to be used in kmeans++ during nested cross-validation",
            [*feature_generators],
            default=['Bottleneck'],
            help="Select used features",
        )
    cross_val_split = st.selectbox(
        'Cross-validation strategy',
        ['GKF', 'LGO', 'SKF'], help="GKF=Group k-fold (**recommended!**), LGO= Leave one group out, SKF = stratified k-fold (only for classification)")
    outer_folds = st.slider('Number of outer folds', 2, 20, 4)
    random_state = st.slider('Random state', 1, 100, 5)
    random_state_list = split_in_ints(st.text_input(
        'provide random states for methods as list separated by , ', f'1,7,42,55,3'))
    n_jobs = st.slider('n_jobs', -1, 100, min(cpu_count(), 32))
    return cv_clustering, include_chirality, butina_cutoff, km_groups,cv_km_features, cross_val_split, outer_folds, random_state, random_state_list, n_jobs


def gen_sb_sw():
    """
    retrieves user input for sample weights using streamlit functions
    """
    use_weighted_samples = st.checkbox('Use sample weights')
    weighted_samples_index = None
    selected_sample_weights = None
    if use_weighted_samples:
        weighted_samples_index = st.text_input('provide values for weighted samples selection as list separated by ,  for each property. Task dependent!', f'0,1',
                                               help="This depends on the specific kind task  \n * **Classification**:  \n \t* in the case of non-categorical data, use index of the class in the range, this increases with the value, e.g. for threshold 1, index 0 is the class <=1  and index 1 is the class >1.  \n \t* for categorical data provide the label of the class.  \n* **Regression**: set cutoff with bigger of smaller indication, e.g. >1 for bigger than 1 and <-0.5 for smaller than -0.5 ")
        selected_sample_weights = split_in_floats(st.text_input('provide the weight for the given samples above as a list separated by ,  for each property',
                                                  f'10,10', help="The samples indicated above are set to this weight, the other samples have weight 1."))
    return use_weighted_samples, weighted_samples_index, selected_sample_weights


def standardize_data(df, smiles_column, verbose, properties, standard_smiles_column='stereo_SMILES', standardization='adme_il17'):
    """
    Standardizes the input smiles, this is not a validation check. 

    :param df: dataframe containing the smiles and properties
    :param smiles_column: the column name of the column in the dataframe containing the smiles
    :param verbose: boolean to indicate verbose output
    :param properties: list of properties that will be fitted
    :param standard_smiles_column: the name for the column which will contain the standardized SMILES
    :param standardization: type of standardization to switch between Chemaxon and adme_il17
    """
    start_time = time.time()
    with st.spinner('Preparing data'):
        add_rdkit_standardized_smiles(
                df, smiles_column, verbose=verbose, outname=standard_smiles_column)
        #df= pd.read_csv(file_name, na_values = ['NAN', '?','NaN'])
        df.dropna(inplace=True, subset=[standard_smiles_column])
        df.dropna(inplace=True, how='all', subset=properties)
        df.reset_index(drop=True, inplace=True)
        str(datetime.timedelta(seconds=time.time() - start_time))
    st.success(
        f'Finished preparing data in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')
    #if path.exists("biosignature.debug.log"): os.remove("biosignature.debug.log")
    return df

from automol.stacking_util import get_clustering_algorithm

def gen_mp_validation(df, train_properties, model_trainer, strategy, categorical, stacked_model, df_smiles, standard_smiles_column, test_size, val_clustering,
                      val_km_groups,val_used_features_km, val_butina_cutoff, val_include_chirality, verbose, random_state, sihl_map, plot_sihl, prop_cliff_butina_th=None, minority_nb=5):
    """
    streamlit function to display validation set progress and warnings.

    :param df: dataframe containing the different properties and smiles
    :param model_trainer: child of ModelTrainer implementing task specific property building methods
    :param strategy: validation strategy, [mixed,stratified,leave_group_out]
    :param categorical: boolean to indicate use of categorical properties in the case of classification
    :param stacked_model: stacking model used to fit the properties
    :param df_smiles: list of SMILES
    :param standard_smiles_column: name of the column containing the SMILES
    :param test_size: ratio determining the size of the validation set
    :paramval_clustering: Clustering algorithms used for validation set
    :param val_km_groups: number of k for kmeans++
    :param val_butina_cutoff: list of Butina cutoff values
    :param val_include_chiralty: include chiralty for MurckoScaffold
    :param verbose: boolean to set verbosity
    :param random_state: integer seed for randomness
    :param sihl_map: colormap for silhouette scores
    :param plot_sihl: boolean to indicate plotting of silhouette scores.
    """
    leave_group_out = None
    prop_cliff_dict = None
    if prop_cliff_butina_th is None:
        prop_cliff_butina_th = 0.45
        
    c_algo=get_clustering_algorithm(clustering=val_clustering,
                             n_clusters=val_km_groups,
                             cutoff=val_butina_cutoff,
                             include_chirality=val_include_chirality,
                             verbose=verbose,
                             random_state=random_state,
                             feature_generators=stacked_model.feature_generators,
                             used_features=val_used_features_km)
    start_time = time.time()
    with st.spinner(f'Clustering data for Validation set'):
        if strategy == 'mixed':
            prop_cliff_butina_th = 0.45
            rel_prop_cliff = 0.5
            # stratified is essentially the remaining number of samples left to reach the desired amount of test samples set by test_size ratio
            #    and is therefore not actually used
            mix_coef_dict = {'prop_cliffs': 0.3,
                             'leave_group_out': 0.3, 'stratified': 0.4}
            if categorical:
                st.warning(
                    'Property cliffs with categorical properties: similar compounds with different class are considered as an property cliff ')

            Train, Validation, leave_group_out, prop_cliff_dict, sihl_dict = mixed_validation(df_orig=df, properties=model_trainer.properties, stacked_model=stacked_model,
                                                                                              standard_smiles_column=standard_smiles_column,
                                                                                              prop_cliff_cut=rel_prop_cliff, prop_cliff_butina=prop_cliff_butina_th, test_size=test_size, clustering=val_clustering,
                                                                                              n_clusters=val_km_groups, cutoff=val_butina_cutoff, include_chirality=val_include_chirality,
                                                                                              verbose=verbose, random_state=random_state, mix_dict=mix_coef_dict, plot_silhouette=True, cmap=sihl_map, categorical_data=categorical, minority_nb=minority_nb,clustering_algorithm=c_algo)

            stacked_model.leave_group_out = leave_group_out
            stacked_model.prop_cliff_dict = prop_cliff_dict
        elif strategy == 'stratified':
            Train, Validation, sihl_dict = stratified_validation(df, train_properties, stacked_model, df_smiles, test_size=test_size, clustering=val_clustering,
                                                                 n_clusters=val_km_groups, cutoff=val_butina_cutoff, include_chirality=val_include_chirality,
                                                                 verbose=verbose, random_state=random_state, plot_silhouette=True, cmap=sihl_map, clustering_algorithm=c_algo)
        else:
            Train, Validation, sihl_dict = leave_grp_out_validation(df, train_properties, stacked_model, df_smiles, test_size=test_size, clustering=val_clustering,
                                                                    n_clusters=val_km_groups, cutoff=val_butina_cutoff, include_chirality=val_include_chirality,
                                                                    verbose=verbose, random_state=random_state, plot_silhouette=True, cmap=sihl_map, clustering_algorithm=c_algo)
            leave_group_out = np.arange(len(Validation))
            stacked_model.leave_group_out = leave_group_out

    st.success(
        f'Finished clustering data for validation set in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')
    if plot_sihl:
        with st.expander("Silhouette scores for clustering", expanded=True):
            st.info(
                'Silhouette score is a metric to define cluster quality, for more information see [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) and [Wikipedia] (https://en.wikipedia.org/wiki/Silhouette_(clustering)#:~:text=The%20silhouette%20value%20is%20a,poorly%20matched%20to%20neighboring%20clusters)')
            nb_cols = 2
            cols = st.columns(nb_cols)
            for index, fig in enumerate(sihl_dict):
                cols[int(index % nb_cols)].plotly_chart(
                    sihl_dict[fig], use_container_width=True)

    # placing the validation and training set inside the model generator
    stacked_model.Validation = Validation
    stacked_model.Train = Train
    stacked_model.smiles = standard_smiles_column  # set col smiles name
    return stacked_model


class ModelTrainer:
    """
    Abstract wrapper to use PropertyBuilder and its specializations within streamlit  
    """

    def __init__(self):
        pass

    def check_properties(self, df, min_num_prop=100):
        self.prop_builder.check_properties(df, min_num_prop=min_num_prop)
        self.print_warnings_streamlit()

    def print_warnings_streamlit(self):
        msgs = []
        msgs = self.prop_builder.retrieve_and_clear_warnings(msgs)
        for w in msgs:
            st.warning(w)

    def generate_train_properties(self, df):
        df, train_properties = self.prop_builder.generate_train_properties(df)
        self.print_warnings_streamlit()
        return df, train_properties

    def generate_sample_weights(self, df, weighted_samples_index, selected_sample_weights):
        if len(self.properties) == 1:
            if not isinstance(weighted_samples_index, list):
                weighted_samples_index = [weighted_samples_index]
            if not isinstance(selected_sample_weights, list):
                selected_sample_weights = [selected_sample_weights]
        weighted_samples_index_dict = {}
        selected_sample_weights_dict = {}
        for ip, p in enumerate(self.train_properties):
            weighted_samples_index_dict[p] = weighted_samples_index[ip]
            selected_sample_weights_dict[p] = selected_sample_weights[ip]
        print(weighted_samples_index_dict)
        df = self.prop_builder.generate_sample_weights(
            df, weighted_samples_index_dict, selected_sample_weights_dict)
        self.print_warnings_streamlit()
        return df


class ClassifierTrainer(ModelTrainer):

    def __init__(self, properties, nb_classes, class_values, scorer, categorical, use_quantiles, prefix='Class', min_allowed_class_samples=30):
        if len(properties) == 1:
            if not isinstance(nb_classes, list):
                nb_classes = [nb_classes]
            if not isinstance(class_values, list):
                class_values = [class_values]
            if not isinstance(class_values[0], list):
                class_values = [class_values]

        self.scorer = scorer
        self.labelnames = {}
        self.index_map = {}
        self.properties = properties
        self.prop_builder = ClassBuilder(properties=properties, nb_classes=nb_classes, class_values=class_values,
                                         categorical=categorical, use_quantiles=use_quantiles,
                                         prefix=prefix, min_allowed_class_samples=min_allowed_class_samples, verbose=False, track_warnings=True)

    def generate_train_properties(self, df):
        df, train_properties = super().generate_train_properties(df)
        self.train_properties = train_properties
        self.labelnames = self.prop_builder.labelnames
        st.write(f'Class value to label mapping: {str(self.labelnames)}')
        return df, train_properties


class RegressionTrainer(ModelTrainer):

    def __init__(self, properties, scorer, remove_outliers, confidence, use_log10, use_logit, percentages, standard_smiles_column):
        self.scorer = scorer
        self.prop_builder = PropertyTransformer(
            properties, False, confidence, use_log10, use_logit, percentages, standard_smiles_column, track_warnings=True)
        self.labelnames = None

    def generate_train_properties(self, df):
        df, train_properties = super().generate_train_properties(df)
        self.properties = train_properties
        self.train_properties = train_properties
        return df, train_properties