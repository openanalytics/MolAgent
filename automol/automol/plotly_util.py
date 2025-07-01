"""functionality to create the different plotly/bokeh figures and the automated report generation.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""


import os
import os.path
from os import path
import inspect
import re 
from pathlib import Path
from pkg_resources import resource_filename

from  matplotlib import pyplot as plt
import matplotlib.cm as cm

import numpy as np

import datetime
from datetime import date

import pdfkit
import json

import jinja2
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader

from .stat_plotly_util import *
from .version import __version__
from .model import *
#from .util import *

## titles of the figures in the pdf report
figure_titles={
    'A':'Classification report.',
    'B':'Confusion matrix.',
    'C':'ROC curve.',
    'D':'Scatterplot with MAE folds.',
    'E':'Scatterplot with moving average error.',
    'F':'Barplot for regression.',
    'G':'Performance metrics for varying threshold.',
    'H':'Precision, recall and F1-score curves.',
    'I':'Predicted probability barplot.',
    'J':'Scatterplot for regression with cutoff.',
    'K':'Hit enrichment curve.',
    'L':'Precision-Recall curve.',
    'M':'Reliability diagram.',
    'Z':'Unknown type.'
}
## details of the figures in the pdf report
figure_details={
    'A':'Precision, recall and F1-score for the different classes.',
    'B':'The confusion matrix for the different classes (relative precentages are given for the global matrix as well as for each row)',
    'C':'The Receiver Operating Characteristics curve for the different classes. The area under the curve is given in the legend.',
    'D':'Scatterplot of true values versus predicted values. The width of the blue bands is determined by one or two times the mean absolute error (MAE). In the legend you can find the percentage of data samples within these bands.',
    'E':'Scatterplot with a distinction between the different kinds of samples within the validation set (stratified, leave group out or property cliff). The centered moving average of the error is given as a trendline (default window = 50 samples). This trendline provides an indication where the model over- or underestimates the predictions.',
    'F':'The samples are binned based on their Predicted value. The samples are also grouped based on the absolute error of these predicted values. The barplot shows the relative counts for each group of predicted values for each group of absolute error. This plot provides insight into which group of predicted value has the largest absolute errors.',
    'G':'Accuracy, recall, precision and positive ratio are given as a function of a varying threshold.',
    'H':'The precision, recall and F1-score for a varying threshold for each class. The optimal thresholds for F1-score and Youden\'s J-statistic are given as well as their corresponding performance metric',
    'I':'Samples are binned based on their predicted probability for each class. The barplot shows the relative counts of each group with respect to their true class. This barplot provides insight into which classes have high predicted probabilities.',
    'J':'Scatterplot of true values versus predicted values. Samples are colored based on their classification result, e.g. True Positive, True Negative, False Positive or False Negative. The gray dotted lines are the cutoff lines and determine the classification',
    'K':'The fraction of positives identified as a function of the fraction of compounds tested for a varying threshold. The area under the curve is provided in the legend.',
    'L':'Precision as a function of recall for a varying threshold. High precision means a high proportion of positive prediction were accurate. High recall means that a high proportion of positives were predicted to be positive.',
    'M':'Observed ratio within group with respect to predictive probability group for each class. The number of samples in each group is annotated next to the markers and the Expected and Maximum Calibration Error is given in the legend. This figure provides insight in how well the predicted probability can be interpreted as uncertainty. ',
    'Z':'Unknown type'
}
## annotations of the figures in the pdf report
figure_annotations={
    'A':'A_',
    'B':'B_',
    'C':'C_',
    'D':'D_',
    'E':'E_',
    'F':'F_',
    'G':'G_',
    'H':'H_',
    'I':'I_',
    'J':'J_',
    'K':'K_',
    'L':'L_',
    'M':'M_',
}

import json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def save_result_dictionary_to_json(result_dict:dict=None, json_file:str='json_results.json'):
    with open( json_file, 'w' ) as f:
        json.dump( result_dict, f, cls=NpEncoder )

def get_figures_from_types(plotly_dictionary,properties,types):
    """ 
    returns selected figures based on types ordered per property
    
    Args:
         plotly_dictionary: dictionary containing the different figures
         properties: list of properties
         types: types of figures
    
    Returns: 
        list of figure keys from the plotly_dictionary ordered per property
    """
    selected_figures=[]
    for p in properties:
        for t in types:
            for fig_key in plotly_dictionary.keys():
                if figure_annotations[t]+p+'<metric>' in fig_key or figure_annotations[t]+p+'_cls_' in fig_key:
                    if fig_key.startswith(t):
                        selected_figures.append(fig_key)
    return selected_figures

def create_figure_captions(figures,start_index=1):
    """
    returns list of captions, list of type details in string format and list of types for the given list of figures
    
    Args:
         figures: list of figures
         start_index: starting index of the figures
    
    Returns: 
        tuple of (captions, figure details and types) in printing str format
    """
    types={}
    captions=[]
    for fig_index,fig in enumerate(figures):
        type_found=False
        for key in figure_annotations:
            if fig.startswith(figure_annotations[key]):
                type_found=True
                if key in types:
                    types[key]=types[key]+[fig_index+start_index]
                else: 
                    types[key]=[fig_index+start_index]
                name=fig[len(figure_annotations[key]):]
                prop,metrics=name.split("<metric>", 1)
                captions.append(f'\33[1m Figure {fig_index+start_index}. {figure_titles[key]}\33[0m Property: {prop}. {metrics}')
        if not type_found:
            types['Z']=True
            captions.append(f'Type Z: {fig}')
    return captions, [f'\33[1mFigures {types[t]} (type {t}). {figure_titles[t]}\33[0m {figure_details[t]}' if len(types[t]) > 1 else 
                      f'\33[1mFigure {types[t][0]} (type {t}). {figure_titles[t]}\33[0m {figure_details[t]}' for t in types.keys()], [t for t in types.keys()]
    
        
def generate_report(plotly_dictionary,selected_figures,captions,types,title='AutoML report Title', authors='Joris Tavernier', mails='joris.tavernier@openanalytics.eu',summary='Project details', model_param={'empty_model':True}, model_tostr_list=[" "],properties=[' '], template_file=None,nb_columns=3,remove_titles=True, wkhtmltopdf_options=None):
    """
    renders the given list of figures and content variables as html which is converted to pdf
    
    Args:
         plotly_dictionary: dictionary of plotly figures
         selected_figures: selected figures
         captions: captions of the figures
         types: types of the figures
         title: title of the report
         authors: authors of the report
         mails: e-mails of the authors
         summary: summary of the report
         model_param: model training parameters as dictionary
         model_tostr_list: list of strs for models
         properties: properties 
         template_file: the file of used as template [=None], the default uses the internal one
         nb_columns: number of columns for the figures
         remove_titles: boolean to remove the titles from the figures for the report
         wkhtmltopdf_options: additional options of the systemcall to wkhtmltopdf [=None]
    
    Returns: 
        pdf_data 
    """
    html_strings=[]
    for m_to_str in model_tostr_list:
        html_strings.append(m_to_str.replace('\33[1m' ,'<b>').replace('\33[0m' ,'</b>').replace("\n","<br>"))
    
    captions=[capti.replace('\33[1m' ,'<b>').replace('\33[0m' ,'</b>') for capti in captions]
    types=[re.sub(' \(type [A-Z]\)', '', t.replace('\33[1m' ,'<b>').replace('\33[0m' ,'</b>')) for t in types]
    
    d='/'.join((os.path.abspath(inspect.getfile(MultiheadAttention))).split("/")[:-1])
    import automol.pdf_template as pdf_template_pkg
    template_dir = str(Path(pdf_template_pkg.__file__).parent)
    work_dir=os.getcwd()
    
    figures=[f'{work_dir}/fig{index}.png' for index,fig in enumerate(selected_figures)]
    for index,fig in enumerate(selected_figures):
        if remove_titles:
            plotly_dictionary[fig].update_layout(title='').write_image(figures[index])
        else:
            plotly_dictionary[fig].write_image(figures[index])
    
    jinja2.filters.FILTERS['zip'] = zip
    if template_file is None:
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape())
        template_file=f'summary.html'
    else:
        env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
    
    #env.filters['zip']= zip
    template = env.get_template(template_file)

    html = template.render(
        title=title,
        authors=authors,
        mails=mails,
        AutoMLv=__version__,
        summary=summary,
        types=types,
        figures=figures,
        captions=captions,
        nb_columns=nb_columns,
        properties=properties,
        html_strings=html_strings,
        logo=resource_filename('automol.pdf_template', 'logo1.jpg'),
        column_width=np.round(100.0/nb_columns,2),
        model_param=json.dumps(model_param, indent = 4) ,#str(model_param),
        date=date.today().strftime("%B %d, %Y")
    )
    #pdfkit.from_string(html, 'filename.pdf', options=options)
    if wkhtmltopdf_options is None: 
        wkhtmltopdf_options = {
            "enable-local-file-access": ""
        }
    pdf = pdfkit.from_string(html, False, options=wkhtmltopdf_options)
    
    for fig_file in figures:
        os.remove(fig_file)
        
    return pdf

class PlotlyDesigner:
    """
    methods to generate all the different plotly figures and save them in dictionaries
    """
    def __init__(self,line_width=2,fig_per_row=3,colors=None):
        """
        Initialization to set line width, colors and nb of figs per row for the figures in the generated results
        
        Args:
             line_width:
             fig_per_row: nb of figs per row for matplotlib subfigures configuration
             colors: list of color strings
        """
        if colors is None:
            ## colors used in the figures
            self.colors=["darkorange","darkred","darkgreen",'deepskyblue','darkviolet','orangered','yellowgreen']
        else: 
            ## colors used in the figures
            self.colors=colors
        ## number of figures per row
        self.fig_per_row=fig_per_row
        ## line width
        self.line_width=line_width
            
            
    def show_classification_report(self,properties,out,y_true,labelnames,cmap='PiYG',fig_size=(800,800),results_dict=None ):
        """ 
        prints the classification results: scikit classification report, ROC-AUC plot, confusion matrix and Recall vs threshold plot
        
        Args:
             properties: property names list present in the output
             out: model output containing the predictions
             y_true: the array of the true values
             labelnames: dictionary of class names 
             cmap: colormap 
             fig_size: figure size as tuple
        
        Returns: 
            Youden_dict: youdens-j values and dictionary of figures
        """
        if not isinstance(properties, (list)):
            properties=[properties]
        if not isinstance(cmap, (list)):
            cmap=[cmap,cmap]
        
        nb_props=len(properties)

        Youden_dict={}
        figures={}
        for ip,p in enumerate(properties):
            y_pred=out[f'predicted_{p}']
            nb_classes=len(list(labelnames[p].values()))
            y_pred_proba=np.concatenate(tuple(np.expand_dims(out[f'predicted_proba_{p}_class_{labelnames[p][c]}'],axis=1) for c in range(nb_classes)), axis=1)
            #y_pred_proba=out[f'predicted_proba_{p}']
            mask=np.isnan(y_true[ip])| np.isnan(y_pred.astype(float))

            fig_clf_mm,metric=plotly_clf_matrix(y_pred[~mask].astype(int), y_true[ip][~mask],title=p,labels=[index for index in range(y_pred_proba.shape[1])], prop_labels=labelnames[p], cmap=cmap[0],fig_size=fig_size, results_dict=results_dict )
            fig_key=figure_annotations['A']
            figures[f'{fig_key}{p}<metric>({metric})']=fig_clf_mm
            
            fig_clf_cm,metric=plotly_clf_confusion(y_pred[~mask].astype(int), y_true[ip][~mask],title=p,prop_labels=labelnames[p],cmap=cmap[1],fig_size=fig_size, results_dict=results_dict  )
            fig_key=figure_annotations['B']
            figures[f'{fig_key}{p}<metric>({metric})']=fig_clf_cm

            Youden_th, Youden_val, fig_clf_ar,m1 = plotly_clf_auc(y_pred_proba[~mask,:], y_true[ip][~mask],title=p, colors=self.colors, line_width=self.line_width, prop_labels=labelnames[p],fig_size=fig_size, results_dict=results_dict  )
            Youden_dict[p]=Youden_th
            Youden_dict[f'val_{p}']=Youden_val
            fig_key=figure_annotations['C']
            figures[f'{fig_key}{p}<metric>({m1})']=fig_clf_ar
        return Youden_dict,figures
    
    def show_additional_classification_report(self,properties,out,y_true,labelnames,cmap='PiYG',fig_size=(800,800), results_dict=None ):
        """
        prints the classification results: scikit classification report, ROC-AUC plot, confusion matrix and Recall vs threshold plot
        
        Args:
             properties: property names list present in the output
             out: model output containing the predictions
             y_true: the array of the true values
             labelnames: dictionary of class names 
             cmap: colormap 
             fig_size: figure size as tuple
        
        Returns: 
            dictionary of figures
        """
        if not isinstance(properties, (list)):
            properties=[properties]
        if not isinstance(cmap, (list)):
            cmap=[cmap,cmap]
        
        nb_props=len(properties)

        Youden_dict={}
        figures={}
        for ip,p in enumerate(properties):
            y_pred=out[f'predicted_{p}']
            nb_classes=len(list(labelnames[p].values()))
            y_pred_proba=np.concatenate(tuple(np.expand_dims(out[f'predicted_proba_{p}_class_{labelnames[p][c]}'],axis=1) for c in range(nb_classes)), axis=1)
            #y_pred_proba=out[f'predicted_proba_{p}']
            mask=np.isnan(y_true[ip])| np.isnan(y_pred.astype(float))
            
            fig_clf_prc,m1 = plotly_clf_prc(y_pred_proba[~mask,:], y_true[ip][~mask],title=p, colors=self.colors, line_width=self.line_width, prop_labels=labelnames[p],fig_size=fig_size, results_dict=results_dict  )
            fig_key=figure_annotations['L']
            figures[f'{fig_key}{p}<metric>({m1})']=fig_clf_prc
            
            fig_clf_calbc,m1 = plotly_clf_calbc(y_pred_proba[~mask,:], y_true[ip][~mask],title=p, colors=self.colors, line_width=self.line_width, prop_labels=labelnames[p],fig_size=fig_size, results_dict=results_dict  )
            fig_key=figure_annotations['M']
            figures[f'{fig_key}{p}<metric>({m1})']=fig_clf_calbc
            

            fig_clf_er,metrics = plotly_enrichment_clf(y_pred_proba[~mask,:], y_true[ip][~mask],title=f'{p}', prop_labels=labelnames[p], colors=self.colors, line_width=self.line_width,fig_size=fig_size, results_dict=results_dict  )
            fig_key=figure_annotations['K']
            figures[f'{fig_key}{p}<metric>({metrics})']=fig_clf_er            
        
        return figures

    
    
    def show_clf_threshold_report(self,properties,out,y_true,labelnames,youden_dict=None,fig_size=(800,800), results_dict=None ):
        """ 
        prints the classification results: scikit classification report, ROC-AUC plot, confusion matrix and Recall vs threshold plot
        
        Args:
             properties: property names list present in the output
             out: model output containing the predictions
             y_true: the array of the true values
             labelnames: dictionary of class names 
             youden_dict: dictionary containing the jouden-j values
             fig_size: figure size as tuple
        
        Returns: 
            F1_dict: optimized F1 dictionary and dictionary of figures
        """
        if not isinstance(properties, (list)):
            properties=[properties]
        
        nb_props=len(properties)
        figures={}
        F1_dict={}
        i=0
        for ip,p in enumerate(properties):
            print("####################################################")
            print(p)
            y_pred=out[f'predicted_{p}']
            nb_classes=len(list(labelnames[p].values()))
            y_pred_proba=np.concatenate(tuple(np.expand_dims(out[f'predicted_proba_{p}_class_{labelnames[p][c]}'],axis=1) for c in range(nb_classes)), axis=1)
            #y_pred_proba=out[f'predicted_proba_{p}']
            mask=np.isnan(y_true[ip])| np.isnan(y_pred.astype(float))
            
            F1_th=[]
            F1_val=[]
            for c in range(nb_classes):
                F1_th_c, F1_val_c, fig_clf_f1 = plotly_clf_f1(y_pred_proba[~mask,:], y_true[ip][~mask],title=p,colors=self.colors,line_width=self.line_width,c=c, labelname=labelnames[p][c], youden=youden_dict[p][c], youden_val=youden_dict[f'val_{p}'][c], fig_size=fig_size , results_dict=results_dict )
                F1_th.append(F1_th_c)
                F1_val.append(F1_val_c)
                fig_key=figure_annotations['H']
                figures[f'{fig_key}{p}_cls_{labelnames[p][c]}<metric>']=fig_clf_f1
            for c in range(nb_classes):
                plot_bar={}
                plot_bar[f'Pred. Prob. {labelnames[p][c]}_{p}']=y_pred_proba[~mask,c]
                plot_bar[f'categories_{p}']=y_true[ip][~mask]
                fig_clf_pb = plotly_confusion_bars_from_categories(pd.DataFrame.from_dict(plot_bar),pro1= f'Pred. Prob. {labelnames[p][c]}_{p}',pro2= f'categories_{p}', 
                        bins1=np.linspace(0,1,11)[1:-1],
                        title=f'Pred. prob. {labelnames[p][c]} for {p}',
                        leg_title=f'Pred. Prob. {labelnames[p][c]}',x_title=f'True class',
                        labelnames=labelnames[p],
                        fig_size=fig_size, results_dict=results_dict 
                )
                fig_key=figure_annotations['I']
                figures[f'{fig_key}{p}_cls_{labelnames[p][c]}<metric>']=fig_clf_pb
                
            F1_dict[p]=F1_th
            F1_dict[f'f1_{p}']=F1_val
        return F1_dict,figures

        
    def show_reg_cutoff_report(self,properties,out,y_true,fig_size=(600,600),cutoff=0.5, good_class= '>',smiles=None,apply_tick_transformation=False, results_dict=None ):
        """ 
        prints the classification results: scikit classification report, ROC-AUC plot, confusion matrix and Recall vs threshold plot
        
        Args:
             properties: property names list present in the output
             out: model output containing the predictions
             y_true: the array of the true values
             fig_size: figure size as tuple
             cutoff: define class threshold list for the properties
             good_class: indicated by > or <, which class is 'positive'
             smiles: list of smiles
             apply_tick_transformation: boolean to transform ticks of the labels
        
        Returns: 
            F1_dict: optimized F1 dictionary and dictionary of figures
        """
        if not isinstance(properties, (list)):
            properties=[properties]
        if not isinstance(cutoff, (list)):
            cutoff=[cutoff]
            
        figures={}

        for ip,p in enumerate(properties):
            #fig, axs = plt.subplots(1, 2, figsize=(16 ,6))
            y_pred=out[f'predicted_{p}']
            mask=np.isnan(y_true[ip])| np.isnan(y_pred.astype(float))
            fig_reg_sc,metrics=plotly_reg_model_with_cutoff(y_pred[~mask],  y_true[ip][~mask],title=f'{p}',fig_size=fig_size,cutoff=cutoff[ip], good_class=good_class,smiles=np.array(smiles)[~mask],apply_tick_transformation=apply_tick_transformation, results_dict=results_dict  )
            fig_key=figure_annotations['J']
            figures[f'{fig_key}{p}<metric>({metrics})']=fig_reg_sc 
            
            b=None#b=np.quantile(y_true[ip][~mask], 0.001)
            e=None#e=np.quantile(y_true[ip][~mask], 0.95)
            if good_class =='<':
                yt_class=(y_true[ip]<cutoff[ip])*1
            else:
                yt_class=(y_true[ip]>cutoff[ip])*1
            fig_reg_enrich,m1,fig_reg_pr,m2=plotly_enrichment(yt_class,y_pred,title=f'<br><sup> Positive {good_class} {cutoff[ip]}</sup> <br> <sup>{p}</sup>',b=b,e=e, good_class= good_class, colors=self.colors, line_width=self.line_width,fig_size=fig_size, results_dict=results_dict )
            fig_key=figure_annotations['K']
            figures[f'{fig_key}{p}<metric>({m1}, Positive {good_class} {cutoff[ip]})']=fig_reg_enrich
            fig_key=figure_annotations['L']
            figures[f'{fig_key}{p}<metric>({m2}, Positive {good_class} {cutoff[ip]})']=fig_reg_pr
            
            b=np.quantile(y_true[ip][~mask], 0.001)
            e=np.quantile(y_true[ip][~mask], 0.95)
            fig_reg_ap=plotly_acc_pre_for_reg(y_true[ip],y_pred,title=f'{p}',b=b,e=e, good_class= good_class, colors=self.colors, line_width=self.line_width,fig_size=fig_size, results_dict=results_dict )
            fig_key=figure_annotations['G']
            figures[f'{fig_key}{p}<metric>(Positive {good_class} threshold)']=fig_reg_ap
            
        return figures
    
    def show_bokeh_scatter(self,properties,out,y_true,fig_size=(600,600),smiles=None,legend_pos="bottom_right",apply_tick_transformation=False ):
        """ 
        creates scatterplot in bokeh with chemical structures
           
        Args:     
             properties: property names list present in the output
             out: model output containing the predictions
             y_true: the array of the true values
             fig_size: figure size as tuple
             smiles: list of smiles
             apply_tick_transformation: boolean to transform ticks of the labels
             legend_pos: the position of the legend
        
        Returns: 
            dictionary of figures
        """
        if not isinstance(properties, (list)):
            properties=[properties]
        figures={}

        for ip,p in enumerate(properties):
            #fig, axs = plt.subplots(1, 2, figsize=(16 ,6))
            y_pred=out[f'predicted_{p}']
            mask=np.isnan(y_true[ip])| np.isnan(y_pred.astype(float))
            fig_reg=bokeh_reg_model(y_pred[~mask],  y_true[ip][~mask],title=f'{p}',fig_size=fig_size,smiles=np.array(smiles)[~mask],legend_pos=legend_pos,apply_tick_transformation=apply_tick_transformation )
            figures[f'reg_bokeh_scatterplot_{p}']=fig_reg
            
        return figures
    
    def show_bokeh_cutoff_report(self,properties,out,y_true,fig_size=(600,600),cutoff=0.5, good_class= '>', smiles=None, legend_pos="bottom_right", apply_tick_transformation=False ):
        """ 
        creates scatterplot in bokeh with chemical structures with threshold
        
        Args:
             properties: property names list present in the output
             out: model output containing the predictions
             y_true: the array of the true values
             fig_size: figure size as tuple
             cutoff: define class threshold list for the properties
             good_class: indicated by > or <, which class is 'positive'
             smiles: list of smiles
             apply_tick_transformation: boolean to transform ticks of the labels
             legend_pos: the position of the legend
        
        Returns: 
            dictionary of figures
        """
        if not isinstance(properties, (list)):
            properties=[properties]
        if not isinstance(cutoff, (list)):
            cutoff=[cutoff]
            
        figures={}

        for ip,p in enumerate(properties):
            #fig, axs = plt.subplots(1, 2, figsize=(16 ,6))
            y_pred=out[f'predicted_{p}']
            mask=np.isnan(y_true[ip])| np.isnan(y_pred.astype(float))
            fig_reg_sc=bokeh_reg_model_with_cutoff(y_pred[~mask],  y_true[ip][~mask],title=f'{p}',fig_size=fig_size,cutoff=cutoff[ip], good_class=good_class,smiles=np.array(smiles)[~mask],legend_pos=legend_pos,apply_tick_transformation=apply_tick_transformation )
            figures[f'reg_bokeh_scatterplot_cutoff_{cutoff[ip]}_{p}']=fig_reg_sc 
            
        return figures
    
    
    
    def show_additional_regression_report(self,properties,out,y_true,mask_scatter=False,prop_cliffs=None, leave_grp_out=None,fig_size=(600,600), bins=50, bin_window_average=False,smiles=None,apply_tick_transformation=False, results_dict=None):
        """        
        Shows the additional regression results
        
        Args:
             properties: property names list present in the output
             out: model output containing the predictions
             y_true: the array of the true values
             mask_scatter: masks the scatter of the plot
             prop_cliffs: property cliff dictionary
             leave_grp_out: list of leave group out indices
             bins: window size for moving average
             bin_window_average: instead of moving average use binned average
             fig_size: figure size as tuple
             smiles: list of smiles
             apply_tick_transformation: boolean to transform ticks of the labels
        
        Returns: 
            dictionary of figures
        """
        if not isinstance(properties, (list)):
            properties=[properties]
        if prop_cliffs is None:
            prop_cliffs={p:None for p in properties}
            
        figures={}

        for ip,p in enumerate(properties):
            #fig, axs = plt.subplots(1, 2, figsize=(16 ,6))
            y_pred=out[f'predicted_{p}']
            mask=np.isnan(y_true[ip])| np.isnan(y_pred.astype(float))
            non_nan_indices=np.cumsum(~mask)-1
            act_cliff_indices=None
            ood_indices=None
            if prop_cliffs[p] is not None:
                act_cliff_indices=non_nan_indices[prop_cliffs[p]]
            if leave_grp_out is not None:
                ood_indices=non_nan_indices[leave_grp_out]
            fig_reg_sc_err,metrics=plotly_reg_model_with_error(y_pred[~mask],  y_true[ip][~mask],title=f'{p}', alpha=0.3, mask_scatter=mask_scatter, bins=bins, prop_cliffs=act_cliff_indices, leave_grp_out=ood_indices,bin_window_average=bin_window_average,fig_size=fig_size, smiles=np.array(smiles)[~mask],apply_tick_transformation=apply_tick_transformation, results_dict=results_dict )
            fig_key=figure_annotations['E']
            figures[f'{fig_key}{p}<metric>({metrics})']=fig_reg_sc_err

            plot_bar={}
            plot_bar[f'predicted_{p}']=y_pred[~mask]
            plot_bar[f'abs_error_{p}']=np.absolute( y_true[ip][~mask]-y_pred[~mask])
            fig_reg_pb=plotly_confusion_bars_from_continuos(pd.DataFrame.from_dict(plot_bar),pro1= f'predicted_{p}',pro2= f'abs_error_{p}', 
                    bins2=np.linspace(np.min(plot_bar[f'abs_error_{p}']), np.max(plot_bar[f'abs_error_{p}']),6)[1:-1],
                    bins1=np.linspace(np.min(plot_bar[f'predicted_{p}']), np.max(plot_bar[f'predicted_{p}']),5)[1:-1],
                    title=f'{p}', fig_size=fig_size,leg_title='abs. err', results_dict=results_dict 
                )
            fig_key=figure_annotations['F']
            figures[f'{fig_key}{p}<metric>']=fig_reg_pb  

        return figures
    
        
    def show_regression_report(self,properties,out,y_true,mask_scatter=False,prop_cliffs=None, leave_grp_out=None,fig_size=(600,600), bins=50, bin_window_average=False,smiles=None,apply_tick_transformation=False, results_dict=None ):
        """        
        Shows the regression results
        
        Args:
             properties: property names list present in the output
             out: model output containing the predictions
             y_true: the array of the true values
             mask_scatter: masks the scatter of the plot
             prop_cliffs: property cliff dictionary
             leave_grp_out: list of leave group out indices
             bins: window size for moving average
             bin_window_average: instead of moving average use binned average
             fig_size: figure size as tuple
             smiles: list of smiles
             apply_tick_transformation: boolean to transform ticks of the labels
        
        Returns: 
            dictionary of figures
        """
        if not isinstance(properties, (list)):
            properties=[properties]
        if prop_cliffs is None:
            prop_cliffs={p:None for p in properties}
            
        figures={}

        for ip,p in enumerate(properties):
            #fig, axs = plt.subplots(1, 2, figsize=(16 ,6))
            y_pred=out[f'predicted_{p}']
            mask=np.isnan(y_true[ip])| np.isnan(y_pred.astype(float))
            non_nan_indices=np.cumsum(~mask)-1
            fig_reg_sc,metrics=plotly_reg_model(y_pred[~mask],  y_true[ip][~mask],title=f'{p}',fig_size=fig_size,smiles=np.array(smiles)[~mask],apply_tick_transformation=apply_tick_transformation, results_dict=results_dict )
            fig_key=figure_annotations['D']
            figures[f'{fig_key}{p}<metric>({metrics})']=fig_reg_sc 

        return figures
        
    