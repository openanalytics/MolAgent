"""
AutoMoL: Pipeline for automated machine learning for drug design.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""
import warnings
import sys, os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import time
import datetime
from datetime import date

from os import path
import argparse
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np, pandas as pd
from matplotlib import pyplot as plt

import automol
from automol.stacking_util import ModelAndParams

from automol.stacking import save_model
from automol.stacking import load_model

from app_model_training import *
from app_model_visuals import *
from app_model_postprocessing import *
from PIL import Image

logo_directory= 'logo.png'   
st.set_page_config(
     page_title="AutoMoL",
     page_icon=logo_directory,
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "AutoMoL: Pipeline for automated machine learning for drug design"
     }
 )

@st.cache_data
def st_predict_smiles(_stacked_model=None,model_name='',props =None, smiles=None,compute_SD=True,convert_log10=False):
    """Cached predictions of the smiles
    """
    out=stacked_model.predict( props =props, smiles=smiles,compute_SD=compute_SD,convert_log10=convert_log10)
    return out

@st.cache_resource
def st_load_model(save_model_to_file,use_gpu=False,get_errors=False):
    """Streamlit cached version for loading the model
    """
    stacked_model,err_dict= load_model( save_model_to_file,use_gpu=use_gpu,retrieve_reproducability_errors=get_errors) 
    return stacked_model,err_dict


@st.cache_data
def st_show_general(_modelvisualizer,model_name='',plotly_dictionary={},revert=False,revert_labels=False,user_col=None, nb_cols=None, use_fixed_width=None, x_width=None, y_height=None, generate_bokeh_reg=None):
    modelvisualizer.show_general_figures(plotly_dictionary)
    return plotly_dictionary



c1,c2=st.columns([8,1])
with c2:
    logo = Image.open(logo_directory)
    st.image(logo,width=100)
with c1:
    st.title('AutoM(o)L')

st.markdown('Pipeline for automated machine learning using the bottleneck transformer.  \n * **Authors**: Mazen Ahmad, Joris Tavernier, Natalia Dyubankova, Marvin Steijaert  \n * **Contact**: Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu  \n &copy; All rights reserved, Open Analytics NV, 2021-2025.' )
st.caption('Maintainer: Joris Tavernier')

st.markdown('**General Usage Guidelines** \n \n Firstly, we, the developers of AutoMoL, assume that the data provided to the pipeline is clean data. The pipeline **does not handle qualifiers or outliers** in the property target. This is the responsibility of the end-user. This should be done with care and with knowledge of the data at hand. \n \n \n Happy Trails!')

train_model=False 
model_param={}

############################################
#section for user options to train the model
############################################
#retrieve data file and names generates files
save_model_to_file,save_output_to_file,sep, df_file, generate_pdf, title, authors, mails, summary, send_mail, sender,recipients  = gen_sb_model()
if df_file is not None:
    df = load_data(df_file,sep=sep)

    st.sidebar.subheader("Task parameters")
    with st.sidebar.expander("General task options",expanded=True):
        smiles_column, properties, task, computional_load, strategy,standardization, feature_generators, used_features= gen_sb_general_tasks(df)

    if len(properties)>0:
        st.sidebar.subheader("Task specific parameters")
        categorical=False
        nb_classes=None
        use_quantiles=False
        class_values=None
        clf_scorer=None
        reg_scorer=None
        confidence=None
        use_log10=False
        use_logit=False
        percentages=False
        remove_outliers=False
        if task=="Classification":
            with st.sidebar.expander("Classification options",expanded=True):
                if len(properties)>1:
                    categorical, nb_classes, use_quantiles, class_values  = gen_sb_classification(properties,df)
                else:
                    categorical, nb_classes, use_quantiles, class_values = gen_sb_single_prop_classification(properties,df)
                if (nb_classes[0]==2 and nb_classes[:-1] == nb_classes[1:]):
                    regclf = st.checkbox('Apply regression methods on binary [0,1] target',help="For binary classification, we added the possibility to apply regression on a target containing 0's and 1's. The output of the regression model is then clipped to the interval [0,1] and used as the probability for class 1. ")
                    if regclf:
                        task="RegressionClassification"
                        
        if task=="Regression":
            with st.sidebar.expander("Regression options",expanded=True):
                use_log10, use_logit, percentages, remove_outliers = gen_sb_regression()
        
        with st.sidebar.expander("Advanced options"):
            if task=="Classification":
                clf_scorer=gen_sb_adv_classification()
            if task=="RegressionClassification":
                clf_scorer=gen_sb_adv_regressionclassification()
            if task=="Regression":
                reg_scorer=gen_sb_adv_regression()
                
            st.subheader("Validation set options")
            prop_cliff_butina_th=0.0
            if strategy=='mixed':
                prop_cliff_butina_th=float(st.number_input('provide Butina threshold for property cliffs (1-similarity)', 0.05,0.95,0.45,0.01))
            val_clustering, test_size, val_include_chirality, val_butina_cutoff, val_km_groups,val_km_features, plot_sihl, sihl_map, minority_nb = gen_sb_validation(strategy,feature_generators)

            st.subheader("Nested group cross-validation options")
            cv_clustering, include_chirality, butina_cutoff, km_groups,cv_km_features,cross_val_split, outer_folds, random_state, random_state_list, n_jobs = gen_sb_ncv(feature_generators)

            st.subheader("Sample weights")
            use_weighted_samples,weighted_samples_index,selected_sample_weights = gen_sb_sw()

    verbose = st.sidebar.checkbox('verbose')
    train_model = st.sidebar.button(label='Train model')
        
if train_model and len(properties)<1:
    train_model=False
    st.error("No properties selected or available to be fitted chosen in the sidebar")
        
##########################################
#section for training the model
##########################################
if train_model:
    #when a model is trained an e-mail can be sent
    if 'send_mail' in st.session_state:
        del st.session_state['send_mail']
    st.subheader('Training the model')
    min_allowed_class_samples=30
    if outer_folds>km_groups:
        if km_groups>2:
            outer_folds=km_groups-1
        else:
            km_groups=outer_folds+1
    
    #standardization
    standard_smiles_column='stereo_SMILES'
    df=standardize_data(df,smiles_column,verbose,properties,standard_smiles_column=standard_smiles_column,standardization=standardization)
    if computional_load=='expensive' and use_weighted_samples:
        st.warning('currently, weighted samples do not work with sklearn stacking methods and is disabled for the expensive computation load, try moderate or cheap')
        use_weighted_samples=False
    
    if task=='Classification' or task=='RegressionClassification':
        model_trainer=ClassifierTrainer(properties=properties,nb_classes=nb_classes,class_values=class_values,scorer=clf_scorer,
                                        categorical=categorical,use_quantiles=use_quantiles)
    else:
        if cross_val_split=='SKF':
            st.warning('Changed cross-validation strategy to GKF, SKF only available for classification and in general not recommended')
            cross_val_split='GKF'
        model_trainer=RegressionTrainer(properties=properties,scorer=reg_scorer,remove_outliers=remove_outliers,confidence=confidence,
                                        use_log10=use_log10,use_logit=use_logit,percentages=percentages,standard_smiles_column=standard_smiles_column)
        
    #check properties
    model_trainer.check_properties(df)        
    #prepare properties: e.g. create classes, apply transformations etc..c 
    df,train_properties=model_trainer.generate_train_properties(df)
    if use_weighted_samples:
        df=model_trainer.generate_sample_weights(df,weighted_samples_index,selected_sample_weights)
    #get stacking_model
    param_Factory=ModelAndParams(
                             task=task,
                             computional_load=computional_load,
                             distribution_defaults=False, #use distributions for parameter selection in randomized search
                             use_gpu=False,
                             normalizer=True, #normalize input base estimators
                             top_normalizer=True,#normalize output base estimators and thus input top estimator
                             random_state=random_state_list,#random_state
                             use_sample_weight=use_weighted_samples,
                             feature_generators=feature_generators,
                             verbose=verbose,
                             n_jobs=n_jobs,
                             labelnames=model_trainer.labelnames)
    stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
    st.write(df.head())   
    
    df_smiles=df.stereo_SMILES
    #split the data in training and validation
    gen_mp_validation(df,train_properties,model_trainer,strategy,categorical,stacked_model,df_smiles,standard_smiles_column,test_size,val_clustering,
                     val_km_groups,val_km_features,val_butina_cutoff,val_include_chirality,verbose,random_state,sihl_map,plot_sihl,prop_cliff_butina_th,minority_nb=minority_nb)
    
    for p in properties:
        prop_count=df[p].count()
        if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
            st.warning(f'Warning: on average less than 10 samples per cluster for property {p}, suggested use is to decrease number of groups')

    # cluster the data within the models into groups to be used for NCV
    start_time=time.time()
    with st.spinner('Clustering data for nested cross-validation'):
        cv_c_algo=get_clustering_algorithm(clustering=cv_clustering,
                             n_clusters=km_groups,
                             cutoff=butina_cutoff,
                             include_chirality=include_chirality,
                             verbose=verbose,
                             random_state=random_state,
                             feature_generators=stacked_model.feature_generators,
                             used_features=cv_km_features)
        stacked_model.Data_clustering(random_state=random_state, clustering_algorithm=cv_c_algo)
    st.success(f'Finished clustering data for nested cross-validation set in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')

    if outer_folds>km_groups:
        if km_groups>2:
            outer_folds=km_groups-1
        else:
            km_groups=outer_folds+1
            
    st.info(f'Training models for {train_properties} ...')
    for ip,p in enumerate(train_properties):
        start_time=time.time()
        sample_weight=None
        if use_weighted_samples:
            sample_weight=stacked_model.Train[f'sw_{p}'].values
            
        with st.spinner(f'Training model for property {p}'):
            stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                        params_grid=params_grid,
                                   paramsearch=paramsearch,
                                   features=used_features,
                                   scoring=model_trainer.scorer,#'neg_mean_absolute_error',#
                                   cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                   use_memory=True,
                                   plot_validation=True, 
                                   refit=False,# no refit with validation. comes later,
                                   blender_params=blender_params,
                                  prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        st.success(f'Finished training model for property {p} in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')
        st.progress((ip+1.0)/len(train_properties))

    
    model_tostr_list=stacked_model.print_metrics()
    for m_to_str in model_tostr_list:
        st.markdown(m_to_str.replace('\33[1m' ,'**').replace('\33[0m' ,'**').replace("\n","  \n"))
    #save_model_to_file='ppb_clf2.pt'
    save_model(stacked_model, save_model_to_file)

st.subheader('Loading the model')
with st.expander("Upload own model"):
    with st.form("modelupload"):
        pt_file = st.file_uploader("Upload model .pt file", type={"pt"},help="Drag or select your pt model")
        # Every form must have a submit button.
        upload_model = st.form_submit_button("Upload .pt model for in app figure analysis")
        if upload_model:
            model_name = pt_file.name
            with open(model_name, mode="wb") as f:
                f.write(pt_file.read())
        
model_files=[]
for file in os.listdir():
    if file.endswith(".pt"):
        model_files.append(file)

if len(model_files) >0:
    model_index=0        
    if save_model_to_file in model_files:
        model_index=model_files.index(save_model_to_file)
        
    load_model_file = st.selectbox(
                     'Which model do you want to use?',
                      model_files,index=model_index)    
else:
    load_model_file='test.pt'

load_model_pushed = st.button(f"load model {load_model_file}")
if not path.exists(load_model_file):
    load_model_pushed=False

#we have a model
if load_model_pushed or train_model:
    st.subheader('Model analysis')
    
    clear_cache=st.button('Clear predictions from cache',help="Streamlit caches the predictions of the model, if the model file has changed push this button to clear the cache")
    if clear_cache:
        st.cache_data.clear()
        st.experimental_rerun()

    if path.exists("biosignature.debug.log"): os.remove("biosignature.debug.log")
    
    stacked_model,err_dict=st_load_model(load_model_file, use_gpu=False,get_errors=True)
    if err_dict:
        st.warning('Reproducability check failed with following errors: ')
        for key,val in err_dict.items():
            st.warning(f'{key}:{val}')
            
    if not hasattr(stacked_model, 'Validation') or stacked_model.Validation.empty:
        st.session_state['valid_model']=False
        st.error('Data has been removed for deployment.')
    elif not stacked_model.models:
        st.session_state['valid_model']=False
        st.error('No models are trained or all properties have been deleted.')
    elif not validate_properties_in_data(stacked_model):
        st.session_state['valid_model']=False
        st.error('Not all properties from the model are available in the validation set of the model, probably due to merging of models. ')
    ####################################################
    # we have data to plot figures
    else:
        st.session_state['valid_model']=True
        st.session_state['saved_model']=load_model_file
        st.session_state['model']=stacked_model
        
if 'valid_model' in st.session_state:
    if not st.session_state['saved_model']==load_model_file:
        st.session_state['valid_model']=False
    if st.session_state['valid_model']:
        stacked_model=st.session_state['model']
        smiles_list=stacked_model.Validation.stereo_SMILES

        class_properties=[p for p in stacked_model.models]
        with st.expander("Figures lay-out configuration"):
            with st.form("Figure lay-out"):
                user_col,nb_cols,use_fixed_width,x_width,y_height=figure_lay_out()
                figure_layout = st.form_submit_button("Apply changes")
                
        plotly_dictionary={}
        classifier=False
        if hasattr(stacked_model, 'labelnames'):
            classifier=True
            modelvisualizer=ClassificationVisualiser(user_col,nb_cols,use_fixed_width,x_width,y_height)
        else:
            modelvisualizer=RegressionVisualiser(user_col,nb_cols,use_fixed_width,x_width,y_height,smiles_list)

        out=st_predict_smiles(_stacked_model=stacked_model,model_name=load_model_file,props =None, smiles=smiles_list,compute_SD=True,convert_log10=modelvisualizer.revert_log10)
        
        
        modelvisualizer.generate_general_figures(class_properties,out,stacked_model)
        plotly_dictionary=st_show_general(modelvisualizer,load_model_file,plotly_dictionary,modelvisualizer.revert_log10,modelvisualizer.apply_tick_transformation
                                         ,modelvisualizer.user_col
                                         ,modelvisualizer.nb_cols
                                         ,modelvisualizer.use_fixed_width
                                         ,modelvisualizer.x_width
                                         ,modelvisualizer.y_height
                                         ,modelvisualizer.generate_bokeh_reg)

        plotly_dictionary=modelvisualizer.generate_additional_figures(plotly_dictionary)    
            
        st.subheader("Model report")
        #setting the model params inside a dictionary
        if df_file is not None and save_model_to_file==load_model_file and len(properties)>0:
            model_param['file_name']=df_file.name
            model_param['sep']=sep
            model_param['verbose']=verbose
            model_param['check_rdkit_desc']=False
            model_param['standardization']=standardization
            model_param['rdkit_standardization']= (standardization=='RDKIT')

            model_param['computional_load']=computional_load
            model_param['normalizer']=True
            model_param['top_normalizer']=True
            model_param['Task']=task
            model_param['properties']=properties
            if classifier:
                model_param['categorical']=categorical
                model_param['nb_classes']=nb_classes
                model_param['class_values']=None
                if use_quantiles:
                    model_param['class_quantiles']=class_values
                else:
                    model_param['class_values']=class_values
                model_param['min_allowed_class_samples']=min_allowed_class_samples
                model_param['scorer']=clf_scorer
                model_param['labelnames']=stacked_model.labelnames
            else:
                model_param['use_log10']=use_log10
                model_param['use_logit']=use_logit
                if use_logit: model_param['percentages']=percentages
                model_param['scorer']=reg_scorer

            model_param['val_clustering']=val_clustering
            model_param['strategy']=strategy
            model_param['test_size']=test_size
            model_param['minority_nb']=minority_nb
            model_param['prop_cliff_butina_th']=prop_cliff_butina_th
            model_param['mix_coef_dict']={
                        'prop_cliffs': 0.3,
                        'leave_group_out': 0.3,
                        'stratified': 0.4}
            if val_clustering=="Scaffold" : model_param['val_include_chirality']=val_include_chirality
            if val_clustering=="Butina" or val_clustering=="HierarchicalButina": model_param['val_butina_cutoff']=val_butina_cutoff
            if val_clustering=="Bottleneck" : model_param['val_km_groups']=val_km_groups

            model_param['cv_clustering']=cv_clustering
            if cv_clustering=="Scaffold" : model_param['include_chirality']=include_chirality
            if cv_clustering=="Butina" or val_clustering=="HierarchicalButina": model_param['butina_cutoff']=butina_cutoff
            if cv_clustering=="Bottleneck" : model_param['km_groups']=km_groups
            model_param['cross_val_split']=cross_val_split
            if cross_val_split != 'LGO': model_param['outer_folds']=outer_folds
            model_param['random_state']=random_state
            model_param['random_state_list']=random_state_list
            model_param['n_jobs_dict']={
                            'outer_jobs': None,
                            'inner_jobs': n_jobs,
                            'method_jobs': 2,
                            }
            model_param['use_sample_weight']=use_weighted_samples
            model_param['use_provided_sample_weight']=False
            if use_weighted_samples: model_param['weighted_samples_index']=weighted_samples_index
            if use_weighted_samples: model_param['select_sample_weights']=selected_sample_weights

            model_param['revert_to_original']=modelvisualizer.revert_log10
            model_param['positive_cls']=modelvisualizer.good_class
            model_param['cutoff']=modelvisualizer.cutoff
            model_param['apply_tick_transformation']=modelvisualizer.apply_tick_transformation
            model_param['x_width']=modelvisualizer.x_width
            model_param['y_height']=modelvisualizer.y_height
            model_param['output_file']=save_output_to_file
            model_param['title']=title
            model_param['authors']=authors
            model_param['mails']=mails
            model_param['sender']=sender
            model_param['receivers']=recipients
            model_param['summary']=summary
            model_param['save_model_to_file']=load_model_file

        else:
            st.warning('No data file is provided or model file for training is different from model file for validation, model parameters are not added to the report or yaml file. Train the model in the sidebar to set the correct training parameters for reproducability of the results.')
            model_param={}
            
        with st.expander("Generate pdf summary?", expanded=True):
            model_str=stacked_model.print_metrics()
            #run pdf generation code
            figure_types, nb_columns = gen_pm_pdf_summary(classifier,model_param,plotly_dictionary,save_output_to_file,model_str,modelvisualizer.class_properties, generate_pdf, title, authors, mails, summary, send_mail, sender,recipients)
            model_param['figure_types']=figure_types
            # set number of columns
            model_param['nb_columns']=nb_columns
    st.subheader("Yaml file")  
    gen_pm_yaml_file(model_param,save_output_to_file)
    
    st.subheader("Deleting properties")  
    with st.expander("Deleting properties", expanded=True):
        del_properties = st.multiselect(
            "Select properties to be removed from the model",
            [p for p in stacked_model.models],
        )
        update_model = st.button(f"Delete properties {del_properties} from model {load_model_file}")
        if update_model:
            with st.spinner('Updating model data'):
                if len(del_properties)>0:
                    stacked_model.delete_properties(del_properties)
                save_model(stacked_model, load_model_file)
                st.legacy_caching.clear_cache()
            st.success('succesfully updated and saved model')
    st.subheader("Merging models")   
    with st.expander("Merge models?",expanded=True):
        st.info("After merging models, figures will fail since no data is available for the new properties to plot")
        available_models=model_files.copy()
        available_models.remove(load_model_file)
        if len(available_models) >0:
            merge_model_file = st.selectbox(
                 'Which model do you want to choose?',
                  available_models)

            other_model,err_dict=st_load_model(merge_model_file, use_gpu=False,get_errors=True)
            if err_dict:
                st.warning('Reproducability check failed with following errors: ')
                for key,val in err_dict.items():
                    st.warning(f'{key}:{val}')

            merged_properties = st.multiselect(
                    f"Select properties to be added from {merge_model_file} to be added to {load_model_file}",
                    [p for p in other_model.models],help='note that identical properties will be overwritten',
                )
            merge_models = st.button(f"Merge properties {merged_properties} from {merge_model_file} with {load_model_file} and save model as {load_model_file}")
            if merge_models:
                stacked_model.merge_model(other_model=other_model,other_props=merged_properties)
                save_model(stacked_model, load_model_file)
        else:
            st.warning('No other models found in the app directory')
            
    st.subheader("Model download")        
    with st.expander("Download model",expanded=True):
        gen_pm_download_model(stacked_model,load_model_file)
        

if len(model_files)>0:
    st.subheader("Clean-up")
    del_mod_files = st.multiselect(
                    "Select models to be removed from the directory",
                    model_files,
                    default=[load_model_file],
                    help="Select models to be removed from the directory",
            )
    remove_models = st.button(f"Remove following models: {del_mod_files}")
    if remove_models:
        for f in del_mod_files:
            os.remove(f)
