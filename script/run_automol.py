import warnings
import sys, os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
import yaml
import argparse

import numpy as np, pandas as pd
from  matplotlib import pyplot as plt

from automol.property_prep import add_stereo_smiles,validate_rdkit_smiles, add_rdkit_standardized_smiles
from automol.property_prep import PropertyTransformer,ClassBuilder

from automol.stacking_util import retrieve_search_options, ModelAndParams

from automol.feature_generators import retrieve_default_offline_generators

from automol.validation import stratified_validation, leave_grp_out_validation, mixed_validation

from automol.plotly_util import PlotlyDesigner
from automol.plotly_util import create_figure_captions, get_figures_from_types
from automol.plotly_util import generate_report
from automol.plotly_util import save_result_dictionary_to_json

from automol.stacking import save_model

from typing import List 
    
def check_value_dict(input_dict,key,default_val):
    if key in input_dict:
        return input_dict[key]
    else:
        return default_val
    
if __name__ == '__main__':
    
    ############################################
    # argument & yaml parsing
    ############################################
    
    warnings.filterwarnings(action='once')
    my_parser = argparse.ArgumentParser(description='AutoMoL: Pipeline for automated machine learning for drugdesign',allow_abbrev=False)
    my_parser.add_argument('--yaml_file',
                       metavar='file', #variable name in help
                       type=str,
                       action='store',
                       help='the YAML file',
                       nargs='?',
                       default='automl_reg.yaml')
    args = my_parser.parse_args()

    with open(args.yaml_file) as f:
        input_dict = yaml.safe_load(f)
    for key, value in input_dict.items():
        print (key + " : " + str(value))
        
    assert 'file_name' in input_dict, 'Provide input data file in yaml file with key file_name'
    file_name=input_dict['file_name']
    assert 'smiles_column' in input_dict, 'Provide column of original smiles in yaml file with key smiles_column'
    smiles_column=input_dict['smiles_column'] 
    assert 'properties' in input_dict, 'Provide properties to be fitted in yaml file with key properties'
    properties=input_dict['properties'] 
    assert 'task' in input_dict, 'Provide task in yaml file with key task'
    task=input_dict['task']
    assert task in ['Regression','Classification','RegressionClassification'], 'Choose task from [Regression,Classification,RegressionClassification]'
    verbose=check_value_dict(input_dict,'verbose',False)
    check_rdkit_desc=check_value_dict(input_dict,'check_rdkit_desc',False)
    if 'standardization' in input_dict:
        if input_dict['standardization'] == 'rdkit' or input_dict['standardization'] == 'RDKIT':
            rdkit_standardization=True
        else:
            rdkit_standardization=False
    else:
        rdkit_standardization=check_value_dict(input_dict,'rdkit_standardization',False)
    standard_smiles_column='stereo_SMILES'
    sep=check_value_dict(input_dict,'sep',',')
    categorical=check_value_dict(input_dict,'categorical',False)
    nb_classes=check_value_dict(input_dict,'nb_classes',None)
    class_values=check_value_dict(input_dict,'class_values',None)
    class_quantiles=check_value_dict(input_dict,'class_quantiles',None)
    min_allowed_class_samples=check_value_dict(input_dict,'min_allowed_class_samples',30)
    use_log10=check_value_dict(input_dict,'use_log10',False)
    percentages=check_value_dict(input_dict,'percentages',False)
    use_logit=check_value_dict(input_dict,'use_logit',False)
    use_sample_weight=check_value_dict(input_dict,'use_sample_weight',False)
    use_provided_sample_weight=check_value_dict(input_dict,'use_provided_sample_weight',False)
    weighted_samples_index=check_value_dict(input_dict,'weighted_samples_index',None)
    select_sample_weights=check_value_dict(input_dict,'select_sample_weights',None)
    computional_load=check_value_dict(input_dict,'computional_load','cheap')
    assert computional_load in ['cheap','moderate','expensive'], 'Choose computional_load from [cheap,moderate,expensive]'
    if 'scorer' in input_dict:
        scorer=input_dict['scorer']
    else:
        if task == 'Classification':
            scorer='balanced_accuracy'
        else:
            scorer='r2'
    random_state=check_value_dict(input_dict,'random_state',5)
    random_state_list=check_value_dict(input_dict,'random_state_list',[1,7,42,55,3])
    n_jobs_dict=check_value_dict(input_dict,'n_jobs_dict',{'outer_jobs':None, 
                'inner_jobs':-1, 
                'method_jobs':2})
    search_type=check_value_dict(input_dict,'search_type',None)
    randomized_iterations=check_value_dict(input_dict,'randomized_iterations',None)
    model_config=check_value_dict(input_dict,'model_config',None)
    use_distributions=check_value_dict(input_dict,'use_distributions',False)
    blender_list=check_value_dict(input_dict,'blender_list',None)
    base_list=check_value_dict(input_dict,'base_list',None)
    used_features=check_value_dict(input_dict,'used_features',[ ['Bottleneck'] for p in properties])
    red_dim_list=check_value_dict(input_dict,'red_dim_list',['passthrough'])
    red_dim_list_per_prop=check_value_dict(input_dict,'red_dim_list_per_prop',None)
    local_dim_red=check_value_dict(input_dict,'local_dim_red',False)
    dim_red_n_components=check_value_dict(input_dict,'dim_red_n_components',None)
    normalizer=check_value_dict(input_dict,'normalizer',True)
    top_normalizer=check_value_dict(input_dict,'top_normalizer',False)
    val_clustering=check_value_dict(input_dict,'val_clustering','Bottleneck')
    val_km_groups=check_value_dict(input_dict,'val_km_groups',30)
    val_include_chirality=check_value_dict(input_dict,'val_include_chirality',False)
    val_butina_cutoff=check_value_dict(input_dict,'val_butina_cutoff',[0.2,0.4,0.6]  )
    strategy=check_value_dict(input_dict,'strategy','mixed')
    assert strategy in ['mixed','stratified','leave_group_out'], 'Choose strategy from [mixed,stratified,leave_group_out]'
    test_size=check_value_dict(input_dict,'test_size',0.25)
    minority_nb=check_value_dict(input_dict,'minority_nb',5)
    prop_cliff_butina_th=check_value_dict(input_dict,'prop_cliff_butina_th',0.45)
    rel_prop_cliff=check_value_dict(input_dict,'rel_prop_cliff',0.5)
    mix_coef_dict=check_value_dict(input_dict,'mix_coef_dict',{ 'prop_cliffs': 0.3,'leave_group_out': 0.3 ,'stratified': 0.4})
    cv_clustering=check_value_dict(input_dict,'cv_clustering','Bottleneck')
    km_groups=check_value_dict(input_dict,'km_groups',30)
    include_chirality=check_value_dict(input_dict,'include_chirality',False)
    butina_cutoff=check_value_dict(input_dict,'butina_cutoff',[0.2,0.4,0.6])
    cross_val_split=check_value_dict(input_dict,'cross_val_split','GKF')
    assert cross_val_split in ['GKF','LGO','SKF'], 'Choose cross_val_split from [GKF,LGO,SKF]'
    outer_folds=check_value_dict(input_dict,'outer_folds',4)
    cmap=check_value_dict(input_dict,'cmap',['PiYG','Blues'])
    revert_to_original=check_value_dict(input_dict,'revert_to_original',False)
    positive_cls=check_value_dict(input_dict,'positive_cls','<')
    apply_tick_transformation=check_value_dict(input_dict,'apply_tick_transformation',False)
    cutoff=check_value_dict(input_dict,'cutoff',None)
    x_width=check_value_dict(input_dict,'x_width',600)
    y_height=check_value_dict(input_dict,'y_height',450)
    output_file=check_value_dict(input_dict,'output_file','summary.pdf')
    title=check_value_dict(input_dict,'title','AutoML report Title')
    authors=check_value_dict(input_dict,'authors','Joris Tavernier')
    mails=check_value_dict(input_dict,'mails','joris.tavernier@openanalytics.eu')
    summary=check_value_dict(input_dict,'summary','Project details')
    figure_types=check_value_dict(input_dict,'figure_types',None)
    nb_columns=check_value_dict(input_dict,'nb_columns',2)
    save_model_to_file=check_value_dict(input_dict,'save_model_to_file','automl_model.pt')    
    json_file=check_value_dict(input_dict,'json_file','automl_model.json')          
    ############################################
    # general script
    ############################################
    model_param={}
    model_param['Data file']=file_name

    df= pd.read_csv(file_name,sep=sep, na_values = ['NAN', '?','NaN'])
    add_rdkit_standardized_smiles(df, smiles_column,verbose=verbose,outname=standard_smiles_column)
    model_param['Standardization']='rdkit'

    if check_rdkit_desc: 
        validate_rdkit_smiles(df, standard_smiles_column,verbose=verbose)

    df.dropna(inplace=True, subset = [standard_smiles_column])
    df.dropna(inplace=True,how='all', subset = properties)
    df.reset_index(drop=True,inplace=True)
    model_param['Properties']=properties
    
    if task=='Classification' or task=='RegressionClassification':        
        model_param['Categorical properties?']=categorical
        if not categorical:
            if class_values is not None:
                model_param['Provided class cutoffs']=class_values
            elif class_quantiles is not None:
                model_param['Provided class quantile cutoffs']=class_quantiles
        if class_values is not None and class_quantiles is not None:
            print('Both class_values and class_quantiles are set, using values')
            class_quantiles=None
        if class_quantiles is not None:
            class_values=class_quantiles
        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=class_values,
                                 categorical=categorical,use_quantiles=class_quantiles is not None,
                                 prefix='Class',min_allowed_class_samples=min_allowed_class_samples,verbose=verbose)
        transformed_data=False
        labelnames=prop_builder.labelnames
        
    else:
        model_param['Log10 transformation?']=use_log10
        model_param['Logit transformation?']=use_logit
        if use_logit: model_param['Divide properties by 100?']=percentages
        transformed_data=use_log10 or use_logit
        prop_builder=PropertyTransformer(properties,use_log10=use_log10,use_logit=use_logit,percentages=percentages)
        labelnames=None
        
    prop_builder.check_properties(df)        
    df,train_properties=prop_builder.generate_train_properties(df)
    original_props=properties
    if transformed_data:
        original_props=prop_builder.original_props
    df_smiles=df[standard_smiles_column]
    
    
    if use_sample_weight: model_param['Provided samples to weighted per property']=weighted_samples_index
    if use_sample_weight: model_param['Provided weights per property']=select_sample_weights

    if use_sample_weight and not use_provided_sample_weight:
        df=prop_builder.generate_sample_weights(df,{train_properties[i]:weighted_samples_index[p]  for i,p in enumerate(original_props) },
                                                {train_properties[i]:select_sample_weights[p]  for i,p in enumerate(original_props) })
        
    model_param['Task']=task
    model_param['Score function']=scorer
    model_param['Random state']=random_state
    model_param['Random state list']=random_state_list
    model_param['computational_load']=computional_load

    if model_config is not None: 
        model_param['model_config']=model_config

    if model_config=='single_stack': 
        use_sample_weight=False
    if model_config=='top_stacking':
        use_sample_weight=False
    if model_config=='inner_stacking' :
        use_sample_weight=False

    if search_type is not None:
        randomized_iterations, hyperopt_defaults, distribution_defaults = retrieve_search_options(search_type=search_type,use_distributions=use_distributions, n_iter=randomized_iterations)
    else :
        hyperopt_defaults= None
        distribution_defaults=use_distributions
    if search_type is not None: model_param['param search type']=search_type
    model_param['randomized_iterations']=randomized_iterations
    model_param['distribution_defaults']=distribution_defaults
    model_param['hyperopt_defaults']=hyperopt_defaults
    
    if red_dim_list_per_prop is not None:
        assert len(red_dim_list_per_prop)==len(properties), 'provided list of dim reduction methods per property must equal length of list of properties'
        model_param['red_dim_list']=red_dim_list_per_prop
    else:
        model_param['red_dim_list']=red_dim_list
    model_param['local_dim_red']=local_dim_red
    model_param['base_list']=base_list
    model_param['blender_list']=blender_list 
    model_param['used_features']=used_features

    param_Factory=ModelAndParams(task=task,
                                 computional_load=computional_load,
                                 distribution_defaults=distribution_defaults, #use distributions for parameter selection in randomized search
                                 hyperopt_defaults=hyperopt_defaults, #use hyperopt
                                 feature_generators=retrieve_default_offline_generators(radius=2,nbits=2048),
                                 use_gpu=False,
                                 normalizer=normalizer, #normalize input base estimators
                                 top_normalizer=top_normalizer,#normalize output base estimators and thus input top estimator
                                 random_state=random_state_list,#random_state
                                 red_dim_list=red_dim_list,#list of dimensionality reducing methods keys from dim_archive
                                 method_list=base_list,#list of keys from method_archive
                                 blender_list=blender_list,#list of classifier keys from blender_archive
                                 model_config=model_config, #string with the model layout/config
                                 randomized_iterations=randomized_iterations, #number of randomized iterations
                                 n_jobs_dict=n_jobs_dict,
                                 labelnames=labelnames,
                                 use_sample_weight=use_sample_weight,
                                 local_dim_red=local_dim_red,
                                 dim_red_n_components=dim_red_n_components,
                                 verbose=verbose)
    stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()

    model_param['Clustering algorithm used for validation']=val_clustering
    model_param['Validation set strategy used']=strategy
    model_param['Ratio of test size wrt to total data size']=test_size
    model_param['minority_nb']=minority_nb
    if val_clustering=="Scaffold" : model_param['Include chiralty?']=val_include_chirality
    if val_clustering=="Butina" or val_clustering=="HierarchicalButina": model_param['Threshold(s) used for butina when creating Validation set']=val_butina_cutoff
    if val_clustering=="Bottleneck" : model_param['Number of clusters for Kmeans++']=val_km_groups

    #split the data in training and validation
    groups_left_out=None
    prop_cliff_dict=None
    if strategy=='mixed':
        Train, Validation,groups_left_out, prop_cliff_dict = mixed_validation(df_orig=df,properties=prop_builder.properties,stacked_model=stacked_model,standard_smiles_column=standard_smiles_column,
                                                      prop_cliff_cut=rel_prop_cliff,prop_cliff_butina=prop_cliff_butina_th,test_size=test_size,clustering=val_clustering,
                                                      n_clusters=val_km_groups,cutoff=val_butina_cutoff,include_chirality=val_include_chirality,
                                                      verbose=verbose,random_state=random_state,mix_dict=mix_coef_dict, categorical_data=categorical,
                                                      minority_nb=minority_nb)

    elif strategy=='stratified':
        Train, Validation = stratified_validation(df,train_properties,stacked_model,df_smiles,test_size=test_size,clustering=val_clustering,
                                                      n_clusters=val_km_groups,cutoff=val_butina_cutoff,include_chirality=val_include_chirality,
                                                      verbose=verbose,random_state=random_state,
                                                      minority_nb=minority_nb)

    else:
        Train, Validation = leave_grp_out_validation(df,train_properties,stacked_model,df_smiles,test_size=test_size,clustering=val_clustering,
                                                      n_clusters=val_km_groups,cutoff=val_butina_cutoff,include_chirality=val_include_chirality,
                                                      verbose=verbose,random_state=random_state)
        groups_left_out=np.arange(len(Validation))

    #placing the validation and training set inside the model generator
    stacked_model.Validation=Validation
    stacked_model.Train=Train
    stacked_model.smiles=standard_smiles_column # set col smiles name
    
    model_param['Clustering algorithm used for Nested cross-validation']=cv_clustering
    if cv_clustering=="Scaffold" : model_param['Include chiralty for nested cross-validation?']=include_chirality
    if cv_clustering=="Butina" or val_clustering=="HierarchicalButina": model_param['Threshold(s) used for butina when clustering for nested cross-validation']=butina_cutoff
    if cv_clustering=="Bottleneck" : model_param['Number of clusters for Kmeans++ for Nested cross-validation']=km_groups

    #property check
    for p in properties:
        prop_count=df[p].count()
        if prop_count<100:
            print('Warning: property',p,', has less than 100 valid values')
        if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
            print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

    #clustering the data       
    stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality,random_state=random_state)
    

    model_param['Used cross-validation technique']=cross_val_split
    if cross_val_split != 'LGO': model_param['Number of folds']=outer_folds
    #use GKF for regression instead of stratified group kfold
    if task=='Regression' and cross_val_split=='SKF':
        cross_val_split='GKF'

    if outer_folds>km_groups:
        if km_groups>2:
            outer_folds=km_groups-1
        else:
            km_groups=outer_folds+1

    results_dictionary={}
    #empty to put this one first (keep in mind order in dictionary is not reliable)
    for index,p in enumerate(train_properties):
        results_dictionary[p]={}
        sample_weight=None
        if use_sample_weight:
            sample_weight=stacked_model.Train[f'sw_{p}'].values

        features=used_features[index]
        if 'rdkit' in features and not check_rdkit_desc:
            print('Warning, smiles are not validated for nan rdkit descriptors')

        dim_list=None
        if red_dim_list_per_prop is not None:
            dim_list=red_dim_list_per_prop[index]
            if isinstance(dim_list[0],list) and not local_dim_red:
                dim_list=None
            elif isinstance(dim_list[0],list):
                assert len(dim_list)==len(features), 'Number of provided features must equal number of dimensionality reduction method lists in the case of featurewise specific dimensionality reduction / feature selection'       
        params_grid,blender_params = param_Factory.get_feature_params(selected_features=features,dim_list=dim_list)

        stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                    params_grid=params_grid,
                                    paramsearch=paramsearch,
                                   features=features,
                                  scoring=scorer,
                                  cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                  use_memory=True,
                                  plot_validation=False, 
                                 refit=False,# no refit with validation. comes later,
                                 blender_params=blender_params
                                  ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
    model_str=stacked_model.print_metrics()
    
    smiles_list=stacked_model.Validation[standard_smiles_column]
    out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)
    if task=='Regression':
        if transformed_data and revert_to_original:
            if prop_cliff_dict is not None:
                original_prop_cliff_dict=prop_cliff_dict.copy()
                prop_cliff_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_cliff_dict.items()}
            properties=original_props
        else:
            properties=train_properties
        
        if cutoff is None:
            min_ytrue=[float(np.nanmin(list(stacked_model.Validation[p].values))) for p in properties]
            max_ytrue=[float(np.nanmax(list(stacked_model.Validation[p].values))) for p in properties]
            #set cutoffs for classification
            cutoff=[float(np.round(min_ytrue[i]+ (max_ytrue[i]-min_ytrue[i])/2,2)) for i,p in enumerate(properties)]
    
        fig_l=PlotlyDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],
                                                      prop_cliffs=prop_cliff_dict, leave_grp_out=groups_left_out,fig_size=(x_width,y_height),
                                                      smiles=list(smiles_list.values), apply_tick_transformation=apply_tick_transformation
                                              ,results_dict=results_dictionary)

        fig_adv=PlotlyDesigner().show_additional_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],
                                                                   prop_cliffs=prop_cliff_dict, leave_grp_out=groups_left_out,fig_size=(x_width,y_height),
                                                                   smiles=list(smiles_list.values), apply_tick_transformation=apply_tick_transformation
                                              ,results_dict=results_dictionary)

        fig_th=PlotlyDesigner().show_reg_cutoff_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],
                                                                    fig_size=(x_width,y_height),cutoff=cutoff,good_class=positive_cls,
                                                                smiles=list(smiles_list.values), apply_tick_transformation=apply_tick_transformation
                                              ,results_dict=results_dictionary )
    else: 
        properties=train_properties
        youden_dict,fig_l=PlotlyDesigner().show_classification_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties], labelnames=labelnames,cmap=cmap,fig_size=(x_width,y_height)
                                              ,results_dict=results_dictionary)
        fig_adv=PlotlyDesigner().show_additional_classification_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties], labelnames=labelnames,cmap=cmap,fig_size=(x_width,y_height)
                                              ,results_dict=results_dictionary)
        _,fig_th=PlotlyDesigner().show_clf_threshold_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],youden_dict=youden_dict, labelnames=labelnames,fig_size=(x_width,y_height)
                                              ,results_dict=results_dictionary)

    plotly_dictionary={**fig_l,**fig_th,**fig_adv}

    _,_,available_types=create_figure_captions(plotly_dictionary.keys())
    
            
    if figure_types is None:
        if task=='Regression':
            figure_types=available_types[:1]
        else:
            figure_types=available_types[:2]
    
    #retrieve figure keys from types ordered per property
    selected_figures=get_figures_from_types(plotly_dictionary=plotly_dictionary,properties=properties,types=figure_types)

    captions,types,_=create_figure_captions(selected_figures)
    pdf_data=generate_report(plotly_dictionary,selected_figures,captions,types,title=title, authors=authors, mails=mails, summary=summary, model_param=model_param,
                             model_tostr_list=model_str, properties=properties, template_file=None, nb_columns=nb_columns )
    with open(output_file, mode="wb") as f:
        f.write(pdf_data)
        
    save_result_dictionary_to_json(result_dict=results_dictionary,json_file=json_file) 
    
    if transformed_data and revert_to_original:
        properties=train_properties

    for p in properties:
        sample_train=None
        sample_val=None
        if use_sample_weight:
            sample_train=stacked_model.Train[f'sw_{p}'].values
            sample_val=stacked_model.Validation[f'sw_{p}'].values
        stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)
    ## clean the class first by removing the computed features
    stacked_model.deep_clean()
    stacked_model.compute_SD=True
    save_model(stacked_model, save_model_to_file)
