"""functionality to standardize the smiles, transform properties and built classes.


Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""


import pandas as pd
import numpy as np

from .feature_generators import RDKITGenerator
from .standardize import standardize

def add_stereo_smiles(df, column_name,verbose=False,outname='stereo_SMILES'):
    """
    function that adds chemaxon standardized SMILES to the pandas dataframe
    
    Chemaxon functionality removed, defaulting to rdkit
    
    Args:
         df: dataframe with the data
         column_name: name of the column containing the smiles
         verbose: set to get print statement
         outname: column name of the standardized smiles
    """
    add_rdkit_standardized_smiles(df=df, column_name=column_name,verbose=verbose,outname=outname)
    
def add_rdkit_standardized_smiles(df, column_name,verbose=False,outname='stereo_SMILES',remove_stereo=False):
    """
    function that adds rdkit standardized SMILES to the pandas dataframe
    
    AutoMol provided rdkit standardization is used
    
    Args:
         df: dataframe with the data
         column_name: name of the column containing the smiles
         verbose: set to get print statement
         outname: column name of the standardized smiles
         remove_stereo: remove_stereo option used in the call to rdkit_standardize_molecule_smiles
    """
    df[outname] = [standardize(smi) if type(smi)==str else np.nan for smi in df[column_name]]
    if verbose: print(' {} smiles out of {} are standardized using rdkit'.format(df[outname].notnull().sum(), len(df)))

    
def valid_rdkit(smi,rdkitgen):
    """
    function that checks if any descriptor is nan
    
    Args:
         smi: the smiles
         rdkitgen: rdkit feature generator
    
    Returns: 
        True is any of the descriptor is nan
    """
    desc=rdkitgen.get_descriptor(smi)
    return not np.isnan(desc).any()

def validate_rdkit_smiles(df, column_name,verbose=False):
    """
    function that sets smiles to nan if one of the rdkit descriptors is nan for that smiles
    
    Args:
         df: pandas dataframe
         column_name: smiles column 
    """
    rdkitgen=RDKITGenerator()
    df[column_name] = [smi if valid_rdkit(smi,rdkitgen) else np.nan for smi in df[column_name]]
    if verbose: print(' {} smiles out of {} are valid for rdkit descriptors'.format(df[column_name].notnull().sum(), len(df)))
    
    
def make_category(df, pro, bins=None, quantile=[0.1] ,prefix='Class', outname=None,verbose=False,precision=3):
    """
     assigns samples to classes based on regression values
     
    Args:
          df: pandas dataframe
          pro: property/column of dataframe with values
          bins: list of values/thresholds to divide target in classes, if None, quantiles are used
          quantile: list of quantiles values to divide samples in classes
          prefix: prefix for new column name of classed property
          outname: set the column name of the classed property
          verbose: boolean to indicate verbosity
          precision: precision of storage of bins
    """
    if not bins:
        bins=df[pro].dropna().quantile(quantile).tolist()
        th=bins
    min_pro= df[pro].min()#-np.inf#
    if bins[0]>min_pro:
        bins=[min_pro] + bins
    max_pro= df[pro].max()
    if bins[-1]<max_pro:
        bins=bins +[max_pro]
    #bins=bins+[np.inf] # new class for nan
    labels_id=[i for i in range(len(bins) -1)]
    labels_id2name={i:f'{round(bins[i],2)} to {round(bins[i+1],2)}' for i in labels_id }
    #labels_id2name[labels_id[-1]]=np.nan
    if outname is None:
        outname= f'{prefix}_{pro}'
    df[outname] = pd.cut(df[pro], bins=bins,labels=labels_id,include_lowest=True,precision=precision)
    df[outname][df[pro].isna()]=np.nan
    df[outname] = pd.to_numeric(df[outname], errors='coerce')
    if verbose: 
        print('Values count of the classes:')
        print(df[outname].value_counts())
    return labels_id2name

from math import log10,log
def v2log(x):
    """
    value to log10 function
    
    Args:
         x: the value x
    
    Returns: 
        log10(x) or np.nan
    """
    try:
        return log10(x)
    except:
        np.nan

def divide_by_100(x):
    """
    divide value by 100
    
    Args:
         x: the value x
    
    Returns: 
        x/100 or np.nan
    """
    try:
        return x/100.0
    except:
        np.nan

def logit(x):
    """
    value to logit function
    
    Args:
         x: the value x
    
    Returns: 
        log(x/(1-x)) or np.nan
    """
    try:
        return log(x/(1-x))
    except:
        np.nan


class PropertyBuilder:
    """
    Class to transform or create classes from given properties such that the AutoML package can work with them. 
    """
       
    def __init__(self,track_warnings=False):
        """
        initialization
        
        Args:
             track_warnings: boolean to track warnings
        """
        self.track_warnings=track_warnings
        self.warning_msgs=[]
        
    def retrieve_warnings(self):
        """
        Returns: 
            returns the accumulated warning messages
        """
        return self.warning_msgs
    
    def clear_warnings(self):
        """
        clears the warning messages
        """
        self.warning_msgs=[]
        
    def print_and_track_warning(self,warn_msg):
        """
        print and adds the warning message to the list
        
        Args:
             warn_msg: warning message
        """
        print(warn_msg)
        if self.track_warnings:
            self.warning_msgs.append(warn_msg)
        
    def retrieve_and_clear_warnings(self,messages=[]):
        """
        retrieve and clears the messages
        
        Args:
             messages: list of messages
        
        Returns: 
            the list of accumulated warnings messages
        """
        messages=self.retrieve_warnings().copy()
        self.clear_warnings()
        return messages        
    
    def check_properties(self,df,min_num_prop=100):
        """
        checks if the properties has at least a minimum number of props
        
        Args:
             df: dataframe
             min_num_prop: minimum number of props
        """
        #some property checks
        for i,p in enumerate(self.properties):
            prop_count=df[p].count()
            if prop_count<min_num_prop:
                self.print_and_track_warning(f'Warning: property {p} has less than 100 valid values')
                    
    
    def generate_train_properties(self,df):
        """
        empty method, to be implemented to create training properties
        
        Args:
             df: dataframe
        """
        pass
    
    def generate_sample_weights(self,df,weighted_samples_index,selected_sample_weights):
        """
        empty method, generates sample_weights
        
        Args:
             df: dataframe:
             weighted_samples_index: dictionary reflecting which samples should be asigned a given weight
             sample_weights: dictionary with property keys, setting the weight of the samples select by the param weighted samples_index
        """
        pass
        
        
class ClassBuilder(PropertyBuilder):
    """
    a class to build the training properties for classification
    """
    
    def __init__(self,properties,nb_classes,class_values,categorical,use_quantiles,prefix='Class',min_allowed_class_samples=30,verbose=False,track_warnings=False):
        """
        specialization of the PropertyBuilder for classification
        
        Args:
             properties: list of properties to be fitted
             nb_classes: list of the number of classes for each property
             class_values: list of list of class cutoffs or quantiles
             use_quantiles: boolean to indicate if the param class_values contains quantiles or absolute values
             prefix: prefix added to the property name for the column name holding the class
             min_allowed_class_samples: the minimum allowed samples per class
             track_warnings: boolean to track warnings
        """
        if len(properties)==1:
            if not isinstance(nb_classes,list):
                nb_classes=[nb_classes]
            if not isinstance(class_values,list):
                    class_values=[class_values]
            if not isinstance(class_values[0],list):
                class_values=[class_values]
        ## the list of properties        
        self.properties=properties
        ## the list of number of classes
        self.nb_classes=nb_classes
        ## the list of class thresholds
        self.class_values=class_values
        ## prefix added to the properties
        self.prefix=prefix
        ## dictionary of labelnames 
        self.labelnames={}
        ## dictionary mapping labels to indices
        self.index_map={}
        ## boolean to indicate categorical properties
        self.categorical=categorical
        ## boolena to indicate use of quantiles
        self.use_quantiles=use_quantiles
        ## minimum allowed number of samples
        self.min_allowed_class_samples=min_allowed_class_samples
        self.verbose=verbose
        super().__init__(track_warnings)
        
    def check_properties(self,df,min_num_prop=100):
        """
        checks the properties
        
        Args:
             df: dataframe
             min_num_prop: minimum number of allowed samples in a property
        """
        #some property checks
        super().check_properties(df,min_num_prop)
        for i,p in enumerate(self.properties):
            prop_count=df[p].count()
            if prop_count<self.nb_classes[i]*self.min_allowed_class_samples:
                self.print_and_track_warning(f'Warning: property {p} does not have enough samples to support {self.nb_classes} classes of {self.min_allowed_class_samples} samples')
                
    def generate_train_properties(self,df,precision=3):
        """
        create the classes
        
        Args:
             precision: the precision of the class thresholds in the labels
        
        Returns: 
            tuple of update dataframe and training properties
        """
        self.class_properties=[self.prefix+'_'+p for p in self.properties]
        
        #continuous data
        if not self.categorical:
            #creating classes
            for index,p in enumerate (self.properties):
                df[p] = pd.to_numeric(df[p], errors='coerce')
                if self.use_quantiles:
                    self.labelnames[self.class_properties[index]]=make_category(df, pro=p,  quantile=self.class_values[index],bins=None,prefix=self.prefix, outname=None, verbose=self.verbose,precision=precision)
                else:
                    self.labelnames[self.class_properties[index]]=make_category(df, pro=p, bins=self.class_values[index],prefix=self.prefix, outname=None,verbose=self.verbose,precision=precision)

            #Verify number of samples per class.
            for i,p in enumerate(self.class_properties):
                redefine=False
                val_cnts=df[[p]].value_counts()
                if not (len(val_cnts)==self.nb_classes[i]):
                    redefine=True
                else:
                    for j in range(len(val_cnts)):
                        nb_in_class=val_cnts.iloc[j]
                        if nb_in_class<self.min_allowed_class_samples:
                            redefine=True
                            break
                if redefine:
                    step=0.8/(self.nb_classes[i])
                    quantiles= np.round([0.1+(i+1)*step for i in range(self.nb_classes[i]-1)],3)
                    self.print_and_track_warning(f'Not enough samples in classes or number of classes is not equal to the requested number of samples, using quantiles {quantiles} for classes for property {p}')
                    self.labelnames[p]=make_category(df, pro=self.properties[i],  quantile=quantiles,prefix=self.prefix, outname=None,precision=precision)
        else:
            self.class_properties=self.properties
            for index,p in enumerate(self.properties):
                mask=df[p].isna()
                classes=np.sort(df[p][~mask].unique())
                if len(classes)!=self.nb_classes[index]:
                    self.print_and_track_warning(f'Changing number of classes of {p} from {self.nb_classes[index]} to {len(classes)}')
                    self.nb_classes[index]=len(classes)
                class_mapping={key:str(val) for key,val in enumerate(classes)}
                index_mapping={str(val):key for key,val in enumerate(classes)}
                df[self.class_properties[index]] =[ np.nan if str(v)=='nan' else index_mapping[str(v)] for v in df[p]]
                self.labelnames[self.class_properties[index]]=class_mapping
                self.index_map[self.class_properties[index]]=index_mapping
            self.properties=self.class_properties
        
        for ip,p in enumerate(self.class_properties):
            if not np.sum(df[p].value_counts())==len(df)-np.sum(df[self.properties[ip]].isna()):
                self.print_and_track_warning(f'Warning, the sum of the number of sample classes in {p} is not equal to the number of valid samples in {self.properties[ip]}')
            if not np.sum(df[p].isna())==np.sum(df[self.properties[ip]].isna()):
                self.print_and_track_warning(f'Warning, the number of NaN samples in {p} is not equal to the number of NaN samples in {self.properties[ip]}')
        print(f'Class value to label mapping: {str(self.labelnames)}')
        return df,self.class_properties
    
    def generate_sample_weights(self,df,weighted_samples_index,selected_sample_weights):
        """
        creates the sample weights for classification

        Args:
             weighted_samples_index: for categorical data, a dictionary with the property and class label. For continous data, the property and class index
             selected_sample_weights: the weights for each property as an dictionary
        
        Returns: 
            the dataframe with the sample weights in the columns sw_<prop> for property <prop>
        """
        if self.categorical:
            assert self.index_map, 'call generate_train_properties first, index_map not yet set' 
            weighted_samples_index=[self.index_map[p][weighted_samples_index[p]] for index,p in enumerate(self.class_properties)]
        else:
            weighted_samples_index=[weighted_samples_index[p] for index,p in enumerate(self.class_properties)]
            
        for index,p in enumerate(self.class_properties):
            sample_weight=np.ones(len(df))
            sample_weight[df[p]==weighted_samples_index[index]]=selected_sample_weights[p]
            df[f'sw_{p}'] = sample_weight

        return df
    
    
class PropertyTransformer(PropertyBuilder):
    """
    Class to potentially transform the data for regression
    """
    
    def __init__(self,properties,remove_outliers=False,confidence=0.0,use_log10=False,use_logit=False,percentages=False,standard_smiles_column='stereo_SMILES',track_warnings=False):
        """
        Specialization of PropertyBuilder for regression that transform the given properties based on the given booleans
        
        Args:
             properties: the list of properties to be fitted
             remove_outliers: [DEPRECATED] boolean to indicate removal of outliers 
             confidence: [DEPRECATED] parameter to define interval of non-outliers, interval is determined by mean +- confidence x standard deviation
             use_log10: boolean to transform data using log10
             use_logit: boolean to transform data using logit
             percentages: boolean to divide properties by 100
             standard_smiles_column: column name in the df with the standardized smiles
             track_warnings: boolean to track warnings
        """
        ## list of properties
        self.properties=properties
        ## deprecated and unused
        self.remove_outliers=remove_outliers
        ## deprecated and unused
        self.confidence=confidence
        ## boolean to apply log10 transformation
        self.use_log10=use_log10
        ## boolean to apply logit transformation
        self.use_logit=use_logit
        ## boolean to divide properties by 100
        self.percentages=percentages
        ## None labelnames for regression
        self.labelnames=None
        ## smiles column containing the standardized smiles
        self.standard_smiles_column=standard_smiles_column
        super().__init__(track_warnings)
    
    def generate_train_properties(self,df):
        """
        potentially transform the properties
        
        Args:
             df: dataframe
        
        Returns: 
            tuple of updated dataframe and training properties
        """
        log_props=[]     
        
        for c in self.properties:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            #compute log10
            if self.use_log10:
                nb_zero=np.count_nonzero(df[c]==0)
                #zero values are removed when using log10
                if nb_zero>0: 
                     self.print_and_track_warning(f'Number of zero values property {c}: {nb_zero}')
                df[f'log10_{c}']=df[c].apply(v2log)
                log_props.append(f'log10_{c}')
            elif self.use_logit:
                prop_name=c
                mask=df[prop_name].isna()
                if self.percentages:
                    prop_name=prop_name.replace("(%) ", "")
                    df[prop_name]=df[c].apply(divide_by_100)
                nb_neg=np.count_nonzero(df[prop_name]<0)
                if nb_neg>0:
                    self.print_and_track_warning(f'Warning: Property {c} has {nb_neg} values less than 0 and these are clipped to 0')
                nb_lt1=np.count_nonzero(df[prop_name]>=1)
                if nb_lt1>0:
                    if self.percentages:
                        self.print_and_track_warning(f'Warning: Property {c} has {nb_lt1} values larger or equal to 100 and these are clipped to {(1-1e-8)*100}')
                    else:
                        self.print_and_track_warning(f'Warning: Property {c} has {nb_lt1} values larger or equal to 1 and these are clipped to {1-1e-8}')
                df[prop_name]=np.clip(df[prop_name], 0, 1-1e-8)
                df[f'logit_{prop_name}']=df[prop_name].apply(logit)
                df.loc[mask,prop_name]=np.nan
                log_props.append(f'logit_{prop_name}')

        if self.use_logit and self.percentages and not self.use_log10:
            self.properties=[p.replace("(%) ", "") for p in self.properties]
        
        transformed_data=self.use_log10 or self.use_logit
        
        #switch log10 properties as active
        if transformed_data:
            self.original_props=self.properties
            self.properties=log_props
            
        if self.remove_outliers:
            self.print_and_track_warning(f'Deprecated functionality, outlier removal is no longer part of the AutoMol package')
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True,subset =self.properties,how='all' )
        df.reset_index(drop=True,inplace=True)
        return df,self.properties
            
    def generate_sample_weights(self,df,weighted_samples_index,selected_sample_weights):
        """
        generates sample weights for regression
        
        Args:
             weighted_samples_index: dictionary with as keys the properties and as values a str with a quantifies (< or >) and a cutoff value, e.g. '>5.4'
             selected_sample_weights: dictionary of weights with property name as key
        
        Returns: 
            updated dataframe with sample weights in columns starting with 'sw_'
        """
        for index,p in enumerate(self.properties):
            sample_weight=np.ones(len(df))
            cutoff_val=float(weighted_samples_index[p][1:])

            if self.use_log10:
                cutoff_val=v2log(cutoff_val)
            elif self.use_logit:
                if self.percentages:
                    cutoff_val=divide_by_100(cutoff_val)
                cutoff_val=logit(cutoff_val)
            if cutoff_val==np.nan:
                self.print_and_track_warning(f'Cutoff value is nan after transformation for property {p}, using 0 as default')
                cutoff_val=0   

            if weighted_samples_index[p][0]=='>':
                sample_weight[df[p]>cutoff_val]=selected_sample_weights[p]
            else:
                if not weighted_samples_index[p][0]=='<':
                    print(f'{weighted_samples_index[p][0]} is not in < or > for property {p}, using < as default')
                sample_weight[df[p]<cutoff_val]=selected_sample_weights[p]

            df[f'sw_{p}'] = sample_weight
        
        return df
