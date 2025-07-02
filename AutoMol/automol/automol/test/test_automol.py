import warnings
import sys, os 
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt
from pkg_resources import resource_filename

from automol.property_prep import add_stereo_smiles,validate_rdkit_smiles, make_category
from automol.stacking_util import ResultDesigner        
from automol.stacking import save_model, load_model
from automol.stacking_util import add_lgbm_xtimes_hyperopt, add_xgb_xtimes_hyperopt
from automol.stacking_methodarchive import ClassifierArchive,ReducedimArchive,RegressorArchive
from automol.property_prep import ClassBuilder
from automol.property_prep import PropertyTransformer
from automol.feature_generators import retrieve_default_offline_generators, FeatureGenerator, ECFPGenerator
from automol.stacking_util import ModelAndParams   
from automol.clustering import ClusteringAlgorithm, MurckoScaffoldClustering, ButinaSplitReassigned, HierarchicalButina, KmeansForSmiles

from automol.test.utils_for_tests import *

verbose=0
data = resource_filename('automol.test', 'ChEMBL_SMILES.csv')
smiles_col='smiles'
dv_properties=['prop1','prop2','prop3','prop4']
cat_properties=['prop5']

class automol_UnitTests(TestCase):
    
    def setUp(self):
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
        warnings.simplefilter('ignore')
        # Other initialization stuff here
    
    def test_a_a_make_category(self):
        print('----------------------------------------------------------------------------------')
        print('---------------------------test_a_a_make_category---------------------------------')
        print('----------------------------------------------------------------------------------')
        df = pd.DataFrame(data={'p': [-2.0001,-1.99991,-1.5,-1,0,0,1, 1, 2,2,2,np.nan,np.nan,None]})
        make_category(df, 'p', bins=[-1.9999,0,2], quantile=None ,prefix='Class', outname='classes',precision=3)
        for i in range(2):
            self.assertEqual(df['classes'][i]==0,True)
        for i in range(6)[2:]:
            self.assertEqual(df['classes'][i]==1,True)
        for i in range(11)[6:]:
            self.assertEqual(df['classes'][i]==2,True)
        for i in range(14)[11:]:
            self.assertEqual(np.isfinite(df['classes'][i]),False)
        self.assertEqual(np.sum(df['classes'].value_counts()),len(df)-np.sum(df['p'].isna()) ,True)
        make_category(df, 'p', bins=[0], quantile=None ,prefix='Class', outname='classes')
        for i in range(6):
            self.assertEqual(df['classes'][i]==0,True)
        for i in range(11)[6:]:
            self.assertEqual(df['classes'][i]==1,True)
        for i in range(14)[11:]:
            self.assertEqual(np.isfinite(df['classes'][i]),False)
        self.assertEqual(np.sum(df['classes'].value_counts()),len(df)-np.sum(df['p'].isna()) ,True)
        make_category(df, 'p', bins=[0,1], quantile=None ,prefix='Class', outname='classes')
        for i in range(6):
            self.assertEqual(df['classes'][i]==0,True)
        for i in range(8)[6:]:
            self.assertEqual(df['classes'][i]==1,True)
        for i in range(11)[8:]:
            self.assertEqual(df['classes'][i]==2,True)
        for i in range(14)[11:]:
            self.assertEqual(np.isfinite(df['classes'][i]),False)
        self.assertEqual(np.sum(df['classes'].value_counts()),len(df)-np.sum(df['p'].isna()) ,True)
        make_category(df, 'p', bins=None, quantile=[0.2] ,prefix='Class', outname='classes')
        for i in range(3):
            self.assertEqual(df['classes'][i]==0,True)
        for i in range(11)[3:]:
            self.assertEqual(df['classes'][i]==1,True)
        for i in range(14)[11:]:
            self.assertEqual(np.isfinite(df['classes'][i]),False)
        self.assertEqual(np.sum(df['classes'].value_counts()),len(df)-np.sum(df['p'].isna()) ,True)
        
    def test_a_a_reg_transformations(self):
        print('----------------------------------------------------------------------------------')
        print('--------------------------test_a_a_reg_transformations---------------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()

        properties=['plain','plain_nan']
        logit_df = pd.DataFrame(data={'plain': [0.01,0.05,0.1,0.5,1,1.1,0.001,0.9],'plain_nan': [0.01,0.05,0.1,0.5,1,1.1,np.nan,None],smiles_col: ['C','C','C','C','C','C','C','C']})
        true_val=[-4.5951198,-2.9444389,-2.1972245,0.000000,18.4206807,18.4206807,-6.9067547,2.1972245]
        use_log10=False
        percentages=False
        use_logit=True
        remove_outliers=False
        confidence=3.8  #3.8
       
        logit_trans=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=smiles_col)
        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=logit_trans.generate_train_properties(logit_df)
        
        self.assertEqual(np.allclose(df['logit_plain'], true_val),True)
        self.assertEqual(np.allclose(df['logit_plain_nan'][:6], true_val[:6]),True)
        for i in range(8)[6:]:
            self.assertEqual(np.isfinite(df['logit_plain_nan'][i]),False)
            
            
        properties=['perc','perc_nan']        
        logit_df = pd.DataFrame(data={'perc': [1,5,10,50,100,110,0.1,90],'perc_nan': [1,5,10,50,100,110,np.nan,None],smiles_col: ['C','C','C','C','C','C','C','C']})
        percentages=True
        logit_trans=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=smiles_col)
        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=logit_trans.generate_train_properties(logit_df)
        
        self.assertEqual(np.allclose(df['logit_perc'], true_val),True)
        self.assertEqual(np.allclose(df['logit_perc_nan'][:6], true_val[:6]),True)
        for i in range(8)[6:]:
            self.assertEqual(np.isfinite(df['logit_perc_nan'][i]),False)
                 
        use_log10=True
        percentages=False
        use_logit=False
            
        properties=['plain','plain_nan']        
        log10_df = pd.DataFrame(data={'plain': [0.1,0.5,1,5,10,11,0.01,9],'plain_nan': [0.1,0.5,1,5,10,0,np.nan,None],smiles_col: ['C','C','C','C','C','C','C','C']})
        log10_trans=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=smiles_col)
        true_val=[-1,-0.3010299,0,0.6989700,1,1.0413926,-2,0.9542425]
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=log10_trans.generate_train_properties(log10_df)
        
        self.assertEqual(np.allclose(df['log10_plain'], true_val),True)
        self.assertEqual(np.allclose(df['log10_plain_nan'][:5], true_val[:5]),True)
        for i in range(8)[5:]:
            self.assertEqual(np.isfinite(df['log10_plain_nan'][i]),False)
        
    def test_a_a_offline_feature_generators(self):
        print('----------------------------------------------------------------------------------')
        print('---------------------test_a_a_offline_feature_generators--------------------------')
        print('----------------------------------------------------------------------------------')
        
        for key,item in retrieve_default_offline_generators(radius=2,nbits=2048).items():
            X=item.generate(['[Cu]', 'C[C@H](N)C(=O)O', 'CC(O)C(=O)O', 'OCC(O)CO', 'Oc1ccccc1','Nc1ccncc1', None, 'C#CC(C)(O)CC', '', 'ClCCCl', 'O=C1CCC(=O)N1', 'O=CCCCC=O','N[C@@H]1CONC1=O', 'S1C=CSC1=C'])
            self.assertTrue(X.shape[1]==item.get_nb_features())
            
    def test_a_a_kmeans_clustering(self):
        print('----------------------------------------------------------------------------------')
        print('-------------------------test_a_a_kmeans_clustering-------------------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()    
        
        standard_smiles_column='SMILES_STEREO'
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=False, check_rdkit_desc=False)
        df_smiles=df[standard_smiles_column]
        feat_gen=retrieve_default_offline_generators(radius=2,nbits=2048)
        for key,item in feat_gen.items():
            print(f'Testing: {key}')
            if item.get_nb_features()<3000:
                clustering_algorithm=KmeansForSmiles(n_groups=10,feature_generators=feat_gen,used_features=[key],random_state=42)
                clustering_algorithm.cluster(df_smiles)
                groups=clustering_algorithm.get_groups()
                self.assertTrue(len(df_smiles)==len(groups))
                generated_feat=clustering_algorithm.get_generated_features()
                for gen_key,gen_item in generated_feat.items():
                    self.assertTrue(gen_key==key)
                    self.assertTrue(gen_item['X'].shape==(len(df_smiles),item.get_nb_features()))
                    
        for key,item in feat_gen.items():
            print(f'Testing: {key}')
            if item.get_nb_features()<3000:
                clustering_algorithm=KmeansForSmiles(n_groups=10,feature_generators={key:RandomNanInjector(item)},used_features=[key],random_state=42)
                clustering_algorithm.cluster(df_smiles)
                groups=clustering_algorithm.get_groups()
                self.assertTrue(len(df_smiles)==len(groups))
                generated_feat=clustering_algorithm.get_generated_features()
                for gen_key,gen_item in generated_feat.items():
                    self.assertTrue(gen_key==key)
                    self.assertTrue(gen_item['X'].shape==(len(df_smiles),item.get_nb_features()))            
                                        
        
    def test_a_a_butina_clustering(self):
        print('----------------------------------------------------------------------------------')
        print('-------------------------test_a_a_butina_clustering-------------------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()    
        
        standard_smiles_column='SMILES_STEREO'
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=False, check_rdkit_desc=False)
        df_smiles=df[standard_smiles_column]

        clustering_algorithm=ButinaSplitReassigned(cutoff = 0.65)
        clustering_algorithm.cluster(df_smiles)
        groups=clustering_algorithm.get_groups()
        self.assertTrue(len(df_smiles)==len(groups))
        generated_feat=clustering_algorithm.get_generated_features()
        for gen_key,gen_item in generated_feat.items():
            self.assertTrue(gen_key=='fps_1024_2')
            self.assertTrue(gen_item['X'].shape==(len(df_smiles),1024))
                    
        
        clustering_algorithm=ButinaSplitReassigned(cutoff = 0.65,feature_generator=ECFPGenerator(radius=2, nBits =2048))
        clustering_algorithm.cluster(df_smiles)
        groups=clustering_algorithm.get_groups()
        self.assertTrue(len(df_smiles)==len(groups))
        generated_feat=clustering_algorithm.get_generated_features()
        for gen_key,gen_item in generated_feat.items():
            self.assertTrue(gen_key=='fps_2048_2')
            self.assertTrue(gen_item['X'].shape==(len(df_smiles),2048))
        
        
    def test_a_a_hierarchicalbutina_clustering(self):
        print('----------------------------------------------------------------------------------')
        print('-------------------------test_a_a_hierarchicalbutina_clustering-------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()    
        
        standard_smiles_column='SMILES_STEREO'
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=False, check_rdkit_desc=False)
        df_smiles=df[standard_smiles_column]

        clustering_algorithm=HierarchicalButina(cutoff = [0.4,0.65])
        clustering_algorithm.cluster(df_smiles)
        groups=clustering_algorithm.get_groups()
        self.assertTrue(len(df_smiles)==len(groups))
        generated_feat=clustering_algorithm.get_generated_features()
        for gen_key,gen_item in generated_feat.items():
            self.assertTrue(gen_key=='fps_1024_2')
            self.assertTrue(gen_item['X'].shape==(len(df_smiles),1024))
                    
        
        clustering_algorithm=HierarchicalButina(cutoff = [0.4,0.65],feature_generator=ECFPGenerator(radius=2, nBits =2048))
        clustering_algorithm.cluster(df_smiles)
        groups=clustering_algorithm.get_groups()
        self.assertTrue(len(df_smiles)==len(groups))
        generated_feat=clustering_algorithm.get_generated_features()
        for gen_key,gen_item in generated_feat.items():
            self.assertTrue(gen_key=='fps_2048_2')
            self.assertTrue(gen_item['X'].shape==(len(df_smiles),2048))
        
    def test_a_a_murckoscaffold_clustering(self):
        print('----------------------------------------------------------------------------------')
        print('-------------------------test_a_a_murckoscaffold_clustering-----------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
        
        standard_smiles_column='SMILES_STEREO'
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=False, check_rdkit_desc=False)
        df_smiles=df[standard_smiles_column]

        clustering_algorithm=MurckoScaffoldClustering(include_chirality=True)
        clustering_algorithm.cluster(df_smiles)
        groups=clustering_algorithm.get_groups()
        self.assertTrue(len(df_smiles)==len(groups))
                            
        clustering_algorithm=MurckoScaffoldClustering(include_chirality=False)
        clustering_algorithm.cluster(df_smiles)
        groups=clustering_algorithm.get_groups()
        self.assertTrue(len(df_smiles)==len(groups))
        
    def test_c_model_delete_and_merge(self):
        print('----------------------------------------------------------------------------------')
        print('------------------------test_c_model_delete_and_merge-----------------------------')
        print('----------------------------------------------------------------------------------')
        model_f='cat_model.pt'
        stacked_model= load_model( model_f,use_gpu=False)
        if len(stacked_model.models)>1:
            prediction_before=stacked_model.predict(['CC','COC'])
            properties=[p for p in stacked_model.models]
            stacked_model2= load_model( model_f,use_gpu=False)
            stacked_model.delete_properties(properties[0])
            self.assertFalse(properties[0] in stacked_model.models)
            stacked_model.merge_model(other_model=stacked_model2,other_props=[properties[0]])
            prediction_after=stacked_model.predict(['CC','COC'])
            for key,item in prediction_before.items():
                self.assertTrue(key in prediction_after)
                for val1,val2 in zip(item,prediction_after[key]):
                    self.assertTrue(val1==val2)
        model_f='regclf_model.pt'
        stacked_model= load_model( model_f,use_gpu=False)
        if len(stacked_model.models)>1:
            prediction_before=stacked_model.predict(['CC','COC'])
            properties=[p for p in stacked_model.models]
            stacked_model2= load_model( model_f,use_gpu=False)
            stacked_model.delete_properties(properties[0])
            self.assertFalse(properties[0] in stacked_model.models)
            stacked_model.merge_model(other_model=stacked_model2,other_props=[properties[0]])
            prediction_after=stacked_model.predict(['CC','COC'])
            for key,item in prediction_before.items():
                self.assertTrue(key in prediction_after)
                for val1,val2 in zip(item,prediction_after[key]):
                    self.assertTrue(val1==val2)
        model_f='clf_model.pt'
        stacked_model= load_model( model_f,use_gpu=False)
        if len(stacked_model.models)>1:
            prediction_before=stacked_model.predict(['CC','COC'])
            properties=[p for p in stacked_model.models]
            stacked_model2= load_model( model_f,use_gpu=False)
            stacked_model.delete_properties(properties[0])
            self.assertFalse(properties[0] in stacked_model.models)
            stacked_model.merge_model(other_model=stacked_model2,other_props=[properties[0]])
            prediction_after=stacked_model.predict(['CC','COC'])
            for key,item in prediction_before.items():
                self.assertTrue(key in prediction_after)
                for val1,val2 in zip(item,prediction_after[key]):
                    self.assertTrue(val1==val2)
        model_f='reg_model.pt'
        stacked_model= load_model( model_f,use_gpu=False)
        if len(stacked_model.models)>1:
            prediction_before=stacked_model.predict(['CC','COC'])
            properties=[p for p in stacked_model.models]
            stacked_model2= load_model( model_f,use_gpu=False)
            stacked_model.delete_properties(properties[0])
            self.assertFalse(properties[0] in stacked_model.models)
            stacked_model.merge_model(other_model=stacked_model2,other_props=[properties[0]])
            prediction_after=stacked_model.predict(['CC','COC'])
            for key,item in prediction_before.items():
                self.assertTrue(key in prediction_after)
                for val1,val2 in zip(item,prediction_after[key]):
                    self.assertTrue(val1==val2)
        
    def test_c_model_predict_output(self):
        print('----------------------------------------------------------------------------------')
        print('-------------------------test_b_model_predict_output------------------------------')
        print('----------------------------------------------------------------------------------')
        
        model_f='cat_model.pt'
        stacked_model= load_model( model_f,use_gpu=False) 
        pred=stacked_model.predict([])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(pred[k].shape==(0,),True)
        pred=stacked_model.predict('')
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
                self.assertEqual(np.isnan(pred[k][0]),True)
        pred=stacked_model.predict('C')
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC','','COC'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
                self.assertEqual(pred[k].shape[0]==3,True)
                self.assertEqual(np.isnan(pred[k][1]),True)
        pred=stacked_model.predict(['CC','O[Se](O)=O'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
        
        model_f='regclf_model.pt'
        stacked_model= load_model( model_f,use_gpu=False) 
        pred=stacked_model.predict([])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(pred[k].shape==(0,),True)
        pred=stacked_model.predict('')
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
                self.assertEqual(np.isnan(pred[k][0]),True)
        pred=stacked_model.predict('C')
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC','','COC'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
                self.assertEqual(pred[k].shape[0]==3,True)
                self.assertEqual(np.isnan(pred[k][1]),True)
        pred=stacked_model.predict(['CC','O[Se](O)=O'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
            
        model_f='clf_model.pt'
        stacked_model= load_model( model_f,use_gpu=False) 
        pred=stacked_model.predict([])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(pred[k].shape==(0,),True)
        pred=stacked_model.predict('')
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
                self.assertEqual(np.isnan(pred[k][0]),True)
        pred=stacked_model.predict('C')
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC','','COC'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
                self.assertEqual(pred[k].shape[0]==3,True)
                self.assertEqual(np.isnan(pred[k][1]),True)
        pred=stacked_model.predict(['CC','O[Se](O)=O'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)

        model_f='reg_model.pt'
        stacked_model= load_model( model_f,use_gpu=False)
        pred=stacked_model.predict([])
        for k in pred:
            self.assertEqual(pred[k].shape==(0,),True)
        pred=stacked_model.predict('')
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
            self.assertEqual(np.isnan(pred[k][0]),True)
        pred=stacked_model.predict('C')
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC'])
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC','','COC'])
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
            self.assertEqual(pred[k].shape[0]==3,True)
            self.assertEqual(np.isnan(pred[k][1]),True)
        pred=stacked_model.predict(['CC','O[Se](O)=O'])
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)

        
            
        pred_normal=stacked_model.predict(['C[C@H](N)C(=O)O', 'CC(O)C(=O)O', 'OCC(O)CO', 'Oc1ccccc1','Nc1ccncc1', 'C#CC(C)(O)CC','ClCCCl', 'O=C1CCC(=O)N1', 'O=CCCCC=O','N[C@@H]1CONC1=O'])
        stacked_model.add_predict_transformation_for_p(dv_properties[0],transformation=lambda a : a/3.0)
        stacked_model.add_predict_transformation_for_p(dv_properties[1],transformation=lambda a : a*5.0)
        stacked_model.add_predict_transformation_for_p(dv_properties[2],transformation=lambda a : a**2.0)
        stacked_model.add_predict_transformation_for_p(dv_properties[3],transformation=lambda a : a/np.linalg.norm(a))
        
        pred_transformed=stacked_model.predict(['C[C@H](N)C(=O)O', 'CC(O)C(=O)O', 'OCC(O)CO', 'Oc1ccccc1','Nc1ccncc1', 'C#CC(C)(O)CC','ClCCCl', 'O=C1CCC(=O)N1', 'O=CCCCC=O','N[C@@H]1CONC1=O'])
        
        for v1,v2 in zip(pred_normal[f'predicted_{dv_properties[0]}'],pred_transformed[f'predicted_{dv_properties[0]}']):
            self.assertTrue(np.abs(v2-v1/3.0)<1e-15)
        for v1,v2 in zip(pred_normal[f'predicted_{dv_properties[1]}'],pred_transformed[f'predicted_{dv_properties[1]}']):
            self.assertTrue(np.abs(v2-v1*5.0)<1e-15)
        for v1,v2 in zip(pred_normal[f'predicted_{dv_properties[2]}'],pred_transformed[f'predicted_{dv_properties[2]}']):
            self.assertTrue(np.abs(v2-v1**2)<1e-15)
        norm_factor=np.linalg.norm(pred_normal[f'predicted_{dv_properties[3]}'])
        for v1,v2 in zip(pred_normal[f'predicted_{dv_properties[3]}'],pred_transformed[f'predicted_{dv_properties[3]}']):
            self.assertTrue(np.abs(v2-v1/norm_factor)<1e-15)
        
        model_f='clfmulti_model.pt'
        stacked_model= load_model( model_f,use_gpu=False) 
        pred=stacked_model.predict([])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(pred[k].shape==(0,),True)
        pred=stacked_model.predict('')
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
                self.assertEqual(np.isnan(pred[k][0]),True)
        pred=stacked_model.predict([''])
        pred=stacked_model.predict('C')
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC','','COC'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
                self.assertEqual(pred[k].shape[0]==3,True)
                self.assertEqual(np.isnan(pred[k][1]),True)
        pred=stacked_model.predict(['CC','O[Se](O)=O'])
        for k in pred:
            if not 'label' in k:
                self.assertEqual(len(pred[k].shape)==1,True)
        
        model_f='regmulti_model.pt'
        stacked_model= load_model( model_f,use_gpu=False)
        pred=stacked_model.predict([])
        for k in pred:
            self.assertEqual(pred[k].shape==(0,),True)
        pred=stacked_model.predict('')
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
            self.assertEqual(np.isnan(pred[k][0]),True)
        pred=stacked_model.predict('C')
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC'])
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC','','COC'])
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
            self.assertEqual(pred[k].shape[0]==3,True)
            self.assertEqual(np.isnan(pred[k][1]),True)
        pred=stacked_model.predict(['CC','O[Se](O)=O'])
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
        
        model_f='rd_ecfp_model.pt'
        stacked_model= load_model( model_f,use_gpu=False) 
        pred=stacked_model.predict([])
        for k in pred:
            self.assertEqual(pred[k].shape==(0,),True)
        pred=stacked_model.predict('')
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
            self.assertEqual(np.isnan(pred[k][0]),True)
        pred=stacked_model.predict('C')
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC'])
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
        pred=stacked_model.predict(['CC','','COC'])
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
            self.assertEqual(pred[k].shape[0]==3,True)
            self.assertEqual(np.isnan(pred[k][1]),True)
        pred=stacked_model.predict(['CC','O[Se](O)=O'])
        for k in pred:
            self.assertEqual(len(pred[k].shape)==1,True)
            self.assertEqual(np.isnan(pred[k][1]),True)
    
    def test_z_clean_up(self):
        model_files=['cat_model.pt','clf_model.pt','reg_model.pt','rd_ecfp_model.pt','regclf_model.pt','clfmulti_model.pt','regmulti_model.pt']
        for f in model_files:
            os.remove(f)
######################################################################################
#                       Classification
######################################################################################
    
    def test_b_clf_ch_multiprop_clsv_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('-----------------test_clf_ch_multiprop_clsv_nosw_vkmac_cvkmgkf4-------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
        
        categorical=False
        properties=dv_properties
        nb_classes=[3,2,2,2]
        class_values= [[2,10],[6.7],[6.5],[5]]
        class_quantiles=None    
        min_allowed_class_samples=25
        use_sample_weight=False
        encoder='CHEMBL'
        task='Classification'
        computional_load='cheap'
        scorer=HammingMultioutputScore()
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        if computional_load=='expensive':
            use_sample_weight=False
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        interest_class=[0 for p in properties]
        save_model_to_file='clfmulti_model.pt'        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=False
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
            
        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=class_values,
                                 categorical=categorical,use_quantiles=class_quantiles is not None,
                                 prefix='Class',min_allowed_class_samples=30,verbose=verbose)

        #check properties
        prop_builder.check_properties(df)        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,train_properties=prop_builder.generate_train_properties(df)
        #retrieve labelnames 
        labelnames=prop_builder.labelnames
        
        class_properties=train_properties
                
        for ip,p in enumerate(class_properties):
            self.assertEqual(np.sum(df[p].value_counts()),len(df)-np.sum(df[properties[ip]].isna()) ,True)
        
        weighted_samples_index={p:0 for i,p in enumerate(train_properties)}
        if categorical:
            weighted_samples_index={p:'0.0' for p in train_properties}
        select_sample_weights={p:10 for p in train_properties}  

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)    

        #define possible blender/final estimator
        blender_list=['dtc']#,'rfc']
        #define base estimators or used method for single_method
        clf_list=['dtc']#, 'lda','rfc']#,'qda','sgdc','dtc','xgb']
        
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=False,
                                     blender_list=blender_list,
                                     method_list=clf_list,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list, verbose=verbose,
                                     n_jobs=n_jobs,
                                     labelnames=labelnames,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES
  
        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,class_properties,strategy,categorical,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
            
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print(f'Warning: on average less than 10 samples per cluster for property {p}, suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality,random_state=random_state)
        
        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        p=class_properties
        
        sample_weight=None
        if use_sample_weight:
            sample_weight=stacked_model.Train[f'sw_{p}'].values

        stacked_model.search_model(df= None,   prop=p,  smiles='stereo_SMILES',
                                    params_grid=params_grid,
                                   paramsearch=paramsearch,
                                  scoring=scorer,#'neg_mean_absolute_error',#
                                  cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                  use_memory=True,
                                  plot_validation=True, 
                                 refit=False,# no refit with validation. comes later,
                                 blender_params=blender_params
                                  ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()

        cmap=['PiYG','Blues']
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES,compute_SD=True,convert_log10=False)

        youden_dict=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        F1_dict=ResultDesigner().show_clf_threshold_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,youden_dict=youden_dict)
        
        [stacked_model.set_property_threshold(prop=p,class_index=interest_class[ip],threshold=F1_dict[p][interest_class[ip]])for ip,p in enumerate(class_properties)]
        
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation[standard_smiles_column],compute_SD=True,convert_log10=False)


        _=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        

        p=stacked_model.generate_multi_property_name(class_properties)
        sample_train=None
        sample_val=None
        if use_sample_weight:
            sample_train=stacked_model.Train[f'sw_{p}'].values
            sample_val=stacked_model.Validation[f'sw_{p}'].values
        stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)
        
        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)

    
    def test_a_clf_ch_cntprop_clsv_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('------------------test_clf_ch_cntprop_clsv_nosw_vkmac_cvkmgkf4--------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
        
        categorical=False
        properties=dv_properties
        nb_classes=[3,2,2,2]
        class_values= [[2,10],[6.7],[6.5],[5]]
        class_quantiles=None    
        min_allowed_class_samples=25
        use_sample_weight=False
        encoder='CHEMBL'
        task='Classification'
        computional_load='cheap'
        scorer='balanced_accuracy'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        if computional_load=='expensive':
            use_sample_weight=False
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        interest_class=[0 for p in properties]
        save_model_to_file='clf_model.pt'        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=False
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
            
        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=class_values,
                                 categorical=categorical,use_quantiles=class_quantiles is not None,
                                 prefix='Class',min_allowed_class_samples=30,verbose=verbose)

        #check properties
        prop_builder.check_properties(df)        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,train_properties=prop_builder.generate_train_properties(df)
        #retrieve labelnames 
        labelnames=prop_builder.labelnames
        
        class_properties=train_properties
                
        for ip,p in enumerate(class_properties):
            self.assertEqual(np.sum(df[p].value_counts()),len(df)-np.sum(df[properties[ip]].isna()) ,True)
        
        weighted_samples_index={p:0 for i,p in enumerate(train_properties)}
        if categorical:
            weighted_samples_index={p:'0.0' for p in train_properties}
        select_sample_weights={p:10 for p in train_properties}  

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)    
        
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=False,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list, verbose=verbose,
                                     n_jobs=n_jobs,
                                     labelnames=labelnames,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES
  
        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,class_properties,strategy,categorical,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
            
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print(f'Warning: on average less than 10 samples per cluster for property {p}, suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality,random_state=random_state)
        
        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in class_properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles='stereo_SMILES',
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True,
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()

        cmap=['PiYG','Blues']
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES,compute_SD=True,convert_log10=False)

        youden_dict=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        F1_dict=ResultDesigner().show_clf_threshold_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,youden_dict=youden_dict)
        
        [stacked_model.set_property_threshold(prop=p,class_index=interest_class[ip],threshold=F1_dict[p][interest_class[ip]])for ip,p in enumerate(class_properties)]
        
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation[standard_smiles_column],compute_SD=True,convert_log10=False)


        _=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        

        for p in class_properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)
        
        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
        
    def test_a_regclf_ch_cntprop_clsv_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('----------------test_regclf_ch_cntprop_clsv_nosw_vkmac_cvkmgkf4-------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
        
        categorical=False
        properties=dv_properties
        nb_classes=[2,2,2,2]
        class_values= [[3],[6.7],[6.5],[5]]
        class_quantiles=None    
        min_allowed_class_samples=25
        use_sample_weight=False
        encoder='CHEMBL'
        task='RegressionClassification'
        computional_load='cheap'
        scorer='balanced_accuracy'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        if computional_load=='expensive':
            use_sample_weight=False
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        interest_class=[1 for p in properties]
        save_model_to_file='regclf_model.pt'        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=False
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
            
        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=class_values,
                                 categorical=categorical,use_quantiles=class_quantiles is not None,
                                 prefix='Class',min_allowed_class_samples=30,verbose=verbose)

        #check properties
        prop_builder.check_properties(df)        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,train_properties=prop_builder.generate_train_properties(df)
        #retrieve labelnames 
        labelnames=prop_builder.labelnames
        
        class_properties=train_properties
                
        for ip,p in enumerate(class_properties):
            self.assertEqual(np.sum(df[p].value_counts()),len(df)-np.sum(df[properties[ip]].isna()) ,True)
        
        weighted_samples_index={p:0 for i,p in enumerate(train_properties)}
        if categorical:
            weighted_samples_index={p:'0.0' for p in train_properties}
        select_sample_weights={p:10 for p in train_properties}  

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)    
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=False,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                                     n_jobs=n_jobs,
                                     labelnames=labelnames,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES
  
        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,class_properties,strategy,categorical,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
            
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print(f'Warning: on average less than 10 samples per cluster for property {p}, suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality,random_state=random_state)
        
        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in class_properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles='stereo_SMILES',
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True,
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()

        cmap=['PiYG','Blues']
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES,compute_SD=True,convert_log10=False)
        youden_dict=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        F1_dict=ResultDesigner().show_clf_threshold_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,youden_dict=youden_dict)
        
        [stacked_model.set_property_threshold(prop=p,threshold=F1_dict[p][interest_class[ip]])for ip,p in enumerate(class_properties)]
        
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation[standard_smiles_column],compute_SD=True,convert_log10=False)


        _=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        

        for p in class_properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)
        
        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
        
    def test_c_clf_mo_cntprop_clsv_sw_vbstrat_cvbskf5(self):
        print('----------------------------------------------------------------------------------')
        print('------------------test_clf_mo_cntprop_clsv_sw_vbstrat_cvbskf5---------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
                
        categorical=False
        properties=dv_properties[:2]
        nb_classes=[3,2]
        class_values= [[2,10],[6.7]]
        class_quantiles=None    
        min_allowed_class_samples=25
        use_sample_weight=True
        encoder='CHEMBL'
        task='Classification'
        computional_load='moderate'
        scorer='recall'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        if computional_load=='expensive':
            use_sample_weight=False
        val_clustering='Butina'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='stratified'
        test_size=0.25
        cv_clustering='Butina'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='SKF'
        outer_folds=3
        interest_class=[0 for p in properties]
        save_model_to_file='test_model.pt'   
        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=True
        check_rdkit_desc=False
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=class_values,
                                 categorical=categorical,use_quantiles=class_quantiles is not None,
                                 prefix='Class',min_allowed_class_samples=30,verbose=verbose)

        #check properties
        prop_builder.check_properties(df)        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,train_properties=prop_builder.generate_train_properties(df)
        #retrieve labelnames 
        labelnames=prop_builder.labelnames
        
        class_properties=train_properties
        for ip,p in enumerate(class_properties):
            self.assertEqual(np.sum(df[p].value_counts()),len(df)-np.sum(df[properties[ip]].isna()) ,True)
            
        weighted_samples_index={p:0 for i,p in enumerate(train_properties)}
        if categorical:
            weighted_samples_index={p:'0.0' for p in train_properties}
        select_sample_weights={p:10 for p in train_properties}    
        

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)    
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=False,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                                     n_jobs=n_jobs,
                                     labelnames=labelnames,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES
        
        clustering_algorithm=KmeansForSmiles(n_groups=30,feature_generators=retrieve_default_offline_generators(model=encoder, radius=2,nbits=2048),used_features='rdkit',random_state=42)

        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df, properties, class_properties, strategy, categorical, stacked_model, standard_smiles_column, df_smiles, test_size, val_clustering, val_km_groups, val_butina_cutoff, val_include_chirality, verbose, random_state, clustering_algorithm=clustering_algorithm)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
            
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality, random_state=random_state)
        
        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in class_properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles='stereo_SMILES',
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True,
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()

        cmap=['PiYG','Blues']
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES,compute_SD=True,convert_log10=False)

        youden_dict=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        F1_dict=ResultDesigner().show_clf_threshold_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,youden_dict=youden_dict)
        
        [stacked_model.set_property_threshold(prop=p,class_index=interest_class[ip],threshold=F1_dict[p][interest_class[ip]])for ip,p in enumerate(class_properties)]
        
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation[standard_smiles_column],compute_SD=True,convert_log10=False)


        _=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        

        for p in class_properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)
        
        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
        
    def test_d_clf_ex_cntprop_clsq_nosw_vhbood_cvhblgo(self):
        print('----------------------------------------------------------------------------------')
        print('------------------test_clf_ex_cntprop_clsv_sw_vbstrat_cvbskf3---------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
                
        categorical=False
        properties=dv_properties[:2]
        nb_classes=[3,2]
        class_values=None
        class_quantiles=[[0.2,0.4],[0.3]]    
        min_allowed_class_samples=25
        use_sample_weight=False
        encoder='CHEMBL'
        task='Classification'
        computional_load='expensive'
        scorer='f1_weighted'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        if computional_load=='expensive':
            use_sample_weight=False
        val_clustering='HierarchicalButina'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='out of domain'
        test_size=0.25
        cv_clustering='HierarchicalButina'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='SKF'
        outer_folds=3
        interest_class=[0 for p in properties]
        save_model_to_file='test_model.pt'    
        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=False
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
            
        if class_quantiles is not None:
            class_values=class_quantiles

        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=class_values,
                                 categorical=categorical,use_quantiles=class_quantiles is not None,
                                 prefix='Class',min_allowed_class_samples=30,verbose=verbose)

        #check properties
        prop_builder.check_properties(df)        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,train_properties=prop_builder.generate_train_properties(df)
        #retrieve labelnames 
        labelnames=prop_builder.labelnames
        
        class_properties=train_properties        
        weighted_samples_index={p:0 for i,p in enumerate(train_properties)}
        if categorical:
            weighted_samples_index={p:'0.0' for p in train_properties}
        select_sample_weights={p:10 for p in train_properties}  

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)    
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=False,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                                     n_jobs=n_jobs,
                                     labelnames=labelnames,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES
        clustering_algorithm=KmeansForSmiles(n_groups=30,feature_generators=retrieve_default_offline_generators(model=encoder, radius=2,nbits=2048),used_features='Bottleneck',random_state=42)
        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,class_properties,strategy,categorical,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state,clustering_algorithm=clustering_algorithm)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
            
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)
        
        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in class_properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles='stereo_SMILES',
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True, 
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()

        cmap=['PiYG','Blues']
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES,compute_SD=True,convert_log10=False)

        youden_dict=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        F1_dict=ResultDesigner().show_clf_threshold_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,youden_dict=youden_dict)
        
        [stacked_model.set_property_threshold(prop=p,class_index=interest_class[ip],threshold=F1_dict[p][interest_class[ip]])for ip,p in enumerate(class_properties)]
        
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation[standard_smiles_column],compute_SD=True,convert_log10=False)
            

        _=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        

        for p in class_properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)
        
        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
        
    def test_a_clf_ch_catprop_clsv_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('------------------test_clf_ch_catprop_clsv_nosw_vkmac_cvkmgkf4--------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
                
        categorical=True
        properties=cat_properties
        nb_classes=[2]
        class_values= None
        class_quantiles=None    
        min_allowed_class_samples=25
        use_sample_weight=False
        encoder='CHEMBL'
        task='Classification'
        computional_load='cheap'
        scorer='balanced_accuracy'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        if computional_load=='expensive':
            use_sample_weight=False
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        interest_class=[0 for p in properties]
        save_model_to_file='cat_model.pt'

        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=False
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=class_values,
                                 categorical=categorical,use_quantiles=class_quantiles is not None,
                                 prefix='Class',min_allowed_class_samples=30,verbose=verbose)

        #check properties
        prop_builder.check_properties(df)        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,train_properties=prop_builder.generate_train_properties(df)
        #retrieve labelnames 
        labelnames=prop_builder.labelnames
        
        class_properties=train_properties
        
        for ip,p in enumerate(class_properties):
            self.assertEqual(np.sum(df[p].value_counts()),len(df)-np.sum(df[properties[ip]].isna()) ,True)
        weighted_samples_index={p:0 for i,p in enumerate(train_properties)}
        if categorical:
            weighted_samples_index={p:'0.0' for p in train_properties}
        select_sample_weights={p:10 for p in train_properties}  

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)    
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=False,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                                     n_jobs=n_jobs,
                                     labelnames=labelnames,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES
  
        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,class_properties,strategy,categorical,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
            
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)
        
        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in class_properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles='stereo_SMILES',
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True, 
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()

        cmap=['PiYG','Blues']
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES,compute_SD=True,convert_log10=False)

        youden_dict=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        F1_dict=ResultDesigner().show_clf_threshold_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,youden_dict=youden_dict)
        
        [stacked_model.set_property_threshold(prop=p,class_index=interest_class[ip],threshold=F1_dict[p][interest_class[ip]])for ip,p in enumerate(class_properties)]
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation[standard_smiles_column],compute_SD=True,convert_log10=False)

        _=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        
        for p in class_properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)
        
        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
        
        
    def test_d_clf_sstack_cntprop_clsv_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('---------------test_clf_sstack_cntprop_clsv_nosw_vkmac_cvkmgkf4-----------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
                
        categorical=False
        properties=dv_properties
        nb_classes=[3,2,2,2]
        class_values= [[2,10],[6.7],[6.5],[5]]
        class_quantiles=None    
        min_allowed_class_samples=25
        use_sample_weight=False
        encoder='CHEMBL'
        task='Classification'
        computional_load='cheap'
        scorer='balanced_accuracy'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        if computional_load=='expensive':
            use_sample_weight=False
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        interest_class=[0 for p in properties]
        save_model_to_file='clf_model.pt' 
        
        include_rdkitfeatures=False
        include_fps=False
        include_Bottleneck_features=True
        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=include_rdkitfeatures
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]

        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=class_values,
                                 categorical=categorical,use_quantiles=class_quantiles is not None,
                                 prefix='Class',min_allowed_class_samples=30,verbose=verbose)

        #check properties
        prop_builder.check_properties(df)        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,train_properties=prop_builder.generate_train_properties(df)
        #retrieve labelnames 
        labelnames=prop_builder.labelnames
        
        class_properties=train_properties   
        
        weighted_samples_index={p:0 for i,p in enumerate(train_properties)}
        if categorical:
            weighted_samples_index={p:'0.0' for p in train_properties}
        select_sample_weights={p:10 for p in train_properties}  
        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)    
            
        #inner_methods, inner_stacking, single_stack, top_method, top_stacking
        model_config='single_stack'

        #set to true if gridsearch has to be applied (warning: normally when using inner_stacking or single_stack, gridsearch fails due to the ridiculous amount of parameters)
        force_gridsearch=False
        #set the value of randomized iterations
        randomized_iterations=30
        #use distributions for parameter selection, 
        #if True a new model is added for method family in each inner fold (the probability that all method parameters are equal in two searches is close to zero)
        #also usefull if the number of parameters is too large when using single_stack
        distribution_defaults=False 
        #use HyperOpt optimization to find the best parameters
        hyperopt_defaults=False

        #number of xgb/lgbm threads
        xgb_threads=None
        #number of random forest threads
        rfr_threads=None
        #n_jobs for randomizedsearch/gridsearch
        n_jobs=12
        # threads for hyoperopt, set to higher value than 1 to use pyspark, faster for heavy methods, but slower fast methods
        hyperopt_threads=None

        #naming prefixes for different steps in the pipeline(s)
        prefixes={'method_prefix':'clf',
                  'dim_prefix':'reduce_dim',
                  'estimator_prefix':'est_pipe'}

        #create own method archives
        method_archive=ClassifierArchive(method_prefix=prefixes['method_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                                         random_state=random_state,xgb_threads=xgb_threads,rfr_threads=rfr_threads)
        dim_archive=ReducedimArchive(method_prefix=prefixes['dim_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                                     random_state=random_state)

        #adding duplicate methods with different random state
        nb_copies=8
        method='lgbm'
        #in the case of hyperopt, it is limited to lgbm and xgb, due to uniqueness requirement in hyperopt parameters. For now workaround is provided for lgbm and xgboost. 
        if hyperopt_defaults:
            if method=='lgbm':
                method_archive,clf_list=add_lgbm_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,xgb_threads=xgb_threads, distribution_defaults=distribution_defaults,random_state=random_state)
            elif method=='xgb':
                method_archive,clf_list=add_xgb_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,xgb_threads=xgb_threads,distribution_defaults=distribution_defaults,random_state=random_state)
        else:
            clf_list=method_archive.duplicate_method_xtimes(method_name=method,x=nb_copies,random_state=random_state)

        #use method archive as blender_archive
        blender_archive=method_archive

        red_dim_list=['passthrough']
        clf_list=['lr' ,'SVC']#+clf_list[:2]#, 'lda','rfc']#,'qda','sgdc','dtc','xgb']
        blender_list=['lr','SVC']+[clf_list[0]]#,'rfc']
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=distribution_defaults,
                                    hyperopt_defaults=hyperopt_defaults,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                                          red_dim_list=red_dim_list,
                             method_list=clf_list,#list of classifier keys from method_archive
                             blender_list=blender_list,#list of classifier keys from blender_archive
                             model_config=model_config, #string with the model layout/config
                             force_gridsearch=force_gridsearch, #set to True when you want to force gridsearch
                             randomized_iterations=randomized_iterations, #number of randomized iterations
                             method_archive=method_archive, #archive class holding the methods
                             dim_archive=dim_archive, 
                             blender_archive=blender_archive, #archive class holding the blender methods
                             prefixes=prefixes #naming convention in parameters for grid/randomized search
                            ,hyperopt_threads=hyperopt_threads,
                                     n_jobs=n_jobs,
                                     labelnames=labelnames,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES
        chem_clustering_algorithm=ButinaSplitReassigned(cutoff = 0.65,feature_generator=ECFPGenerator(radius=2, nBits =2048))
        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties, class_properties,strategy,categorical, stacked_model,standard_smiles_column,df_smiles, test_size,val_clustering, val_km_groups,val_butina_cutoff, val_include_chirality,verbose, random_state, chem_clustering_algorithm=chem_clustering_algorithm)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
            
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)
        
        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in class_properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles='stereo_SMILES',
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                       include_Bottleneck_features=include_Bottleneck_features,include_fps=include_fps,include_rdkitfeatures=include_rdkitfeatures,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True, 
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()

        cmap=['PiYG','Blues']
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES,compute_SD=True,convert_log10=False)

        youden_dict=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        F1_dict=ResultDesigner().show_clf_threshold_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,youden_dict=youden_dict)
        
        [stacked_model.set_property_threshold(prop=p,class_index=interest_class[ip],threshold=F1_dict[p][interest_class[ip]])for ip,p in enumerate(class_properties)]
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation[standard_smiles_column],compute_SD=True,convert_log10=False)

        _=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        

        for p in class_properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)
        
        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
    
    def test_d_clf_innm_cntprop_clsv_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('-----------------test_clf_innm_cntprop_clsv_nosw_vkmac_cvkmgkf4------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
                
        categorical=False
        properties=dv_properties
        nb_classes=[3,2,2,2]
        class_values= [[2,10],[6.7],[6.5],[5]]
        class_quantiles=None    
        min_allowed_class_samples=25
        use_sample_weight=False
        encoder='CHEMBL'
        task='Classification'
        computional_load='cheap'
        scorer='balanced_accuracy'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        if computional_load=='expensive':
            use_sample_weight=False
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        interest_class=[0 for p in properties]
        save_model_to_file='clf_model.pt' 
        
        include_rdkitfeatures=False
        include_fps=False
        include_Bottleneck_features=True
        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=include_rdkitfeatures
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]

        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=class_values,
                                 categorical=categorical,use_quantiles=class_quantiles is not None,
                                 prefix='Class',min_allowed_class_samples=30,verbose=verbose)

        #check properties
        prop_builder.check_properties(df)        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,train_properties=prop_builder.generate_train_properties(df)
        #retrieve labelnames 
        labelnames=prop_builder.labelnames
        
        class_properties=train_properties   
        
        weighted_samples_index={p:0 for i,p in enumerate(train_properties)}
        if categorical:
            weighted_samples_index={p:'0.0' for p in train_properties}
        select_sample_weights={p:10 for p in train_properties}  
        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)    
            
        #inner_methods, inner_stacking, single_stack, top_method, top_stacking
        model_config='inner_methods'

        #set to true if gridsearch has to be applied (warning: normally when using inner_stacking or single_stack, gridsearch fails due to the ridiculous amount of parameters)
        force_gridsearch=False
        #set the value of randomized iterations
        randomized_iterations=30
        #use distributions for parameter selection, 
        #if True a new model is added for method family in each inner fold (the probability that all method parameters are equal in two searches is close to zero)
        #also usefull if the number of parameters is too large when using single_stack
        distribution_defaults=True 
        #use HyperOpt optimization to find the best parameters
        hyperopt_defaults=False

        #number of xgb/lgbm threads
        xgb_threads=2
        #number of random forest threads
        rfr_threads=None
        #n_jobs for randomizedsearch/gridsearch
        n_jobs=20
        # threads for hyoperopt, set to higher value than 1 to use pyspark, faster for heavy methods, but slower fast methods
        hyperopt_threads=None

        #naming prefixes for different steps in the pipeline(s)
        prefixes={'method_prefix':'clf',
                  'dim_prefix':'reduce_dim',
                  'estimator_prefix':'est_pipe'}

        #create own method archives
        method_archive=ClassifierArchive(method_prefix=prefixes['method_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                                         random_state=random_state,xgb_threads=xgb_threads,rfr_threads=rfr_threads)
        dim_archive=ReducedimArchive(method_prefix=prefixes['dim_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                                     random_state=random_state)

        #adding duplicate methods with different random state
        nb_copies=8
        method='rfc'
        #in the case of hyperopt, it is limited to lgbm and xgb, due to uniqueness requirement in hyperopt parameters. For now workaround is provided for lgbm and xgboost. 
        if hyperopt_defaults:
            if method=='lgbm':
                method_archive,clf_list=add_lgbm_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,xgb_threads=xgb_threads, distribution_defaults=distribution_defaults,random_state=random_state)
            elif method=='xgb':
                method_archive,clf_list=add_xgb_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,xgb_threads=xgb_threads,distribution_defaults=distribution_defaults,random_state=random_state)
        else:
            clf_list=method_archive.duplicate_method_xtimes(method_name=method,x=nb_copies,random_state=random_state)

        #use method archive as blender_archive
        blender_archive=method_archive

        red_dim_list=['passthrough']
        clf_list=clf_list[:2]#, 'lda','rfc']#,'qda','sgdc','dtc','xgb']
        blender_list=['lr','SVC']+[clf_list[0]]#,'rfc']
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=distribution_defaults,
                                    hyperopt_defaults=hyperopt_defaults,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list, 
                                          red_dim_list=red_dim_list,
                             method_list=clf_list,#list of classifier keys from method_archive
                             blender_list=blender_list,#list of classifier keys from blender_archive
                             model_config=model_config, #string with the model layout/config
                             force_gridsearch=force_gridsearch, #set to True when you want to force gridsearch
                             randomized_iterations=randomized_iterations, #number of randomized iterations
                             method_archive=method_archive, #archive class holding the methods
                             dim_archive=dim_archive, 
                             blender_archive=blender_archive, #archive class holding the blender methods
                             prefixes=prefixes #naming convention in parameters for grid/randomized search
                            ,hyperopt_threads=hyperopt_threads,
                                     n_jobs=n_jobs,
                                     labelnames=labelnames,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES
  
        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,class_properties,strategy,categorical,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
            
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality , random_state=random_state)
        
        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in class_properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles='stereo_SMILES',
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                       include_Bottleneck_features=include_Bottleneck_features,include_fps=include_fps,include_rdkitfeatures=include_rdkitfeatures,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True, 
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()

        cmap=['PiYG','Blues']
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES,compute_SD=True,convert_log10=False)

        youden_dict=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        F1_dict=ResultDesigner().show_clf_threshold_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,youden_dict=youden_dict)
        
        [stacked_model.set_property_threshold(prop=p,class_index=interest_class[ip],threshold=F1_dict[p][interest_class[ip]])for ip,p in enumerate(class_properties)]
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation[standard_smiles_column],compute_SD=True,convert_log10=False)

        _=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        

        for p in class_properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)
        
        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
        
    def test_d_clf_innstack_cntprop_clsv_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('---------------test_clf_innstack_cntprop_clsv_nosw_vkmac_cvkmgkf4-----------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
                
        categorical=False
        properties=dv_properties[1:2]
        nb_classes=[2,2,]
        class_values= [[6.7],[6.5]]
        class_quantiles=None    
        min_allowed_class_samples=25
        use_sample_weight=False
        encoder='CHEMBL'
        task='Classification'
        computional_load='cheap'
        scorer='balanced_accuracy'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        if computional_load=='expensive':
            use_sample_weight=False
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        interest_class=[0 for p in properties]
        save_model_to_file='clf_model.pt' 
        
        include_rdkitfeatures=False
        include_fps=False
        include_Bottleneck_features=True
        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=include_rdkitfeatures
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]

        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=class_values,
                                 categorical=categorical,use_quantiles=class_quantiles is not None,
                                 prefix='Class',min_allowed_class_samples=30,verbose=verbose)

        #check properties
        prop_builder.check_properties(df)        
        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,train_properties=prop_builder.generate_train_properties(df)
        #retrieve labelnames 
        labelnames=prop_builder.labelnames
        
        class_properties=train_properties   
        
        weighted_samples_index={p:0 for i,p in enumerate(train_properties)}
        if categorical:
            weighted_samples_index={p:'0.0' for p in train_properties}
        select_sample_weights={p:10 for p in train_properties}  
        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)    
            
        #inner_methods, inner_stacking, single_stack, top_method, top_stacking
        model_config='inner_stacking'

        #set to true if gridsearch has to be applied (warning: normally when using inner_stacking or single_stack, gridsearch fails due to the ridiculous amount of parameters)
        force_gridsearch=False
        #set the value of randomized iterations
        randomized_iterations=30
        #use distributions for parameter selection, 
        #if True a new model is added for method family in each inner fold (the probability that all method parameters are equal in two searches is close to zero)
        #also usefull if the number of parameters is too large when using single_stack
        distribution_defaults=True 
        #use HyperOpt optimization to find the best parameters
        hyperopt_defaults=True

        #number of xgb/lgbm threads
        xgb_threads=2
        #number of random forest threads
        rfr_threads=None
        #n_jobs for randomizedsearch/gridsearch
        n_jobs=12
        # threads for hyoperopt, set to higher value than 1 to use pyspark, faster for heavy methods, but slower fast methods
        hyperopt_threads=None

        #naming prefixes for different steps in the pipeline(s)
        prefixes={'method_prefix':'clf',
                  'dim_prefix':'reduce_dim',
                  'estimator_prefix':'est_pipe'}

        #create own method archives
        method_archive=ClassifierArchive(method_prefix=prefixes['method_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                                         random_state=random_state,xgb_threads=xgb_threads,rfr_threads=rfr_threads)
        dim_archive=ReducedimArchive(method_prefix=prefixes['dim_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                                     random_state=random_state)

        #adding duplicate methods with different random state
        nb_copies=8
        method='lgbm'
        #in the case of hyperopt, it is limited to lgbm and xgb, due to uniqueness requirement in hyperopt parameters. For now workaround is provided for lgbm and xgboost. 
        if hyperopt_defaults:
            if method=='lgbm':
                method_archive,clf_list=add_lgbm_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,xgb_threads=xgb_threads, distribution_defaults=distribution_defaults,random_state=random_state)
            elif method=='xgb':
                method_archive,clf_list=add_xgb_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,xgb_threads=xgb_threads,distribution_defaults=distribution_defaults,random_state=random_state)
        else:
            clf_list=method_archive.duplicate_method_xtimes(method_name=method,x=nb_copies,random_state=random_state)

        #use method archive as blender_archive
        blender_archive=method_archive

        red_dim_list=['passthrough']
        clf_list=clf_list[:2]#['lr' ,'SVC']#+clf_list[:2]#, 'lda','rfc']#,'qda','sgdc','dtc','xgb']
        blender_list=['lr','SVC']#,'rfc']
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=distribution_defaults,
                                    hyperopt_defaults=hyperopt_defaults,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                                          red_dim_list=red_dim_list,
                             method_list=clf_list,#list of classifier keys from method_archive
                             blender_list=blender_list,#list of classifier keys from blender_archive
                             model_config=model_config, #string with the model layout/config
                             force_gridsearch=force_gridsearch, #set to True when you want to force gridsearch
                             randomized_iterations=randomized_iterations, #number of randomized iterations
                             method_archive=method_archive, #archive class holding the methods
                             dim_archive=dim_archive, 
                             blender_archive=blender_archive, #archive class holding the blender methods
                             prefixes=prefixes #naming convention in parameters for grid/randomized search
                            ,hyperopt_threads=hyperopt_threads,
                                     n_jobs=n_jobs,
                                     labelnames=labelnames,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES
  
        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,class_properties,strategy,categorical,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
            
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)
        
        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in class_properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles='stereo_SMILES',
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                       include_Bottleneck_features=include_Bottleneck_features,include_fps=include_fps,include_rdkitfeatures=include_rdkitfeatures,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True, 
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()

        cmap=['PiYG','Blues']
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES,compute_SD=True,convert_log10=False)

        youden_dict=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        F1_dict=ResultDesigner().show_clf_threshold_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,youden_dict=youden_dict)
        
        [stacked_model.set_property_threshold(prop=p,class_index=interest_class[ip],threshold=F1_dict[p][interest_class[ip]])for ip,p in enumerate(class_properties)]
        out=stacked_model.predict( props =None, smiles=stacked_model.Validation[standard_smiles_column],compute_SD=True,convert_log10=False)

        _=ResultDesigner().show_classification_report(class_properties,out,[stacked_model.Validation[f'{p}'].values for p in class_properties],labelnames=labelnames,cmap=cmap)
        

        for p in class_properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)
        
        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)        
        
######################################################################################
#                       Regression
######################################################################################        
    def test_a_reg_orig_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('------------------------test_reg_orig_nosw_vkmac_cvkmgkf4-------------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()

        properties=dv_properties
        use_log10=False
        percentages=False
        use_logit=False
        remove_outliers=False
        confidence=3.8  #3.8
        use_sample_weight=False
        encoder='CHEMBL'
        task='Regression'
        computional_load='cheap'
        scorer='r2'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        revert_to_original=False
        save_model_to_file='reg_model.pt'

        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=False
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)

        transformed_data=use_log10 or use_logit
        
        prop_builder=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=standard_smiles_column)

        #check properties
        prop_builder.check_properties(df)        

        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=prop_builder.generate_train_properties(df)
        
        if transformed_data:
            original_props=prop_builder.original_props
            properties=log_props
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True,subset =properties,how='all' )
        df.reset_index(drop=True,inplace=True)

        weighted_samples_index={p:'<1' for p in properties}
        select_sample_weights={p:10 for p in properties}
        
        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)

        
        if computional_load=='expensive':
            use_sample_weight=False
            
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=False,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                                     n_jobs=n_jobs,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()


        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES


        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,properties,strategy,False,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)

        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True,
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()
        smiles_list=stacked_model.Validation.stereo_SMILES
        out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)
        
        if transformed_data and revert_to_original:
            original_prop_clif_dict=prop_clif_dict
            prop_clif_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_clif_dict.items()}
            properties=original_props

        ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],prop_cliffs=prop_clif_dict, leave_grp_out=leave_grp_out)

        if prop_clif_dict is not None:
            prop_cliffs_trimmed={}
            for ip,p in enumerate(properties): 
                if transformed_data:
                    predict_p=log_props[ip]
                else:
                    predict_p=p
                
                out=stacked_model.predict( props =predict_p, smiles=stacked_model.Validation.stereo_SMILES.values[prop_clif_dict[p]],compute_SD=True,convert_log10=transformed_data and revert_to_original)
                
                prop_cliffs_trimmed[p]=np.arange(len(prop_clif_dict[p]))
                if len(stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]])>0:
                    ResultDesigner().show_regression_report(p,out,y_true=[stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]]],prop_cliffs=prop_cliffs_trimmed, leave_grp_out=None,bins=10)
        
        if leave_grp_out is not None:
            out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES.values[leave_grp_out],compute_SD=True,convert_log10=transformed_data and revert_to_original)


            if transformed_data and revert_to_original:
                properties=original_props

            leave_grp_out_trim=np.arange(len(leave_grp_out))
            ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values[leave_grp_out] for p in properties],prop_cliffs=None, leave_grp_out=leave_grp_out_trim,bins=10)
        
        sample_train=None
        sample_val=None

        if transformed_data and revert_to_original:
            properties=log_props

        for p in properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        plt.close('all')
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)

        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)

    def test_b_reg_multiprop_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('---------------------test_reg_multiprop_nosw_vkmac_cvkmgkf4-----------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()

        properties=dv_properties
        use_log10=False
        percentages=False
        use_logit=False
        remove_outliers=False
        confidence=3.8  #3.8
        use_sample_weight=False
        encoder='CHEMBL'
        task='Regression'
        computional_load='cheap'
        scorer='r2'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        revert_to_original=False
        save_model_to_file='regmulti_model.pt'

        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=False
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)

        transformed_data=use_log10 or use_logit
        
        prop_builder=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=standard_smiles_column)

        #check properties
        prop_builder.check_properties(df)        

        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=prop_builder.generate_train_properties(df)
        
        if transformed_data:
            original_props=prop_builder.original_props
            properties=log_props
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True,subset =properties,how='all' )
        df.reset_index(drop=True,inplace=True)

        weighted_samples_index={p:'<1' for p in properties}
        select_sample_weights={p:10 for p in properties}
        
        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)

        
        if computional_load=='expensive':
            use_sample_weight=False

        blender_list=['lasso','kernelridge']#,'kernelridge','dtr']
        #define base estimators/or method used in single method
        reg_list=['lasso']#
        
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=False,
                                     use_gpu=False,
                                     blender_list=blender_list,
                                     method_list=reg_list,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                                     n_jobs=n_jobs,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()


        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES


        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,properties,strategy,False,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)

        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        p=properties
        
        sample_weight=None
        if use_sample_weight:
            sample_weight=stacked_model.Train[f'sw_{p}'].values

        stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                    params_grid=params_grid,
                                   paramsearch=paramsearch,
                                  scoring=scorer,#'neg_mean_absolute_error',#
                                  cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                  use_memory=True,
                                  plot_validation=True, 
                                 refit=False,# no refit with validation. comes later,
                                 blender_params=blender_params
                                  ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()
        smiles_list=stacked_model.Validation.stereo_SMILES
        out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)
        
        if transformed_data and revert_to_original:
            original_prop_clif_dict=prop_clif_dict
            prop_clif_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_clif_dict.items()}
            properties=original_props

        ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],prop_cliffs=prop_clif_dict, leave_grp_out=leave_grp_out)

        

        if transformed_data and revert_to_original:
            properties=log_props

        p=stacked_model.generate_multi_property_name(properties)
        sample_train=None
        sample_val=None
        if use_sample_weight:
            sample_train=stacked_model.Train[f'sw_{p}'].values
            sample_val=stacked_model.Validation[f'sw_{p}'].values
        stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        plt.close('all')
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)

        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)

    
        
    def test_d_reg_ex_log10_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('--------------------test_d_reg_ex_log10_nosw_vkmac_cvkmgkf4----------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()


        properties=dv_properties
        use_log10=True
        percentages=False
        use_logit=False
        remove_outliers=False
        confidence=3.8  #3.8
        use_sample_weight=False
        encoder='CHEMBL'
        task='Regression'
        computional_load='expensive'
        scorer='r2'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        revert_to_original=True
        save_model_to_file='test_model.pt'

        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=False
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)

        transformed_data=use_log10 or use_logit
        prop_builder=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=standard_smiles_column)

        #check properties
        prop_builder.check_properties(df)        

        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=prop_builder.generate_train_properties(df)
        
        if transformed_data:
            original_props=prop_builder.original_props
            properties=log_props
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True,subset =properties,how='all' )
        df.reset_index(drop=True,inplace=True)
        
        weighted_samples_index={p:'<1' for p in properties}
        select_sample_weights={p:10 for p in properties}
        
        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)
        
        if computional_load=='expensive':
            use_sample_weight=False
        
        param_Factory=ModelAndParams(model=encoder,
                                         task=task,
                                         computional_load=computional_load,
                                         distribution_defaults=False,
                                         use_gpu=False,
                                         normalizer=True,
                                         top_normalizer=True,
                                         random_state=random_state_list,
                                         n_jobs=n_jobs,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()


        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES


        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,properties,strategy,False,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)

        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True, 
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()
        smiles_list=stacked_model.Validation.stereo_SMILES
        out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)
        
        if transformed_data and revert_to_original:
            original_prop_clif_dict=prop_clif_dict
            prop_clif_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_clif_dict.items()}
            properties=original_props

        ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],prop_cliffs=prop_clif_dict, leave_grp_out=leave_grp_out)

        if prop_clif_dict is not None:
            prop_cliffs_trimmed={}
            for ip,p in enumerate(properties): 
                if transformed_data:
                    predict_p=log_props[ip]
                else:
                    predict_p=p
                
                out=stacked_model.predict( props =predict_p, smiles=stacked_model.Validation.stereo_SMILES.values[prop_clif_dict[p]],compute_SD=True,convert_log10=transformed_data and revert_to_original)
                
                prop_cliffs_trimmed[p]=np.arange(len(prop_clif_dict[p]))
                if len(stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]])>0:
                    ResultDesigner().show_regression_report(p,out,y_true=[stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]]],prop_cliffs=prop_cliffs_trimmed, leave_grp_out=None,bins=10)
        
        if leave_grp_out is not None:
            out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES.values[leave_grp_out],compute_SD=True,convert_log10=transformed_data and revert_to_original)


            if transformed_data and revert_to_original:
                properties=original_props

            leave_grp_out_trim=np.arange(len(leave_grp_out))
            ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values[leave_grp_out] for p in properties],prop_cliffs=None, leave_grp_out=leave_grp_out_trim,bins=10)
        
        sample_train=None
        sample_val=None

        if transformed_data and revert_to_original:
            properties=log_props

        for p in properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        plt.close('all')
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)

        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
        
    def test_c_reg_mo_logit_vscac_cvscgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('------------------------test_c_reg_mo_logit_vscac_cvscgkf4------------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()

        properties=[dv_properties[0]]
        use_log10=False
        percentages=True
        use_logit=True
        remove_outliers=True
        confidence=3.8  #3.8
        use_sample_weight=False
        encoder='CHEMBL'
        task='Regression'
        computional_load='moderate'
        scorer='r2'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        val_clustering='Scaffold'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Scaffold'
        include_chirality=True
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        revert_to_original=True
        save_model_to_file='test_model.pt'

        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=False
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)

        transformed_data=use_log10 or use_logit
        prop_builder=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=standard_smiles_column)

        #check properties
        prop_builder.check_properties(df)        

        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=prop_builder.generate_train_properties(df)
        
        if transformed_data:
            original_props=prop_builder.original_props
            properties=log_props
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True,subset =properties,how='all' )
        df.reset_index(drop=True,inplace=True)
        weighted_samples_index={p:'<1' for p in properties}
        select_sample_weights={p:10 for p in properties}

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)
        
        if computional_load=='expensive':
            use_sample_weight=False
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=False,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                                     n_jobs=n_jobs,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()


        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES


        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,properties,strategy,False,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)

        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1

        for p in properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True,
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()
        smiles_list=stacked_model.Validation.stereo_SMILES
        out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)

        if transformed_data and revert_to_original:
            original_prop_clif_dict=prop_clif_dict
            prop_clif_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_clif_dict.items()}
            properties=original_props
        #for p in properties:
        #    print(out[f'predicted_{p}'])
        #    print(stacked_model.Validation[f'{p}'].values)
            
        ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],prop_cliffs=prop_clif_dict, leave_grp_out=leave_grp_out)

        if prop_clif_dict is not None:
            prop_cliffs_trimmed={}
            for ip,p in enumerate(properties):
                if len(prop_clif_dict[p])==0:continue
                
                if transformed_data:
                    predict_p=log_props[ip]
                else:
                    predict_p=p
                    
                out=stacked_model.predict( props =predict_p, smiles=stacked_model.Validation.stereo_SMILES.values[prop_clif_dict[p]],compute_SD=True,convert_log10=transformed_data and revert_to_original)
                
                prop_cliffs_trimmed[p]=np.arange(len(prop_clif_dict[p]))
                if len(stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]])>0:
                    ResultDesigner().show_regression_report(p,out,y_true=[stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]]],prop_cliffs=prop_cliffs_trimmed, leave_grp_out=None,bins=10)
        
        if leave_grp_out is not None:
            out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES.values[leave_grp_out],compute_SD=True,convert_log10=transformed_data and revert_to_original)


            if transformed_data and revert_to_original:
                properties=original_props

            leave_grp_out_trim=np.arange(len(leave_grp_out))
            ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values[leave_grp_out] for p in properties],prop_cliffs=None, leave_grp_out=leave_grp_out_trim,bins=10)
        
        sample_train=None
        sample_val=None

        if transformed_data and revert_to_original:
            properties=log_props

        for p in properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        plt.close('all')
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)

        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
        
    def test_a_reg_innstack_dist_rdecfp_orig_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('-------------test_reg_innstack_dist_rdecfp_orig_nosw_vkmac_cvkmgkf4---------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
        
        
        properties=dv_properties[:2]
        use_log10=True
        percentages=False
        use_logit=False
        remove_outliers=False
        confidence=3.8  #3.8
        use_sample_weight=False
        encoder='CHEMBL'
        task='Regression'
        computional_load='cheap'
        scorer='r2'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        revert_to_original=False
        include_Bottleneck_features=False
        include_fps=True
        include_rdkitfeatures=True
        save_model_to_file='rd_ecfp_model.pt'
        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=include_rdkitfeatures
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
        #inner_methods, inner_stacking, single_stack, top_method, top_stacking
        model_config='inner_stacking'

        #set to true if gridsearch has to be applied (warning: normally when using inner_stacking or single_stack, gridsearch fails due to the ridiculous amount of parameters)
        force_gridsearch=False
        #set the value of randomized iterations
        randomized_iterations=30
        #use distributions for parameter selection, 
        #if True a new model is added for method family in each inner fold (the probability that all method parameters are equal in two searches is close to zero)
        distribution_defaults=True 
        #use HyperOpt optimization to find the best parameters
        hyperopt_defaults=False
        #number of xgb/lgbm threads
        xgb_threads=None
        #number of random forest threads
        rfr_threads=None
        #n_jobs for randomizedsearch/gridsearch
        n_jobs=16
        # threads for hyoperopt, set to higher value than 1 to use pyspark, faster for heavy methods, but slower fast methods
        hyperopt_threads=None

        #naming prefixes for different steps in the pipeline(s)
        prefixes={'method_prefix':'reg',
                  'dim_prefix':'reduce_dim',
                  'estimator_prefix':'est_pipe'}


        
        method_archive=RegressorArchive(method_prefix=prefixes['method_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                                 random_state=random_state,xgb_threads=xgb_threads,rfr_threads=rfr_threads)
        dim_archive=ReducedimArchive(method_prefix=prefixes['dim_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                             random_state=random_state)

        
        nb_copies=5
        method='lgbm'
        #in the case of hyperopt, it is limited to lgbm and xgb, due to uniqueness requirement in hyperopt parameters. For now workaround is provided for lgbm and xgboost. 
        if hyperopt_defaults:
            if method=='lgbm':
                method_archive,reg_list=add_lgbm_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,
                                                                 xgb_threads=xgb_threads, distribution_defaults=distribution_defaults,
                                                                 random_state=random_state,regressor=True)
            elif method=='xgb':
                method_archive,reg_list=add_xgb_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,
                                                                xgb_threads=xgb_threads,distribution_defaults=distribution_defaults,
                                                                random_state=random_state,regressor=True)
        else:
            reg_list=method_archive.duplicate_method_xtimes(method_name=method,x=nb_copies,random_state=random_state)
        
        blender_archive=method_archive

        
        red_dim_list=['passthrough']  #if pca does not have enough memory, the kernel randomly stops
        reg_list=[ 'svr','lasso','kernelridge']#+reg_list[:2]#,'sgdr','rfr']#,'lad']
        blender_list=['svr','lasso']#,'kernelridge','dtr']
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)

        transformed_data=use_log10 or use_logit
        prop_builder=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=standard_smiles_column)

        #check properties
        prop_builder.check_properties(df)        

        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=prop_builder.generate_train_properties(df)
        
        if transformed_data:
            original_props=prop_builder.original_props
            properties=log_props
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True,subset =properties,how='all' )
        df.reset_index(drop=True,inplace=True)
        weighted_samples_index={p:'<1' for p in properties}
        select_sample_weights={p:10 for p in properties}

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)
        
        if computional_load=='expensive':
            use_sample_weight=False
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=distribution_defaults,
                                    hyperopt_defaults=hyperopt_defaults,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                        red_dim_list=red_dim_list,
                             method_list=reg_list,
                             blender_list=blender_list,
                             model_config=model_config, 
                             force_gridsearch=force_gridsearch, 
                             randomized_iterations=randomized_iterations, 
                             method_archive=method_archive, 
                             dim_archive=dim_archive, 
                             blender_archive=blender_archive, 
                             prefixes=prefixes,hyperopt_threads=hyperopt_threads,
                            n_jobs_dict={'outer_jobs':1,
                                         'inner_jobs':n_jobs,
                                         'method_jobs':1},use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()


        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES


        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,properties,strategy,False,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)

        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1
                
                
        

        for p in properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                       include_Bottleneck_features=include_Bottleneck_features,include_fps=include_fps,include_rdkitfeatures=include_rdkitfeatures,
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True, 
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()
        smiles_list=stacked_model.Validation.stereo_SMILES
        out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)
        
        if transformed_data and revert_to_original:
            original_prop_clif_dict=prop_clif_dict
            prop_clif_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_clif_dict.items()}
            properties=original_props

        ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],prop_cliffs=prop_clif_dict, leave_grp_out=leave_grp_out)

        if prop_clif_dict is not None:
            prop_cliffs_trimmed={}
            for ip,p in enumerate(properties): 
                if transformed_data:
                    predict_p=log_props[ip]
                else:
                    predict_p=p
                
                out=stacked_model.predict( props =predict_p, smiles=stacked_model.Validation.stereo_SMILES.values[prop_clif_dict[p]],compute_SD=True,convert_log10=transformed_data and revert_to_original)
                
                prop_cliffs_trimmed[p]=np.arange(len(prop_clif_dict[p]))
                if len(stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]])>0:
                    ResultDesigner().show_regression_report(p,out,y_true=[stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]]],prop_cliffs=prop_cliffs_trimmed, leave_grp_out=None,bins=10)
        
        if leave_grp_out is not None:
            out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES.values[leave_grp_out],compute_SD=True,convert_log10=transformed_data and revert_to_original)


            if transformed_data and revert_to_original:
                properties=original_props

            leave_grp_out_trim=np.arange(len(leave_grp_out))
            ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values[leave_grp_out] for p in properties],prop_cliffs=None, leave_grp_out=leave_grp_out_trim,bins=10)
        
        sample_train=None
        sample_val=None

        if transformed_data and revert_to_original:
            properties=log_props

        for p in properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        plt.close('all')
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)

        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
    
    
    
    def test_a_reg_topm_NANplusECFP_dist_offline_orig_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('--------------test_a_reg_topm_NANplusECFP_dist_offline_orig_nosw_vkmac_cvkmgkf4---')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
        
        
        properties=dv_properties[:2]
        use_log10=True
        percentages=False
        use_logit=False
        remove_outliers=False
        confidence=3.8  #3.8
        use_sample_weight=True
        encoder='CHEMBL'
        task='Regression'
        computional_load='cheap'
        scorer='r2'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        revert_to_original=False
        include_Bottleneck_features=False
        include_fps=False
        include_rdkitfeatures=False
        save_model_to_file='test_model.pt'
        
        feature_generators=retrieve_default_offline_generators(model=encoder, radius=2,nbits=2048)
        feature_generators['NAN']=RandomNanInjector(feature_generators['Bottleneck'])                    
        #define used dimensionality reduction methods
        red_dim_list=['passthrough','v_threshold']#['passthrough','pca','SelectPercentile','v_threshold','Kbest','rfe','frommodel']  #if pca does not have enough memory, the kernel randomly stops
        #define possible blender/final estimator
        blender_list=['svr','lasso']#,'kernelridge','dtr']
        #define base estimators/or method used in single method
        reg_list=[ 'svr','lasso','kernelridge','sgdr']#,'lad']
        #select Features for each property, for fps define it as fps_<nbits>_<radius> even if it is not yet present in the given keys (only for fps!)
        used_features=[ ['NAN','fps_1024_3'], ['NAN','Bottleneck']]
        #select Features for each property, this can be feature type specific per property or try the same for all feature types, thus this can be 3 nested lists (property, feature, dim reduction methods) or 2 (property and dim reduction methods)
        red_dim_list_per_prop=[
                          [['passthrough'],['passthrough','pca','SelectPercentile','v_threshold','Kbest']], # feature type specific
                          ['passthrough'] #try these for all feature types
                          ]
        #set this boolean for featuretype specific dimensionality reduction,
        #if False all features are concatenated first and then given to dimensionality reduction, uses red_dim_list if red_dim_list_per_prop[i] is a double nested list for prop i
        #when True each feature set has its own dimensionality reduction type
        local_dim_red=True


        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=include_rdkitfeatures
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
        #inner_methods, inner_stacking, single_stack, top_method, top_stacking
        model_config='top_method'

        #set to true if gridsearch has to be applied (warning: normally when using inner_stacking or single_stack, gridsearch fails due to the ridiculous amount of parameters)
        force_gridsearch=False
        #set the value of randomized iterations
        randomized_iterations=30
        #use distributions for parameter selection, 
        #if True a new model is added for method family in each inner fold (the probability that all method parameters are equal in two searches is close to zero)
        distribution_defaults=True 
        #use HyperOpt optimization to find the best parameters
        hyperopt_defaults=False
        #number of xgb/lgbm threads


        #naming prefixes for different steps in the pipeline(s)
        prefixes={'method_prefix':'reg',
                  'dim_prefix':'reduce_dim',
                  'estimator_prefix':'est_pipe'}

        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)

        transformed_data=use_log10 or use_logit
        prop_builder=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=standard_smiles_column)

        #check properties
        prop_builder.check_properties(df)        

        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=prop_builder.generate_train_properties(df)
        
        if transformed_data:
            original_props=prop_builder.original_props
            properties=log_props
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True,subset =properties,how='all' )
        df.reset_index(drop=True,inplace=True)
        weighted_samples_index={p:'<1' for p in properties}
        select_sample_weights={p:10 for p in properties}

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)
        
        if computional_load=='expensive':
            use_sample_weight=False

        n_jobs_dict={'outer_jobs':None,
            'inner_jobs':-1,
            'method_jobs':1}
            
        param_Factory=ModelAndParams(model=encoder,task=task,
                                     distribution_defaults=distribution_defaults, #use distributions for parameter selection in randomized search
                                     hyperopt_defaults=hyperopt_defaults, #use hyperopt
                                     use_gpu=False,
                                     normalizer=True, #normalize input base estimators
                                     top_normalizer=False,#normalize output base estimators and thus input top estimator
                                     random_state=random_state_list,#random_state
                                     red_dim_list=red_dim_list,#list of dimensionality reducing methods keys from dim_archive
                                     method_list=reg_list,#list of classifier keys from method_archive
                                     blender_list=blender_list,#list of classifier keys from blender_archive
                                     model_config=model_config, #string with the model layout/config
                                     randomized_iterations=randomized_iterations, #number of randomized iterations
                                     n_jobs_dict=n_jobs_dict,
                                     use_sample_weight=use_sample_weight,
                                     feature_generators=feature_generators,
                                     local_dim_red=local_dim_red)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()


        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES

        clustering_algorithm=KmeansForSmiles(n_groups=30,feature_generators=retrieve_default_offline_generators(model=encoder, radius=2,nbits=2048),used_features=['Bottleneck'],random_state=42)
        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df, properties,properties, strategy,False,stacked_model, standard_smiles_column,df_smiles, test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state,clustering_algorithm=clustering_algorithm)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')
                
        clustering_algorithm2=KmeansForSmiles(n_groups=30,feature_generators=retrieve_default_offline_generators(model=encoder, radius=2,nbits=2048),used_features='Bottleneck',random_state=42)
        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state,clustering_algorithm=clustering_algorithm2)

        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1
                
                
        stacked_model.verbose=1
        for index,p in enumerate(properties):
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values
            features=used_features[index]
            dim_list=red_dim_list_per_prop[index]
            if isinstance(dim_list[0],list) and not local_dim_red:
                dim_list=None
            params_grid,blender_params = param_Factory.get_feature_params(selected_features=features,dim_list=dim_list)
            stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                      params_grid=params_grid,
                                       paramsearch=paramsearch,
                                       include_Bottleneck_features=include_Bottleneck_features,include_fps=include_fps,include_rdkitfeatures=include_rdkitfeatures,
                                       features=features,
                                      scoring=scorer,
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True,
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                      blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        stacked_model.verbose=0
        #model_str=stacked_model.print_metrics()
        smiles_list=stacked_model.Validation.stereo_SMILES
        out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)
        
        if transformed_data and revert_to_original:
            original_prop_clif_dict=prop_clif_dict
            prop_clif_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_clif_dict.items()}
            properties=original_props

        ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],prop_cliffs=prop_clif_dict, leave_grp_out=leave_grp_out)

        if prop_clif_dict is not None:
            prop_cliffs_trimmed={}
            for ip,p in enumerate(properties): 
                if transformed_data:
                    predict_p=log_props[ip]
                else:
                    predict_p=p
                
                out=stacked_model.predict( props =predict_p, smiles=stacked_model.Validation.stereo_SMILES.values[prop_clif_dict[p]],compute_SD=True,convert_log10=transformed_data and revert_to_original)
                
                prop_cliffs_trimmed[p]=np.arange(len(prop_clif_dict[p]))
                if len(stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]])>0:
                    if np.sum(~np.isnan(out[f'predicted_{p}'].astype(float)))>1:
                        ResultDesigner().show_regression_report(p,out,y_true=[stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]]],prop_cliffs=prop_cliffs_trimmed, leave_grp_out=None,bins=10)
        
        if leave_grp_out is not None:
            out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES.values[leave_grp_out],compute_SD=True,convert_log10=transformed_data and revert_to_original)


            if transformed_data and revert_to_original:
                properties=original_props

            leave_grp_out_trim=np.arange(len(leave_grp_out))
            ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values[leave_grp_out] for p in properties],prop_cliffs=None, leave_grp_out=leave_grp_out_trim,bins=10)
        
        sample_train=None
        sample_val=None

        if transformed_data and revert_to_original:
            properties=log_props

        for p in properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        plt.close('all')
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)

        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
        
    """def test_a_reg_topm_dist_online_orig_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('--------------test_a_reg_topm_dist_online_orig_nosw_vkmac_cvkmgkf4----------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()
        
        
        properties=dv_properties[:2]
        use_log10=True
        percentages=False
        use_logit=False
        remove_outliers=False
        confidence=3.8  #3.8
        use_sample_weight=False
        encoder='CHEMBL'
        task='Regression'
        computional_load='cheap'
        scorer='r2'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        revert_to_original=False
        include_Bottleneck_features=False
        include_fps=False
        include_rdkitfeatures=False
        save_model_to_file='test_model.pt'
        
        feature_generators=retrieve_default_feature_generators(model=encoder, radius=2,nbits=2048)
        #define used dimensionality reduction methods
        red_dim_list=['passthrough','v_threshold']#['passthrough','pca','SelectPercentile','v_threshold','Kbest','rfe','frommodel']  #if pca does not have enough memory, the kernel randomly stops
        #define possible blender/final estimator
        blender_list=['svr','lasso']#,'kernelridge','dtr']
        #define base estimators/or method used in single method
        reg_list=[ 'svr','lasso','kernelridge','sgdr']#,'lad']
        #select Features for each property, for fps define it as fps_<nbits>_<radius> even if it is not yet present in the given keys (only for fps!)
        used_features=[ ['Bottleneck','fps_2048_2'], ['Bottleneck']]
        #select Features for each property, this can be feature type specific per property or try the same for all feature types, thus this can be 3 nested lists (property, feature, dim reduction methods) or 2 (property and dim reduction methods)
        red_dim_list_per_prop=[
                          [['passthrough'],['passthrough','pca','SelectPercentile','v_threshold','Kbest']], # feature type specific
                          ['passthrough'] #try these for all feature types
                          ]
        #set this boolean for featuretype specific dimensionality reduction,
        #if False all features are concatenated first and then given to dimensionality reduction, uses red_dim_list if red_dim_list_per_prop[i] is a double nested list for prop i
        #when True each feature set has its own dimensionality reduction type
        local_dim_red=True


        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=include_rdkitfeatures
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
        #inner_methods, inner_stacking, single_stack, top_method, top_stacking
        model_config='top_method'

        #set to true if gridsearch has to be applied (warning: normally when using inner_stacking or single_stack, gridsearch fails due to the ridiculous amount of parameters)
        force_gridsearch=False
        #set the value of randomized iterations
        randomized_iterations=30
        #use distributions for parameter selection, 
        #if True a new model is added for method family in each inner fold (the probability that all method parameters are equal in two searches is close to zero)
        distribution_defaults=True 
        #use HyperOpt optimization to find the best parameters
        hyperopt_defaults=False
        #number of xgb/lgbm threads
        xgb_threads=None
        #number of random forest threads
        rfr_threads=None
        #n_jobs for randomizedsearch/gridsearch
        n_jobs=16
        # threads for hyoperopt, set to higher value than 1 to use pyspark, faster for heavy methods, but slower fast methods
        hyperopt_threads=None

        #naming prefixes for different steps in the pipeline(s)
        prefixes={'method_prefix':'reg',
                  'dim_prefix':'reduce_dim',
                  'estimator_prefix':'est_pipe'}


        
        method_archive=RegressorArchive(method_prefix=prefixes['method_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                                 random_state=random_state,xgb_threads=xgb_threads,rfr_threads=rfr_threads)
        dim_archive=ReducedimArchive(method_prefix=prefixes['dim_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                             random_state=random_state)

        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)

        transformed_data=use_log10 or use_logit
        prop_builder=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=standard_smiles_column)

        #check properties
        prop_builder.check_properties(df)        

        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=prop_builder.generate_train_properties(df)
        
        if transformed_data:
            original_props=prop_builder.original_props
            properties=log_props
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True,subset =properties,how='all' )
        df.reset_index(drop=True,inplace=True)
        weighted_samples_index={p:'<1' for p in properties}
        select_sample_weights={p:10 for p in properties}

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)
        
        if computional_load=='expensive':
            use_sample_weight=False
            
        n_jobs_dict={'outer_jobs':None,
            'inner_jobs':-1,
            'method_jobs':1}
        
        param_Factory=ModelAndParams(model=encoder,task=task,
                                     distribution_defaults=distribution_defaults, #use distributions for parameter selection in randomized search
                                     hyperopt_defaults=hyperopt_defaults, #use hyperopt
                                     use_gpu=False,
                                     normalizer=True, #normalize input base estimators
                                     top_normalizer=False,#normalize output base estimators and thus input top estimator
                                     random_state=random_state_list,#random_state
                                     red_dim_list=red_dim_list,#list of dimensionality reducing methods keys from dim_archive
                                     method_list=reg_list,#list of classifier keys from method_archive
                                     blender_list=blender_list,#list of classifier keys from blender_archive
                                     model_config=model_config, #string with the model layout/config
                                     randomized_iterations=randomized_iterations, #number of randomized iterations
                                     n_jobs_dict=n_jobs_dict,
                                     use_sample_weight=use_sample_weight,
                                     feature_generators=feature_generators,
                                     local_dim_red=local_dim_red)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()


        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES


        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,properties,strategy,False,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)

        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1
                
                
        
        for index,p in enumerate(properties):
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values
            features=used_features[index]
            dim_list=red_dim_list_per_prop[index]
            if isinstance(dim_list[0],list) and not local_dim_red:
                dim_list=None
            params_grid,blender_params = param_Factory.get_feature_params(selected_features=features,dim_list=dim_list)
            stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                      params_grid=params_grid,
                                       paramsearch=paramsearch,
                                       include_Bottleneck_features=include_Bottleneck_features,include_fps=include_fps,include_rdkitfeatures=include_rdkitfeatures,
                                       features=features,
                                      scoring=scorer,
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True, 
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                      blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)

        #model_str=stacked_model.print_metrics()
        smiles_list=stacked_model.Validation.stereo_SMILES
        out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)
        
        if transformed_data and revert_to_original:
            original_prop_clif_dict=prop_clif_dict
            prop_clif_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_clif_dict.items()}
            properties=original_props

        ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],prop_cliffs=prop_clif_dict, leave_grp_out=leave_grp_out)

        if prop_clif_dict is not None:
            prop_cliffs_trimmed={}
            for ip,p in enumerate(properties): 
                if transformed_data:
                    predict_p=log_props[ip]
                else:
                    predict_p=p
                
                out=stacked_model.predict( props =predict_p, smiles=stacked_model.Validation.stereo_SMILES.values[prop_clif_dict[p]],compute_SD=True,convert_log10=transformed_data and revert_to_original)
                
                prop_cliffs_trimmed[p]=np.arange(len(prop_clif_dict[p]))
                if len(stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]])>0:
                    ResultDesigner().show_regression_report(p,out,y_true=[stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]]],prop_cliffs=prop_cliffs_trimmed, leave_grp_out=None,bins=10)
        
        if leave_grp_out is not None:
            out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES.values[leave_grp_out],compute_SD=True,convert_log10=transformed_data and revert_to_original)


            if transformed_data and revert_to_original:
                properties=original_props

            leave_grp_out_trim=np.arange(len(leave_grp_out))
            ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values[leave_grp_out] for p in properties],prop_cliffs=None, leave_grp_out=leave_grp_out_trim,bins=10)
        
        sample_train=None
        sample_val=None

        if transformed_data and revert_to_original:
            properties=log_props

        for p in properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        plt.close('all')
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)

        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)"""
        
    def test_d_reg_innmet_hyperdist_orig_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('--------------test_d_reg_innmet_hyperdist_orig_nosw_vkmac_cvkmgkf4----------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()

        properties=dv_properties[:2]
        use_log10=False
        percentages=False
        use_logit=False
        remove_outliers=False
        confidence=3.8  #3.8
        use_sample_weight=False
        encoder='CHEMBL'
        task='Regression'
        computional_load='cheap'
        scorer='r2'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        revert_to_original=False
        include_Bottleneck_features=True
        include_fps=False
        include_rdkitfeatures=False
        save_model_to_file='test_model.pt'
        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=include_rdkitfeatures
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
        #inner_methods, inner_stacking, single_stack, top_method, top_stacking
        model_config='inner_methods'

        #set to true if gridsearch has to be applied (warning: normally when using inner_stacking or single_stack, gridsearch fails due to the ridiculous amount of parameters)
        force_gridsearch=False
        #set the value of randomized iterations
        randomized_iterations=30
        #use distributions for parameter selection, 
        #if True a new model is added for method family in each inner fold (the probability that all method parameters are equal in two searches is close to zero)
        distribution_defaults=True 
        #use HyperOpt optimization to find the best parameters
        hyperopt_defaults=True
        #number of xgb/lgbm threads
        xgb_threads=2
        #number of random forest threads
        rfr_threads=None
        #n_jobs for randomizedsearch/gridsearch
        n_jobs=16
        # threads for hyoperopt, set to higher value than 1 to use pyspark, faster for heavy methods, but slower fast methods
        hyperopt_threads=None
        
        #naming prefixes for different steps in the pipeline(s)
        prefixes={'method_prefix':'reg',
                  'dim_prefix':'reduce_dim',
                  'estimator_prefix':'est_pipe'}

        
        method_archive=RegressorArchive(method_prefix=prefixes['method_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                                 random_state=random_state,xgb_threads=xgb_threads,rfr_threads=rfr_threads)
        dim_archive=ReducedimArchive(method_prefix=prefixes['dim_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                             random_state=random_state)

        
        nb_copies=5
        method='lgbm'
        #in the case of hyperopt, it is limited to lgbm and xgb, due to uniqueness requirement in hyperopt parameters. For now workaround is provided for lgbm and xgboost. 
        if hyperopt_defaults:
            if method=='lgbm':
                method_archive,reg_list=add_lgbm_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,
                                                                 xgb_threads=xgb_threads, distribution_defaults=distribution_defaults,
                                                                 random_state=random_state,regressor=True)
            elif method=='xgb':
                method_archive,reg_list=add_xgb_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,
                                                                xgb_threads=xgb_threads,distribution_defaults=distribution_defaults,
                                                                random_state=random_state,regressor=True)
        else:
            reg_list=method_archive.duplicate_method_xtimes(method_name=method,x=nb_copies,random_state=random_state)
        
        blender_archive=method_archive

        
        red_dim_list=['passthrough','v_threshold']  #if pca does not have enough memory, the kernel randomly stops
        reg_list=[ 'svr','lasso','kernelridge']+reg_list[:2]#,'sgdr','rfr']#,'lad']
        blender_list=['svr','lasso']#,'kernelridge','dtr']

        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)

        transformed_data=use_log10 or use_logit
        prop_builder=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=standard_smiles_column)

        #check properties
        prop_builder.check_properties(df)        

        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=prop_builder.generate_train_properties(df)
        
        if transformed_data:
            original_props=prop_builder.original_props
            properties=log_props
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True,subset =properties,how='all' )
        df.reset_index(drop=True,inplace=True)
        weighted_samples_index={p:'<1' for p in properties}
        select_sample_weights={p:10 for p in properties}

        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)
        
        if computional_load=='expensive':
            use_sample_weight=False
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=distribution_defaults,
                                    hyperopt_defaults=hyperopt_defaults,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                        red_dim_list=red_dim_list,
                             method_list=reg_list,
                             blender_list=blender_list,
                             model_config=model_config, 
                             force_gridsearch=force_gridsearch, 
                             randomized_iterations=randomized_iterations, 
                             method_archive=method_archive, 
                             dim_archive=dim_archive, 
                             blender_archive=blender_archive, 
                             prefixes=prefixes,hyperopt_threads=hyperopt_threads,
                            n_jobs=n_jobs,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES


        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,properties,strategy,False,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)

        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1
                
                
        

        for p in properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                       include_Bottleneck_features=include_Bottleneck_features,include_fps=include_fps,include_rdkitfeatures=include_rdkitfeatures,
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True, 
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()
        smiles_list=stacked_model.Validation.stereo_SMILES
        out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)
        
        if transformed_data and revert_to_original:
            original_prop_clif_dict=prop_clif_dict
            prop_clif_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_clif_dict.items()}
            properties=original_props

        ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],prop_cliffs=prop_clif_dict, leave_grp_out=leave_grp_out)

        if prop_clif_dict is not None:
            prop_cliffs_trimmed={}
            for ip,p in enumerate(properties): 
                if transformed_data:
                    predict_p=log_props[ip]
                else:
                    predict_p=p
                
                out=stacked_model.predict( props =predict_p, smiles=stacked_model.Validation.stereo_SMILES.values[prop_clif_dict[p]],compute_SD=True,convert_log10=transformed_data and revert_to_original)
                
                prop_cliffs_trimmed[p]=np.arange(len(prop_clif_dict[p]))
                if len(stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]])>0:
                    ResultDesigner().show_regression_report(p,out,y_true=[stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]]],prop_cliffs=prop_cliffs_trimmed, leave_grp_out=None,bins=10)
        
        if leave_grp_out is not None:
            out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES.values[leave_grp_out],compute_SD=True,convert_log10=transformed_data and revert_to_original)


            if transformed_data and revert_to_original:
                properties=original_props

            leave_grp_out_trim=np.arange(len(leave_grp_out))
            ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values[leave_grp_out] for p in properties],prop_cliffs=None, leave_grp_out=leave_grp_out_trim,bins=10)
        
        sample_train=None
        sample_val=None

        if transformed_data and revert_to_original:
            properties=log_props

        for p in properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        plt.close('all')
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)

        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
    
    def test_d_reg_sinstack_hyper_orig_nosw_vkmac_cvkmgkf4(self):
        print('----------------------------------------------------------------------------------')
        print('----------------test_reg_sinstack_hyper_orig_nosw_vkmac_cvkmgkf4------------------')
        print('----------------------------------------------------------------------------------')
        test_obj=test_util()

        properties=dv_properties[:2]
        use_log10=False
        percentages=False
        use_logit=False
        remove_outliers=False
        confidence=3.8  #3.8
        use_sample_weight=False
        encoder='CHEMBL'
        task='Regression'
        computional_load='cheap'
        scorer='r2'
        random_state=5
        random_state_list=[1,7,42,55,3]
        n_jobs=20
        val_clustering='Bottleneck'
        val_include_chirality=False
        val_km_groups=30
        if val_clustering=="Butina":
            val_butina_cutoff=0.5
        else:
            val_butina_cutoff=[0.2,0.4,0.6]  
        strategy='mixed'
        test_size=0.25
        cv_clustering='Bottleneck'
        include_chirality=False
        km_groups=20
        if cv_clustering=="Butina":
            butina_cutoff=0.6
        else:
            butina_cutoff=[0.2,0.4,0.6]
        cross_val_split='GKF'
        outer_folds=4
        revert_to_original=False
        include_Bottleneck_features=True
        include_fps=False
        include_rdkitfeatures=False
        save_model_to_file='test_model.pt'
        
        standard_smiles_column='SMILES_STEREO'
        adme_il17=False
        check_rdkit_desc=include_rdkitfeatures
        
        df=test_obj.read_data(file_name=data,smiles_column=smiles_col, verbose=verbose, nb_samples=157,
                              standard_smiles_column=standard_smiles_column, adme_il17=adme_il17, check_rdkit_desc=check_rdkit_desc)
        df_smiles=df[standard_smiles_column]
        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)
        
        #inner_methods, inner_stacking, single_stack, top_method, top_stacking
        model_config='single_stack'

        #set to true if gridsearch has to be applied (warning: normally when using inner_stacking or single_stack, gridsearch fails due to the ridiculous amount of parameters)
        force_gridsearch=False
        #set the value of randomized iterations
        randomized_iterations=30
        #use distributions for parameter selection, 
        #if True a new model is added for method family in each inner fold (the probability that all method parameters are equal in two searches is close to zero)
        distribution_defaults=False 
        #use HyperOpt optimization to find the best parameters
        hyperopt_defaults=True
        #number of xgb/lgbm threads
        xgb_threads=2
        #number of random forest threads
        rfr_threads=None
        #n_jobs for randomizedsearch/gridsearch
        n_jobs=16
        # threads for hyoperopt, set to higher value than 1 to use pyspark, faster for heavy methods, but slower fast methods
        hyperopt_threads=None

        #naming prefixes for different steps in the pipeline(s)
        prefixes={'method_prefix':'reg',
                  'dim_prefix':'reduce_dim',
                  'estimator_prefix':'est_pipe'}


        
        method_archive=RegressorArchive(method_prefix=prefixes['method_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                                 random_state=random_state,xgb_threads=xgb_threads,rfr_threads=rfr_threads)
        dim_archive=ReducedimArchive(method_prefix=prefixes['dim_prefix'],distribution_defaults=distribution_defaults,hyperopt_defaults=hyperopt_defaults,
                             random_state=random_state)

        
        nb_copies=5
        method='xgb'
        #in the case of hyperopt, it is limited to lgbm and xgb, due to uniqueness requirement in hyperopt parameters. For now workaround is provided for lgbm and xgboost. 
        if hyperopt_defaults:
            if method=='lgbm':
                method_archive,reg_list=add_lgbm_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,
                                                                 xgb_threads=xgb_threads, distribution_defaults=distribution_defaults,
                                                                 random_state=random_state,regressor=True)
            elif method=='xgb':
                method_archive,reg_list=add_xgb_xtimes_hyperopt(method_archive,prefixes['method_prefix'],nb_copies,
                                                                xgb_threads=xgb_threads,distribution_defaults=distribution_defaults,
                                                                random_state=random_state,regressor=True)
        else:
            reg_list=method_archive.duplicate_method_xtimes(method_name=method,x=nb_copies,random_state=random_state)
        
        blender_archive=method_archive

        
        red_dim_list=['passthrough']  #if pca does not have enough memory, the kernel randomly stops
        reg_list=[ 'svr','lasso','kernelridge']#+reg_list[:2]#,'sgdr','rfr']#,'lad']
        blender_list=['svr','lasso']#,'kernelridge','dtr']

        
        df.dropna(inplace=True,how='all', subset = properties)
        df.reset_index(drop=True,inplace=True)

        transformed_data=use_log10 or use_logit
        prop_builder=PropertyTransformer(properties,remove_outliers,confidence,use_log10,use_logit,percentages,standard_smiles_column=standard_smiles_column)

        #check properties
        prop_builder.check_properties(df)        

        #prepare properties: e.g. create classes, apply transformations etc..c 
        df,log_props=prop_builder.generate_train_properties(df)
        
        if transformed_data:
            original_props=prop_builder.original_props
            properties=log_props
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True,subset =properties,how='all' )
        df.reset_index(drop=True,inplace=True)

        
        weighted_samples_index={p:'<1' for p in properties}
        select_sample_weights={p:10 for p in properties}
        
        if use_sample_weight:
            df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)
        
        if computional_load=='expensive':
            use_sample_weight=False
        param_Factory=ModelAndParams(model=encoder,
                                     task=task,
                                     computional_load=computional_load,
                                     distribution_defaults=distribution_defaults,
                                    hyperopt_defaults=hyperopt_defaults,
                                     use_gpu=False,
                                     normalizer=True,
                                     top_normalizer=True,
                                     random_state=random_state_list,
                        red_dim_list=red_dim_list,
                             method_list=reg_list,
                             blender_list=blender_list,
                             model_config=model_config, 
                             force_gridsearch=force_gridsearch, 
                             randomized_iterations=randomized_iterations, 
                             method_archive=method_archive, 
                             dim_archive=dim_archive, 
                             blender_archive=blender_archive, 
                             prefixes=prefixes,hyperopt_threads=hyperopt_threads,
                            n_jobs=n_jobs,use_sample_weight=use_sample_weight)
        stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()


        
        standard_smiles_column='stereo_SMILES'
        df_smiles=df.stereo_SMILES


        Train,Validation,leave_grp_out,prop_clif_dict= test_obj.create_clf_validation(df,properties,properties,strategy,False,stacked_model,standard_smiles_column,df_smiles,
                                                         test_size,val_clustering,val_km_groups,val_butina_cutoff,val_include_chirality,verbose,random_state)
        stacked_model.Validation=Validation
        stacked_model.Train=Train
        stacked_model.smiles=standard_smiles_column
        #property check
        for p in properties:
            prop_count=df[p].count()
            if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
                print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

        stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality ,random_state=random_state)

        if outer_folds>km_groups:
            if km_groups>2:
                outer_folds=km_groups-1
            else:
                km_groups=outer_folds+1
                
                
        

        for p in properties:
            sample_weight=None
            if use_sample_weight:
                sample_weight=stacked_model.Train[f'sw_{p}'].values

            stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                        params_grid=params_grid,
                                       paramsearch=paramsearch,
                                      scoring=scorer,#'neg_mean_absolute_error',#
                                       include_Bottleneck_features=include_Bottleneck_features,include_fps=include_fps,include_rdkitfeatures=include_rdkitfeatures,
                                      cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                      use_memory=True,
                                      plot_validation=True, 
                                     refit=False,# no refit with validation. comes later,
                                     blender_params=blender_params
                                      ,prefix_dict=prefixes,random_state=random_state,sample_weight=sample_weight)
        #model_str=stacked_model.print_metrics()
        smiles_list=stacked_model.Validation.stereo_SMILES
        out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)
        
        if transformed_data and revert_to_original:
            original_prop_clif_dict=prop_clif_dict
            prop_clif_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_clif_dict.items()}
            properties=original_props

        ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],prop_cliffs=prop_clif_dict, leave_grp_out=leave_grp_out)

        if prop_clif_dict is not None:
            prop_cliffs_trimmed={}
            for ip,p in enumerate(properties): 
                if transformed_data:
                    predict_p=log_props[ip]
                else:
                    predict_p=p
                
                out=stacked_model.predict( props =predict_p, smiles=stacked_model.Validation.stereo_SMILES.values[prop_clif_dict[p]],compute_SD=True,convert_log10=transformed_data and revert_to_original)
                
                prop_cliffs_trimmed[p]=np.arange(len(prop_clif_dict[p]))
                if len(stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]])>0:
                    ResultDesigner().show_regression_report(p,out,y_true=[stacked_model.Validation[f'{p}'].values[prop_clif_dict[p]]],prop_cliffs=prop_cliffs_trimmed, leave_grp_out=None,bins=10)
        
        if leave_grp_out is not None:
            out=stacked_model.predict( props =None, smiles=stacked_model.Validation.stereo_SMILES.values[leave_grp_out],compute_SD=True,convert_log10=transformed_data and revert_to_original)


            if transformed_data and revert_to_original:
                properties=original_props

            leave_grp_out_trim=np.arange(len(leave_grp_out))
            ResultDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values[leave_grp_out] for p in properties],prop_cliffs=None, leave_grp_out=leave_grp_out_trim,bins=10)
        
        sample_train=None
        sample_val=None

        if transformed_data and revert_to_original:
            properties=log_props

        for p in properties:
            sample_train=None
            sample_val=None
            if use_sample_weight:
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
            stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)

        ## clean the class first by removing the computed features
        stacked_model.clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, save_model_to_file)
        plt.close('all')
        stacked_model.validate(df=None, # df with smiles and the properties
                               props=None, # name of the task 
                            true_props=None,# name of the property in df
                            smiles=standard_smiles_column)

        stacked_model2= load_model( save_model_to_file,use_gpu=False) 
        #stacked_model2.print_metrics()
        self.assertEqual(True,True)
    
    
