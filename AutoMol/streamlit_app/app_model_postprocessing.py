"""
**************************************************************************************
AutoMoL: Pipeline for automated machine learning for drug design.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
**************************************************************************************
Module that contains the functionality for postprocessing of JnJ AutoML stacking models

includes methods:

gen_pm_pdf_summary: generate pdf summary
gen_pm_download_model: update and download model

"""

import streamlit as st

from automol.plotly_util import create_figure_captions, generate_report,get_figures_from_types
from automol.stacking import save_model

from app_model_training import from_list_to_markdown

import ssl
import smtplib
from email.message import EmailMessage

import yaml

from typing import List 
 
def send_email(sender_email: str, receivers: List[str], header: str, subject: str, content: str, attachment: str):
        port = 25
        smtp_server = 'smtp.eu.jnj.com'
        context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_1)
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls(context=context)
            msg = EmailMessage()
            msg.set_content(content)
            msg['Subject'] = subject
            msg['From'] = header
            msg['To'] = receivers
            with open(attachment, 'rb') as content_file:
                content = content_file.read()
            msg.add_attachment(content, maintype='application', subtype=attachment.split('.')[-1], filename=attachment)
            server.send_message(msg=msg, from_addr=sender_email, to_addrs=receivers)
            pass
        pass
    
    
def gen_pm_pdf_summary(classifier,model_param,plotly_dictionary,save_output_to_file,model_str,class_properties,
                       generate_pdf, title, authors, mails, summary, send_mail, sender,recipients ):
    nb_columns = st.selectbox(
       'The number of figures columns in the report',
      [1,2,3,4,5],index=2,help="The figures are saved in columns, this value sets the number of columns")
    
    _,types,available_types=create_figure_captions(plotly_dictionary.keys())
    markdown_str=from_list_to_markdown(string_list=types) 
    st.markdown(markdown_str)
    if classifier:
        def_fig=available_types[:2]
    else:
        def_fig=available_types
    figure_types = st.multiselect(
            "Select figure types to be saved in the report",
            sorted(available_types),
            default=def_fig,
            help="Select figures from the list",
            )
    
    if generate_pdf:
        selected_saved_figures=get_figures_from_types(plotly_dictionary=plotly_dictionary,properties=class_properties,types=figure_types)
        finetuned_figures = st.multiselect(
            "Finetune figure selection",
            selected_saved_figures,
            default=selected_saved_figures,
            help="Select figures from the list",
            )
        captions,types,_=create_figure_captions(finetuned_figures)
        pdf=generate_report(plotly_dictionary,finetuned_figures,captions,types,title=title, authors=authors,
                 mails=mails,summary=summary,model_param=model_param,model_tostr_list=model_str,properties=class_properties,nb_columns=nb_columns )
        st.download_button(
                "⬇️ Download pdf summary",
                data=pdf,
                file_name=save_output_to_file,
            )
        
    return figure_types, nb_columns

def gen_pm_yaml_file(model_param,save_output_to_file):

    yaml_file=save_output_to_file.replace(".pdf", ".yaml")
    st.download_button(
                "⬇️ Download yaml file",
                data=yaml.dump(model_param),
                file_name=yaml_file,
        )
    
    
def gen_pm_download_model(stacked_model,load_model_file,):
        
    st.warning("Preparing the deployment model, cleans and refits the model")
    use_weighted_samples=False
    if f'sw_{next(iter(stacked_model.models))}' in stacked_model.Train.columns:
        use_weighted_samples=st.checkbox(f'Use sample weights when refitting?')
    deploy_model=False
    if hasattr(stacked_model, 'Validation'):
        deploy_model=st.button('Prepare model for deployment')
    if deploy_model:
        if use_weighted_samples:
            for p in stacked_model.models:
                if stacked_model.prefix_dict:
                    prefixes=stacked_model.prefix_dict
                else:
                    if hasattr(stacked_model, 'labelnames'):
                        prefixes={'method_prefix':'clf',
                               'dim_prefix':'reduce_dim',
                               'estimator_prefix':'est_pipe'}
                    else:
                        prefixes={'method_prefix':'reg',
                                   'dim_prefix':'reduce_dim',
                                   'estimator_prefix':'est_pipe'}
                sample_train=stacked_model.Train[f'sw_{p}'].values
                sample_val=stacked_model.Validation[f'sw_{p}'].values
                stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)
        else:
            stacked_model.refit_model()

        if hasattr(stacked_model, 'leave_group_out'):
            delattr(stacked_model, 'leave_group_out')
        if hasattr(stacked_model, 'prop_cliff_dict'):
            delattr(stacked_model, 'prop_cliff_dict')
        stacked_model.deep_clean()
        stacked_model.compute_SD=True
        save_model(stacked_model, load_model_file)
        st.experimental_memo.clear()
        st.experimental_rerun()
    
    with open(load_model_file, 'rb') as f:
        st.download_button(f'⬇️ Download model {load_model_file}', f, file_name=load_model_file)

        
        