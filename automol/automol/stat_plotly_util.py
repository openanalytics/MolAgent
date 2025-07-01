"""implementation of the plotly visualization.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""

import sys ,os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score,roc_curve, auc, classification_report,accuracy_score,f1_score,recall_score,precision_score       
from sklearn.metrics import silhouette_samples, silhouette_score


from scipy.stats import pearsonr
import plotly.graph_objects as go

from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AllChem
from rdkit import  DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.PandasTools import ChangeMoleculeRendering

from .stat_util import generate_distance_matrix_lowerdiagonal
from IPython.display import SVG

#Bokeh library for plotting
import json
from bokeh.plotting import figure, show, output_notebook, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.transform import factor_cmap
from bokeh.plotting import figure, output_file, save
from bokeh.models import Title
from bokeh.models import Span
#################################

def get_color(colorscale_name, loc):
    """
    retrieve color based on scale and index
    """
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...] 
    colorscale = cv.validate_coerce(colorscale_name)
    
    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)
        

import plotly.colors
from PIL import ImageColor

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

         colorscale: A plotly continuous colorscale defined with RGB string colors.
         intermed: value in the range [0, 1]
    Returns:
        color in rgb string format
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )

def plotly_silhouette_scores(smiles_data,X,cluster_labels,distance,cmap,fig_size=(600, 600),verbose=False, results_dict=None):
    """
    plots the silhouette scores 
    
    from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    
    Args:
         X: data matrics of distance matrix (set distance to precomputed if distance matrix)
         cluster_labels: the assigned labels
         distance: the distance metric or "precomputed"
    """
    n_clusters=np.max(cluster_labels)+1
    ############################
    #DL
    fig = go.Figure()

    sihl_dict={}
    silhouette_avg = float(silhouette_score(X=X, labels=cluster_labels,metric=distance))
    if verbose: print(f"The average silhouette_score using DL features and metric {distance} is : {np.round(silhouette_avg,4)}")
    if results_dict is not None:
        results_dict[f'silhouette score DL+{distance}']= {np.round(silhouette_avg,4)}
    sample_silhouette_values = silhouette_samples(X=X, labels=cluster_labels,metric=distance)
    y_lower = 10
    y_ticks=[]
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values=np.append(ith_cluster_silhouette_values,[-1e-8,0])
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = get_color(cmap, (float(i) / n_clusters))
        
        fig.add_trace(go.Scatter(x=ith_cluster_silhouette_values, y=np.arange(y_lower, y_upper), 
            mode='lines',
            showlegend=False,
            line=dict(color=color, width=2)))
        y_boundary=y_upper*np.ones(ith_cluster_silhouette_values.shape)
        y_boundary[ith_cluster_silhouette_values<0]=y_lower
        fig.add_trace(go.Scatter(
            x=ith_cluster_silhouette_values,
            y=y_boundary,
            showlegend=False,
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines',line_color=color, fillcolor=f"rgba{color[3:-1]}, 0.7)"))

        # Compute the new y_lower for next plot
        y_ticks.append(y_lower)
        y_lower = y_upper + 10  # 10 for the 0 samples

    # The vertical line for average silhouette score of all the values
    fig.add_vline(x=silhouette_avg,line=dict(color="red", width=2, dash='dash'),
                 annotation_text=f'average silhouette={np.round(silhouette_avg,3)}',
                 annotation_font_size=14,
                 annotation_position="top right")

    
    fig.update_layout(title=f'Silhouette plot DL features with metric: {distance}<br><sup>Average score: {np.round(silhouette_avg,3)}</sup>',
                title_font_size=18,
               xaxis_title="The silhouette coefficient values",
               yaxis_title="Cluster label",
               xaxis_range=[-0.5, 1],
               yaxis_range=[0, y_upper+5],font=dict(size=18),height=int(n_clusters*fig_size[1]/10.0), width=fig_size[0],
                xaxis = dict(
                        tickmode = 'array',
                        tickvals = [-0.5,-0.2,-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1],
                        ticktext = [-0.5,-0.2,-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]
                    ),
                yaxis = dict(
                        tickmode = 'array',
                        tickvals = y_ticks,
                        ticktext = [i for i in range(len(y_ticks))]
                    )
                 )
    sihl_dict[f'{distance}_sihl']=fig
    
    ############################
    #Fingerprints
    mols =[Chem.MolFromSmiles(s) for s in smiles_data]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
    # scaffold sets
    nfps=len(fps)
    dists = []
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
        
    dist_mat=generate_distance_matrix_lowerdiagonal(dists,nfps)
    
    fig = go.Figure()

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = float(silhouette_score(X=dist_mat, labels=cluster_labels,metric='precomputed'))
    if verbose: print(f"The average silhouette_score using ECFP and Tanimoto is: {np.round(silhouette_avg,4)}")
    if results_dict is not None:
        results_dict[f'silhouette score ECFP+Tanimoto']= {np.round(silhouette_avg,4)}

    sample_silhouette_values = silhouette_samples(X=dist_mat, labels=cluster_labels,metric='precomputed')
    y_lower = 10
    y_ticks=[]
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values=np.append(ith_cluster_silhouette_values,[-1e-8,0])
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = get_color(cmap, (float(i) / n_clusters))
        
        fig.add_trace(go.Scatter(x=ith_cluster_silhouette_values, y=np.arange(y_lower, y_upper), 
            mode='lines',
            showlegend=False,
            line=dict(color=color, width=2)))
        y_boundary=y_upper*np.ones(ith_cluster_silhouette_values.shape)
        y_boundary[ith_cluster_silhouette_values<0]=y_lower
        fig.add_trace(go.Scatter(
            x=ith_cluster_silhouette_values,
            y=y_boundary,
            showlegend=False,
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines',line_color=color, fillcolor=f"rgba{color[3:-1]}, 0.7)"))

        # Compute the new y_lower for next plot
        y_ticks.append(y_lower)
        y_lower = y_upper + 10  # 10 for the 0 samples


    # The vertical line for average silhouette score of all the values
    fig.add_vline(x=silhouette_avg,line=dict(color="red", width=2, dash='dash'),
                 annotation_text=f'average silhouette={np.round(silhouette_avg,3)}',
                 annotation_font_size=14,
                 annotation_position="top right")

    fig.update_layout(title=f'Silhouette plot for ECFP with Tanimoto<br><sup>Average score: {np.round(silhouette_avg,3)}</sup>',
                title_font_size=18,
               xaxis_title="The silhouette coefficient values",
               yaxis_title="Cluster label",
               xaxis_range=[-0.5, 1],
               yaxis_range=[0, y_upper+5],font=dict(size=18),height=int(n_clusters*fig_size[1]/10.0), width=fig_size[0],
                xaxis = dict(
                        tickmode = 'array',
                        tickvals = [-0.5,-0.2,-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1],
                        ticktext = [-0.5,-0.2,-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]
                    ),
                yaxis = dict(
                        tickmode = 'array',
                        tickvals = y_ticks,
                        ticktext = [i for i in range(len(y_ticks))]
                    )
                 )
    sihl_dict[f'tanimoto_sihl']=fig

    return sihl_dict



def plotly_confusion_bars_from_continuos(df_raw,pro1=None,pro2=None, 
                bins2=None,bins1= range(1,11),color=None,#['red','yellow','green']
                fig_size=(600, 600),
                title=None,
                leg_title='',
                x_title=None, results_dict=None):
    """
    Displays barplots where bins of one property is set wrt to bins of the other property
    
    Args:
         df_raw: dataframe containing both properties
         pro1: column name property 1
         pro2: column name property 2
         bins2: bins of the second property
         bin1: bins of the first property
         title: title of the plot
         leg_title: legend title
         x_title: title of x-labels 
    """
    df= df_raw.loc[(df_raw[pro1].notnull()) &(df_raw[pro2].notnull()),[pro1,pro2]].copy()
    bins1 = [min(bins1)-1000] + list(bins1) + [max(bins1)+1000]
    labels1=[f'<{np.round(bins1[1],2)}']+ [f"{np.round(bins1[i],2)}-{np.round(bins1[i+1],2)}" for i in range(1,len(bins1)-2)]+ [f'>{np.round(bins1[-2],2)}']
    #print(bins1,labels1)
    try:
        df['pro1_bin'] = pd.cut(df[pro1],bins=bins1,labels=labels1)
    except ValueError:
        try:
            df['pro1_bin'] = pd.cut(df[pro1],bins=bins1,labels=labels1,ordered=False)
        except ValueError:
            return go.Figure()
    lims=1000
    bins2 = [min(bins2)-lims] + list(bins2) + [max(bins2)+lims]
    labels2=[f'<{np.round(bins2[1],2)}']+ [f"{np.round(bins2[i],2)}-{np.round(bins2[i+1],2)}" for i in range(1,len(bins2)-2)]+ [f'>{np.round(bins2[-2],2)}']
    nr_blues=len(labels2)//2
    nr_reds=len(labels2)-1*(nr_blues)
    #blue_colors = np.array([[0.,0.2,1,1]]*nr_blues)
    #blue_colors[:,3]=np.linspace(0.8, 0.1, nr_blues)
    green_colors = np.array([[0.,100.0/255,0.,1]]*nr_blues);    green_colors[:,3]=np.linspace(0.1, 0.8, nr_blues)
    red_colors = np.array([[0.9,0.0,0,1]]*nr_reds)
    red_colors[:,3]=np.linspace(0.8, 0.1, nr_reds)
    rgba_colors=np.vstack([green_colors[::-1],red_colors[::-1]])
    
    if not color or len(color) !=len(labels2):
        #color=[ c for c in mcolors.TABLEAU_COLORS][0:len(labels2)]
        color=rgba_colors
    colors2= {l:c for l,c in zip(labels2,color)}
    try:
        df['pro2_bin'] = pd.cut(df[pro2],bins=bins2,labels=labels2)
    except ValueError:
        try:
            df['pro2_bin'] = pd.cut(df[pro2],bins=bins2,labels=labels2,ordered=False)
        except ValueError:
            return go.Figure()
    
    ##############
    ##############
    res = {}
    bin_counts = []
    for k,v in df.groupby('pro1_bin'):
        if len(v) < 1:continue
        res[k]=v['pro2_bin'].value_counts(normalize=True).to_dict()
        bin_counts.append(len(v))
    dftm=pd.DataFrame.from_dict(res, orient='index').reindex(columns=labels2)#.sort_values('pro2_bin')
    fig = go.Figure(data=[
        go.Bar(name=col,
               x=dftm.index,
               y=dftm[col],
               marker_color=f'rgba({colors2[col][0]},{colors2[col][1]},{colors2[col][2]},{colors2[col][3]})',
               customdata= np.stack(([int(bin_counts[index]*y_val) for index,y_val in enumerate(dftm[col])], [ labels2[col_index] for i in dftm[col] ]),-1),
               hovertemplate='<b>Cnt</b>: %{customdata[0]:d} <br><b>Rel. Cnt</b>: %{y:.2f} <br><b>Abs. err</b>: %{customdata[1]}<br><b>Bin</b>: %{x} ',
               text=[np.round(y_val,2) if y_val>0.05 else "" for y_val in dftm[col]],
               textposition='inside',
               textfont_size=10,
               textfont_color="black"
              ) 
        for col_index,col in enumerate(dftm) ])
    # Change the bar mode
    fig.update_layout(barmode='stack')
    if not x_title: x_title=f"Bins of {pro1}"
    for i,t in enumerate(bin_counts):
        fig.add_annotation(x=labels1[i], y = 1.0,
                           text = str(t),
                           showarrow = False,
                           yshift = 8,
                           font=dict(family="Courier New, monospace",
                                     size=10,
                                     color="black"
                                    )
                          )
    
    fig.update_layout(title=f'Absolute error barplot <br><sup>{title}</sup>',
                    title_font_size=18,height=fig_size[1], width=fig_size[0],
                   xaxis_title=x_title,
                   yaxis_title=f'Relative counts',
                    legend=dict(
                        title_text=leg_title,
                        traceorder="normal",
                        font=dict(
                            size=11,
                            color="black"
                            )
                        )
                     )
    return fig

def moving_average(a, n=3) :
    """
    computes centered moving average for a given vector a and window n
    """
    assert len(a)>=2*(n-1)+1, 'window n too large'
    ret = np.cumsum(a, dtype=float)
    reverse = np.cumsum(a[::-1], dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    reverse[n:] = reverse[n:] - reverse[:-n]
    ret[n-1:-n+1]=ret[n-1:-n+1]-a[n-1:-n+1]
    ret[n-1:-n+1]=(ret[n-1:-n+1]+reverse[-n:n-2:-1])/(2*(n-1)+1)
    for i in range(n-1):
        ret[i]=reverse[-1-i]
        for j in range(i):
            ret[i]=ret[i]+a[j]
            ret[-1-i]=ret[-1-i]+a[-1-j]
        ret[i]=ret[i]/(n+i)
        ret[-1-i]=ret[-1-i]/(n+i)
    return ret


def get_moving_abs_err_mean(y_true,y_pred,window=30):
    """
    takes the average of the absolute error over a window left and right for each point, no padding at end points
    """
    indices=np.argsort(y_true)
    return y_true[indices], moving_average(np.absolute(y_true[indices]-y_pred[indices]), window)

def get_moving_err_mean(y_true,y_pred,window=30):
    """
    takes the average of the error over a window left and right for each point, no padding at end points
    """
    indices=np.argsort(y_true)
    return y_true[indices], moving_average(y_true[indices]-y_pred[indices], window)

def get_dyn_err(y_true,y_pred, bins=10,use_quantile=False,min_count=10):
    df=pd.DataFrame(zip(y_true,y_pred), columns=['y_true','y_pred']).dropna(axis=0, how='any')
    if use_quantile:  bins=np.unique(np.quantile(df['y_true'].values,np.linspace(0,1, num=bins)))
    df['true_bin'] = pd.cut(df['y_true'],bins=bins)
    df['err']=df['y_true']-df['y_pred']
    m=[] ;err_mean=[];err_mean_neg=[];err_mean_pos=[];err_std=[];bin_counts = []
    for k,v in df.groupby('true_bin'):
        if len(v)< 1:
            continue
        bin_counts.append(len(v))
        #m.append(v['y_true'].median())
        m.append(v['y_true'].min()+0.5*(v['y_true'].max() - v['y_true'].min()))
        #m.append( 0.5( v['y_true'].max() - v['y_true'].min() ))
        err_mean_pos.append(v.loc[df.err >0 ,'err'].mean())
        err_mean_neg.append(v.loc[df.err < 0 ,'err'].mean())
        err_mean.append(v['err'].mean())
        err_std.append(v['err'].std())
    return np.array(m), moving_average(np.array(err_mean),np.minimum(4,int(len(err_mean)/2+1/2))),np.array(err_mean_neg),np.array(err_mean_pos), np.array(err_std)

def plotly_reg_model_with_error(y_pred, y_true,title='', extend_axis_lim=0, b=None, e=None ,metrics='all',alpha=0.35, mask_scatter=False,bins=10,prop_cliffs=None, leave_grp_out=None, bin_window_average=True,fig_size=(600,600),smiles=None ,apply_tick_transformation=False, results_dict=None ):
    """
    Scatterplot with moving average error and distinction between different points
    """
    #metrics="RMSE = %.5f , R2 = %.2f " % (np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred))
    mask=np.isnan(y_true) | np.isnan(y_pred.astype(float))
    y_true=y_true[~mask]
    y_pred=y_pred[~mask]
    smiles=smiles[~mask]
    assert len(y_true) > 0, 'No data'
    mae=mean_absolute_error(y_true, y_pred)
    if metrics=='R2':
        metrics= "MAE = %.3f , R^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), r2_score(y_true, y_pred))
    elif metrics== 'P_Corr':
        metrics= "MAE = %.3f , P_Corr = %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0])
    else:
        metrics= "MAE = %.3f , P_Corr = %.2f, R^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0], r2_score(y_true, y_pred))
    if not b:
        b=min(y_true.min(), y_pred.min())-extend_axis_lim
    if not e:
        e=max(y_true.max(), y_pred.max()) +extend_axis_lim
    line_X = np.linspace(b, e)
    lw = 1
    fig = go.Figure()
    if not mask_scatter:
        stratified_mask=np.zeros(len(y_true))>0
        leg_labels=[]
        if prop_cliffs is not None:
            fig.add_trace(go.Scatter(x=y_true[prop_cliffs], y=y_pred[prop_cliffs],
                        mode='markers',
                        name='prop.-cliffs',
                        marker_symbol='cross',
                        hovertemplate =
                            '<b>SMILES</b>: %{text}'+
                            '<br><b>True value </b>: %{x:.2f}<br>'+
                            '<b>Predicted value</b>: %{y:.2f}',
                        text = smiles[prop_cliffs],
                        marker=dict(color=f'rgba(139, 0, 139,{alpha})',line_width=0.5, size=5)))
            stratified_mask[prop_cliffs]=True
        if leave_grp_out is not None:
            fig.add_trace(go.Scatter(x=y_true[leave_grp_out], y=y_pred[leave_grp_out],
                        mode='markers',
                        name='leave-group-out',
                        marker_symbol='diamond',
                        hovertemplate =
                            '<b>smiles</b>: %{text}'+
                            '<br><b>True value </b>: %{x:.2f}<br>'+
                            '<b>Predicted value</b>: %{y:.2f}',
                        text = smiles[leave_grp_out],
                        marker=dict(color=f'rgba(0, 100, 0,{alpha})',line_width=0.5, size=5)))
            stratified_mask[leave_grp_out]=True
        if len(y_true[~stratified_mask])>0:
            fig.add_trace(go.Scatter(x=y_true[~stratified_mask], y=y_pred[~stratified_mask],
                        mode='markers',
                        name='stratified',
                        marker_symbol='circle',
                        hovertemplate =
                            '<b>smiles</b>: %{text}'+
                            '<br><b>True value </b>: %{x:.2f}<br>'+
                            '<b>Predicted value</b>: %{y:.2f}',
                        text = smiles[~stratified_mask],
                        marker=dict(color=f'rgba(255, 0, 0,{alpha})',line_width=0.5, size=4)))
            
           
    fig.add_trace(go.Scatter(x=line_X, y=line_X, 
            mode='lines',
            showlegend=False,
            line=dict(color='navy', width=1, dash='dash')))
    
    if bin_window_average:
        m, err_mean,err_mean_neg,err_mean_pos, err_std=get_dyn_err(y_true,y_pred, bins=bins)
    else:
        m, err_mean = get_moving_err_mean(y_true,y_pred, window=np.minimum(bins,int(len(y_true)/2+1/2)))
    err=err_mean 
    fig.add_trace(go.Scatter(x=m, y=m, 
            mode='lines',
            name='Ideal',
            line=dict(color='rgba(0, 0, 255,0.9)', width=lw, dash='dash')))
    fig.add_trace(go.Scatter(
            x=m,
            y= m - 1*err,
            name='Moving average error',
            showlegend=True,
            #fill='tonexty', # fill area between trace0 and trace1
            mode='lines',line_color=f'rgba(0, 0, 255,0.6)', 
            #fillcolor=f'rgba(0, 0, 255,0.3)'
                )
            )
    #fig.add_trace(go.Scatter(x=m, y=m - 1*err, 
    #        mode='lines',
    #        showlegend=False,
    #        line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash')))
    #fig.add_trace(go.Scatter(
    #        x=m,
    #        y= m - 2*err,
    #        name='2 x Cent. Mov. MAE',
    #        showlegend=True,
    #        fill='tonexty',
    #        mode='lines',line_color=f'rgba(0, 0, 255,0.25)', fillcolor=f'rgba(0, 0, 255,0.15)'))
        
    fig.update_layout(title=f'Scatterplot with centered moving average of the error<br><sup>{title}</sup> <br><sup>{metrics}</sup>',
                    title_font_size=18,
                   xaxis_title="True value",
                   yaxis_title="Predicted value",
                   font=dict(size=18),height=fig_size[1], width=fig_size[0],
                   legend=dict(
                        yanchor="bottom",
                        y=0.005,
                        xanchor="right",
                        x=0.995,
                        traceorder="normal",
                        bgcolor = 'rgba(255,255,255,0.5)',
                        font=dict(
                            size=11,
                            color="black"
                        ),
                    ),
                    hoverlabel=dict(
                        font_size=10,
                    )
                )
    if apply_tick_transformation:
        ticks=np.linspace(b, e,8)
        transformed_values=None
        if title.startswith('logit_'):
            transformed_values= np.round(1/(1+np.exp(-ticks)),3)
            label_prefix='1/(1+exp(-'
            label_end='))'
        elif title.startswith('log10_'):
            transformed_values= np.round(10**ticks,3)
            label_prefix='10**('
            label_end=')'
        if transformed_values is not None:
            fig.update_layout(
                xaxis_title=f'{label_prefix}True value{label_end}',
                yaxis_title=f'{label_prefix}Predicted value{label_end}',
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = ticks,
                    ticktext = transformed_values
                ),
                yaxis = dict(
                    tickmode = 'array',
                    tickvals = ticks,
                    ticktext = transformed_values
                )
            )

    return fig,metrics


def plotly_reg_model(y_pred, y_true,title='', extend_axis_lim=0, b=None, e=None ,metrics='all',fig_size=(600,600),smiles=None,
                     apply_tick_transformation=False, results_dict=None  ):
    """
    scatterplot with MAE folds
    """
    #metrics="RMSE = %.5f , R2 = %.2f " % (np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred))
    mask=np.isnan(y_true) | np.isnan(y_pred.astype(float))
    y_true=y_true[~mask]
    y_pred=y_pred[~mask]
    assert len(y_true) > 0, 'No data'
    mae=mean_absolute_error(y_true, y_pred)
    err=np.abs(y_true-y_pred)
    fold1=round(100*sum(err<= 1*mae)/len(err),1)
    fold2=round(100*sum(err<= 2*mae)/len(err),1)
    if results_dict is not None:
        results_dict[title][f' MAE']=mean_absolute_error(y_true, y_pred)
        results_dict[title][f' P_Corr']=pearsonr(y_true, y_pred)[0]
        results_dict[title][f' R2']=r2_score(y_true, y_pred)
    
    if metrics=='R2':
        metrics= "MAE = %.3f , R^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), r2_score(y_true, y_pred))
    elif metrics== 'P_Corr':
        metrics= "MAE = %.3f , P_Corr = %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0])
    else:
        metrics= "MAE = %.3f , P_Corr = %.2f, R^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0], r2_score(y_true, y_pred))
    
    
    if not b:
        b=min(y_true.min(), y_pred.min())-extend_axis_lim
    if not e:
        e=max(y_true.max(), y_pred.max()) +extend_axis_lim
    line_X = np.linspace(b, e)
    lw = 1
    fig = go.Figure()
    if smiles is not None:
        fig.add_trace(go.Scatter(x=y_true, y=y_pred,
                    mode='markers',
                    name='samples',             
                    showlegend=False,
                    marker_symbol='circle',
                    hovertemplate =
                        '<b>smiles</b>: %{text}'+
                        '<br><b>True value </b>: %{x:.2f}<br>'+
                        '<b>Predicted value</b>: %{y:.2f}',
                    text = smiles,
                    marker=dict(color=f'rgba(255, 0, 0,0.5)',line_width=0.5, size=5)))
            
    else:
        fig.add_trace(go.Scatter(x=y_true, y=y_pred,
                    mode='markers',
                    name='stratified',             
                    showlegend=False,
                    marker_symbol='circle',
                    marker=dict(color=f'rgba(255, 0, 0,0.5)',line_width=0.5, size=5)))
                    
    fig.add_trace(go.Scatter(x=line_X, y=line_X+2*mae, 
            mode='lines',
            showlegend=False,
            line_color=f'rgba(0, 0, 255,0.3)'))
    fig.add_trace(go.Scatter(x=line_X, y=line_X+1*mae, 
            mode='lines',
            showlegend=False,
            fill='tonexty', # fill area between trace0 and trace1
            line_color=f'rgba(0, 0, 255,0.4)', fillcolor=f'rgba(0, 0, 255,0.1)'))  
    fig.add_trace(go.Scatter(x=line_X, y=line_X, 
            mode='lines',
            showlegend=False,
            fill='tonexty', # fill area between trace0 and trace1
            line_color=f'rgba(0, 0, 255,0.5)', fillcolor=f'rgba(0, 0, 255,0.25)'))  
    fig.add_trace(go.Scatter(x=line_X, y=line_X-1*mae, 
            mode='lines',
            name=f'{fold1}% w. 1 fold MAE',
            fill='tonexty', # fill area between trace0 and trace1
            line_color=f'rgba(0, 0, 255,0.4)', fillcolor=f'rgba(0, 0, 255,0.25)'))  
    fig.add_trace(go.Scatter(x=line_X, y=line_X-2*mae, 
            mode='lines',
            name=f'{fold2}% w. 2 folds MAE',
            fill='tonexty', # fill area between trace0 and trace1
            line_color=f'rgba(0, 0, 255,0.3)', fillcolor=f'rgba(0, 0, 255,0.1)'))  
    fig.update_layout(title=f'Scatterplot with MAE folds<br><sup>{title}</sup> <br><sup>{metrics}</sup>',
                    title_font_size=18,
                   xaxis_title="True value",
                   yaxis_title="Predicted value",
                   font=dict(size=18),height=fig_size[1], width=fig_size[0],
                   legend=dict(
                        yanchor="bottom",
                        y=0.005,
                        xanchor="right",
                        x=0.995,
                        traceorder="normal",
                        bgcolor = 'rgba(255,255,255,0.5)',
                        font=dict(
                            size=11,
                            color="black"
                        ),
                    ),
                    hoverlabel=dict(
                        font_size=10,
                    )
                )
    
    if apply_tick_transformation:
        ticks=np.linspace(b, e,8)
        transformed_values=None
        if title.startswith('logit_'):
            transformed_values= np.round(1/(1+np.exp(-ticks)),3)
            label_prefix='1/(1+exp(-'
            label_end='))'
        elif title.startswith('log10_'):
            transformed_values= np.round(10**ticks,3)
            label_prefix='10**('
            label_end=')'
        if transformed_values is not None:
            fig.update_layout(
                xaxis_title=f'{label_prefix}True value{label_end}',
                yaxis_title=f'{label_prefix}Predicted value{label_end}',
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = ticks,
                    ticktext = transformed_values
                ),
                yaxis = dict(
                    tickmode = 'array',
                    tickvals = ticks,
                    ticktext = transformed_values
                )
            )

    return fig,metrics

def plotly_reg_model_with_cutoff(y_pred, y_true,title='', extend_axis_lim=0, b=None, e=None ,metrics='all',fig_size=(600,600),cutoff=0.5, good_class= '>',smiles=None,apply_tick_transformation=False, results_dict=None ):
    """
    scatterplot with classification results based on given cutoff
    """
    #metrics="RMSE = %.5f , R2 = %.2f " % (np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred))
    mask=np.isnan(y_true) | np.isnan(y_pred.astype(float))
    y_true=y_true[~mask]
    y_pred=y_pred[~mask]
    assert len(y_true) > 0, 'No data'
    mae=mean_absolute_error(y_true, y_pred)
    err=np.abs(y_true-y_pred)
    if metrics=='R2':
        metrics= "MAE = %.3f , R^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), r2_score(y_true, y_pred))
    elif metrics== 'P_Corr':
        metrics= "MAE = %.3f , P_Corr = %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0])
    else:
        metrics= "MAE = %.3f , P_Corr = %.2f, R^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0], r2_score(y_true, y_pred))
    if not b:
        b=min(y_true.min(), y_pred.min())-extend_axis_lim
    if not e:
        e=max(y_true.max(), y_pred.max()) +extend_axis_lim
    line_X = np.linspace(b, e)
    lw = 1
    
    if good_class =='<':
        t_label=(y_true < cutoff)
        p_label=(y_pred < cutoff)
    else:
        t_label=(y_true > cutoff)
        p_label=(y_pred > cutoff)
    
    TP= ((p_label==1) & (t_label==1))
    FP= ((p_label==1) & (t_label==0))
    FN= ((p_label==0) & (t_label==1))
    TN= ((p_label==0) & (t_label==0))
    if results_dict is not None:
        results_dict[title][f' True Positive cutoff={cutoff}']=sum(TP)
        results_dict[title][f' False Positive cutoff={cutoff}']=sum(FP)
        results_dict[title][f' False Negative cutoff={cutoff}']=sum(FN)
        results_dict[title][f' True Negative cutoff={cutoff}']=sum(TN)
    
    N=y_true.shape[0]
    tp=round(100*sum(TP)/N,1)
    tn=round(100*sum(TN)/N,1)
    fp=round(100*sum(FP)/N,1)
    fn=round(100*sum(FN)/N,1)
    if results_dict is not None:
        results_dict[title][f' True Positive (%) cutoff={cutoff}']=tp
        results_dict[title][f' False Positive (%) cutoff={cutoff}']=fp
        results_dict[title][f' False Negative (%) cutoff={cutoff}']=fn
        results_dict[title][f' True Negative (%) cutoff={cutoff}']=tn
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true[TP], y=y_pred[TP],
                mode='markers',             
                name=f'True Pos.: {sum(TP)} ({tp}%)',
                marker_symbol='circle',
                hovertemplate =
                    '<b>smiles</b>: %{text}'+
                    '<br><b>True value </b>: %{x:.2f}<br>'+
                    '<b>Predicted value</b>: %{y:.2f}',
                text = smiles[TP],
                marker=dict(color=f'rgba(0, 100, 0,0.5)',line_width=0.5, size=5)))
    fig.add_trace(go.Scatter(x=y_true[TN], y=y_pred[TN],
                mode='markers',            
                name=f'True Neg.: {sum(TN)} ({tn}%)',
                marker_symbol='circle',
                hovertemplate =
                    '<b>smiles</b>: %{text}'+
                    '<br><b>True value </b>: %{x:.2f}<br>'+
                    '<b>Predicted value</b>: %{y:.2f}',
                text = smiles[TN],
                marker=dict(color=f'rgba(255, 0, 0,0.5)',line_width=0.5, size=5)))
    fig.add_trace(go.Scatter(x=y_true[FP], y=y_pred[FP],
                mode='markers',            
                name=f'False Pos.: {sum(FP)} ({fp}%)',
                marker_symbol='circle',
                hovertemplate =
                    '<b>smiles</b>: %{text}'+
                    '<br><b>True value </b>: %{x:.2f}<br>'+
                    '<b>Predicted value</b>: %{y:.2f}',
                text = smiles[FP],
                marker=dict(color=f'rgba(0, 191, 255,0.5)',line_width=0.5, size=5)))
    fig.add_trace(go.Scatter(x=y_true[FN], y=y_pred[FN],
                mode='markers',             
                name=f'False Neg.: {sum(FN)} ({fn}%)',
                marker_symbol='circle',
                hovertemplate =
                    '<b>smiles</b>: %{text}'+
                    '<br><b>True value </b>: %{x:.2f}<br>'+
                    '<b>Predicted value</b>: %{y:.2f}',
                text = smiles[FN],
                marker=dict(color=f'rgba(255, 213, 0,0.5)',line_width=0.5, size=5)))
    annot_pos="top right"
    yp_max=np.max(y_true)
    yp_min=np.min(y_true)
    if cutoff>yp_min+(yp_max-yp_min)/2:
        annot_pos="top left"
        
    fig.add_vline(x=cutoff,line=dict(color=f'rgba(112,128,144,0.8)', width=2.5, dash='dash'),
                 annotation_text=f'cutoff={cutoff}',
                 annotation_font_size=12,
                 annotation_position=annot_pos,
                 annotation_textangle=0)
    
    annot_pos="top left"
    yp_max=np.max(y_pred)
    yp_min=np.min(y_pred)
    if cutoff<yp_min+(yp_max-yp_min)/2:
        annot_pos="bottom right"
    fig.add_hline(y=cutoff,line=dict(color=f'rgba(112,128,144,0.8)', width=2.5, dash='dash'),
                 annotation_text=f'cutoff={cutoff}',
                 annotation_font_size=12,
                 annotation_position=annot_pos,
                 annotation_textangle=0)
    
     
    fig.update_layout(title=f'Classification scatterplot using cutoff={cutoff} <br><sup>{title}</sup> <br><sup>{metrics}</sup>', 
                   title_font_size=18,
                   xaxis_title="True value",
                   yaxis_title="Predicted value",
                   font=dict(size=18),height=fig_size[1], width=fig_size[0],
                   legend=dict(
                        yanchor="bottom",
                        y=0.005,
                        xanchor="right",
                        x=0.995,
                        traceorder="normal",
                        bgcolor = 'rgba(255,255,255,0.5)',
                        font=dict(
                            size=11,
                            color="black"
                        ),
                    ),
                    hoverlabel=dict(
                        font_size=10,
                    )
                )
    if apply_tick_transformation:
        ticks=np.linspace(b, e,8)
        transformed_values=None
        if title.startswith('logit_'):
            transformed_values= np.round(1/(1+np.exp(-ticks)),3)
            label_prefix='1/(1+exp(-'
            label_end='))'
        elif title.startswith('log10_'):
            transformed_values= np.round(10**ticks,3)
            label_prefix='10**('
            label_end=')'
        if transformed_values is not None:
            fig.update_layout(
                xaxis_title=f'{label_prefix}True value{label_end}',
                yaxis_title=f'{label_prefix}Predicted value{label_end}',
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = ticks,
                    ticktext = transformed_values
                ),
                yaxis = dict(
                    tickmode = 'array',
                    tickvals = ticks,
                    ticktext = transformed_values
                )
            )

    return fig,metrics+f', Positive {good_class} {cutoff}, True Pos.: {tp}%, True Neg.: {tn}%, False Pos.: {fp}%, False Neg.: {fn}%'



def _prepareMol(mol,kekulize):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc

def moltosvg(mol,molSize=(200,100),kekulize=True,drawer=None,**kwargs):
    mc = _prepareMol(mol,kekulize)
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc,**kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:',''))


def bokeh_reg_model(y_pred, y_true,title='', extend_axis_lim=0, b=None, e=None ,metrics='all',fig_size=(600,600),smiles=None,legend_pos="bottom_right",apply_tick_transformation=False ):
    """
    scatterplot with MAE folds and chemical structures
    """
    #metrics="RMSE = %.5f , R2 = %.2f " % (np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred))
    mask=np.isnan(y_true) | np.isnan(y_pred.astype(float))
    y_true=y_true[~mask]
    y_pred=y_pred[~mask]
    assert len(y_true) > 0, 'No data'
    mae=mean_absolute_error(y_true, y_pred)
    err=np.abs(y_true-y_pred)
    fold1=round(100*sum(err<= 1*mae)/len(err),1)
    fold2=round(100*sum(err<= 2*mae)/len(err),1)    
    if metrics=='R2':
        metrics= "MAE = %.3f , R^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), r2_score(y_true, y_pred))
    elif metrics== 'P_Corr':
        metrics= "MAE = %.3f , P_Corr = %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0])
    else:
        metrics= "MAE = %.3f , P_Corr = %.2f, R^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0], r2_score(y_true, y_pred))
    if not b:
        b=min(y_true.min(), y_pred.min())-extend_axis_lim
    if not e:
        e=max(y_true.max(), y_pred.max()) +extend_axis_lim   
    line_X = np.linspace(b, e)
    lw = 1
    
    df = pd.DataFrame(smiles.tolist(),columns =['SMILES'])
    PandasTools.AddMoleculeColumnToFrame(df,smilesCol='SMILES')
    svgs = np.array([moltosvg(m).data for m in df.ROMol])
    #svgs = np.array([moltosvg(m) for m in df.ROMol])
    ChangeMoleculeRendering(renderer='PNG')

    
    
    fig = figure(width=fig_size[0], height=fig_size[1], tools=['reset,box_zoom,wheel_zoom,zoom_in,zoom_out,pan,save'])
    
                             
    fig.add_layout(Title(text=f'{metrics}', text_font_style="italic"), 'above')
    fig.add_layout(Title(text=f'{title}', text_font_style="italic"), 'above')
    fig.add_layout(Title(text=f'Scatterplot with MAE folds', text_font_size="16pt"), 'above')

    source = ColumnDataSource(data=dict(x=y_true, y=y_pred, desc= smiles,svgs=svgs))
    cr=fig.scatter('x', 'y', size=7, source=source, name=f'scatter', line_color='black', fill_color=(255, 0, 0), fill_alpha=0.5, line_width=0.5)

    #bokeh 3.0
    hover = HoverTool(tooltips="""
        <div>
            <div> @svgs{safe}
            </div>
            <div>
                <span style="font-size: 8px;">SMILES: @desc</span>
            </div>
            <div>
            <span style="font-size: 10px; font-weight: bold;">Predicted value: @y</span>
            </div>
            <div>
            <span style="font-size: 10px; font-weight: bold;">True value: @x </span>
            </div>
        </div>
        """
    ,renderers=[cr]) #names=['scatter'])
    
    fig.add_tools(hover)
    
    fig.varea(x=line_X,
        y1=line_X+2*mae,
        y2=line_X-2*mae, fill_color=(0, 0, 255), fill_alpha=0.1,legend_label=f'{fold2}% w. 2 folds MAE')
    fig.varea(x=line_X,
        y1=line_X+mae,
        y2=line_X-mae, fill_color=(0, 0, 255), fill_alpha=0.25, legend_label=f'{fold1}% w. 1 fold MAE')
    
    fig.line(line_X, line_X+mae, line_width=2, line_color=(0, 0, 255), line_alpha=0.3)
    fig.line(line_X, line_X+2*mae, line_width=2, line_color=(0, 0, 255), line_alpha=0.3)
    fig.line(line_X, line_X, line_width=2, line_color=(0, 0, 255), line_alpha=0.3)
    fig.line(line_X, line_X-mae, line_width=2, line_color=(0, 0, 255), line_alpha=0.3)
    fig.line(line_X, line_X-2*mae, line_width=2, line_color=(0, 0, 255), line_alpha=0.3)

    
    fig.background_fill_color = "lightgray"
    fig.background_fill_alpha = 0.3
    fig.legend.border_line_width = 2
    fig.legend.border_line_color = "black"
    fig.legend.background_fill_color = "lightgray"
    fig.legend.background_fill_alpha = 0.4                     
    fig.legend.label_text_font_size = "10px"
    
    fig.xaxis.axis_label ="True value"
    fig.yaxis.axis_label ="Predicted value"
    
    if apply_tick_transformation:
        ticks=np.linspace(b, e,8)
        transformed_values=None
        if title.startswith('logit_'):
            transformed_values= { tick:str(np.round(1/(1+np.exp(-tick)),3)) for tick in ticks}
            label_prefix='1/(1+exp(-'
            label_end='))'
        elif title.startswith('log10_'):
            transformed_values= { tick:str(np.round(10**tick,3)) for tick in ticks}
            label_prefix='10**('
            label_end=')'
        if transformed_values is not None:
            fig.xaxis.axis_label=f'{label_prefix}True value{label_end}'
            fig.yaxis.axis_label=f'{label_prefix}Predicted value{label_end}'
            fig.xaxis.ticker = ticks
            fig.xaxis.major_label_overrides = transformed_values
            fig.yaxis.ticker = ticks
            fig.yaxis.major_label_overrides = transformed_values
    
    fig.legend.location = legend_pos

    return fig


def bokeh_reg_model_with_cutoff(y_pred, y_true,title='', extend_axis_lim=0, b=None, e=None ,metrics='all',fig_size=(600,600),cutoff=0.5, good_class= '>',smiles=None,legend_pos="bottom_right",apply_tick_transformation=False):
    """
    scatterplot with classification results and chemical structures
    """
    #metrics="RMSE = %.5f , R2 = %.2f " % (np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred))
    mask=np.isnan(y_true) | np.isnan(y_pred.astype(float))
    y_true=y_true[~mask]
    y_pred=y_pred[~mask]
    assert len(y_true) > 0, 'No data'
    mae=mean_absolute_error(y_true, y_pred)
    err=np.abs(y_true-y_pred)
    if metrics=='R2':
        metrics= "MAE = %.3f , R^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), r2_score(y_true, y_pred))
    elif metrics== 'P_Corr':
        metrics= "MAE = %.3f , P_Corr = %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0])
    else:
        metrics= "MAE = %.3f , P_Corr = %.2f, R^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0], r2_score(y_true, y_pred))
    if not b:
        b=min(y_true.min(), y_pred.min())-extend_axis_lim
    if not e:
        e=max(y_true.max(), y_pred.max()) +extend_axis_lim
    line_X = np.linspace(b, e)
    lw = 1
    
        
    if good_class =='<':
        t_label=(y_true < cutoff)
        p_label=(y_pred < cutoff)
    else:
        t_label=(y_true > cutoff)
        p_label=(y_pred > cutoff)
    
    TP= ((p_label==1) & (t_label==1))
    FP= ((p_label==1) & (t_label==0))
    FN= ((p_label==0) & (t_label==1))
    TN= ((p_label==0) & (t_label==0))
    
    N=y_true.shape[0]
    tp=round(100*sum(TP)/N,1)
    tn=round(100*sum(TN)/N,1)
    fp=round(100*sum(FP)/N,1)
    fn=round(100*sum(FN)/N,1)  
    
    df = pd.DataFrame(smiles.tolist(),columns =['SMILES'])
    PandasTools.AddMoleculeColumnToFrame(df,smilesCol='SMILES')
    svgs = np.array([moltosvg(m).data for m in df.ROMol])
    
    ChangeMoleculeRendering(renderer='PNG')

    hover = HoverTool(tooltips="""
        <div>
            <div>@svgs{safe}
            </div>
            <div>
                <span style="font-size: 10px;">SMILES: @desc</span>
            </div>
            <div>
            <span style="font-size: 10px; font-weight: bold;">Predicted value: @y</span>
            </div>
            <div>
            <span style="font-size: 10px; font-weight: bold;">True value: @x</span>
            </div>
        </div>
        """
    )
    
    fig = figure(plot_width=fig_size[0], plot_height=fig_size[1], tools=['reset,box_zoom,wheel_zoom,zoom_in,zoom_out,pan,save',hover])
                             
    fig.add_layout(Title(text=f'{metrics}', text_font_style="italic"), 'above')
    fig.add_layout(Title(text=f'{title}', text_font_style="italic"), 'above')
    fig.add_layout(Title(text=f'Classification Scatterplot w. cutoff= {cutoff}', text_font_size="16pt"), 'above')
               
    source = ColumnDataSource(data=dict(x=y_true[TP], y=y_pred[TP], desc= smiles[TP], 
                                        svgs=svgs[TP]))
    fig.circle('x', 'y', size=5, source=source, line_color='black', fill_color=(0, 100, 0), fill_alpha=0.5, line_width=0.5, legend_label=f'True Pos.: {sum(TP)}({tp}%)');
    source = ColumnDataSource(data=dict(x=y_true[TN], y=y_pred[TN], desc= smiles[TN], 
                                        svgs=svgs[TN]))
    fig.circle('x', 'y', size=5, source=source, line_color='black', fill_color=(255, 0, 0), fill_alpha=0.5, line_width=0.5, legend_label=f'True Neg.: {sum(TN)}({tn}%)');
    source = ColumnDataSource(data=dict(x=y_true[FP], y=y_pred[FP], desc= smiles[FP], 
                                        svgs=svgs[FP]))
    fig.circle('x', 'y', size=5, source=source, line_color='black', fill_color=(0, 191, 255), fill_alpha=0.5, line_width=0.5, legend_label=f'False Pos.: {sum(FP)}({fp}%)');
    source = ColumnDataSource(data=dict(x=y_true[FN], y=y_pred[FN], desc= smiles[FN], 
                                        svgs=svgs[FN]))
    fig.circle('x', 'y', size=5, source=source, line_color='black', fill_color=(255, 213, 0), fill_alpha=0.5, line_width=0.5, legend_label=f'False Neg.: {sum(FN)}({fn}%)');

    
    vcutoff = Span(location=cutoff,
                    dimension='height', line_color=(112,128,144,0.8),
                    line_dash='dashed', line_width=2.5)
    fig.add_layout(vcutoff)
    hcutoff = Span(location=cutoff,
                    dimension='width', line_color=(112,128,144,0.8),
                    line_dash='dashed', line_width=2.5)
    fig.add_layout(hcutoff)
    fig.background_fill_color = "lightgray"
    fig.background_fill_alpha = 0.3
    fig.legend.border_line_width = 2
    fig.legend.border_line_color = "black"
    fig.legend.background_fill_color = "lightgray"
    fig.legend.background_fill_alpha = 0.4                     
    fig.legend.label_text_font_size = "10px"
    
    fig.xaxis.axis_label ="True value"
    fig.yaxis.axis_label ="Predicted value"
    
    if apply_tick_transformation:
        ticks=np.linspace(b, e,8)
        transformed_values=None
        if title.startswith('logit_'):
            transformed_values= { tick:str(np.round(1/(1+np.exp(-tick)),3)) for tick in ticks}
            label_prefix='1/(1+exp(-'
            label_end='))'
        elif title.startswith('log10_'):
            transformed_values= { tick:str(np.round(10**tick,3)) for tick in ticks}
            label_prefix='10**('
            label_end=')'
        if transformed_values is not None:
            fig.xaxis.axis_label=f'{label_prefix}True value{label_end}'
            fig.yaxis.axis_label=f'{label_prefix}Predicted value{label_end}'
            fig.xaxis.ticker = ticks
            fig.xaxis.major_label_overrides = transformed_values
            fig.yaxis.ticker = ticks
            fig.yaxis.major_label_overrides = transformed_values
    
    fig.legend.location = legend_pos

    return fig

###############################################################################################

def var_trapezoid_rule(x,y):
    assert len(x)==len(y), 'provided length are not equal'
    area=0
    for i in range(len(x)-1):
        area=area+((x[i+1]-x[i])*(y[i+1]+y[i])/2)
    return area

def average_precision(r,p):
    assert len(r)==len(p), 'provided length are not equal'
    ap=0
    for i in range(len(r)-1):
        ap=ap+((r[i+1]-r[i])*(p[i+1]))
    return ap

def plotly_acc_pre_for_reg(y_true,y_pred,title='' ,b=None, e=None , good_class= '>',colors=None,line_width=2, fig_size=(600,600), results_dict=None):
    """
    Plots active ratio, recall, precision and accuracy for different thresholds
    """
    mask=np.isnan(y_true) | np.isnan(y_pred.astype(float))
    y_true=y_true[~mask]
    y_pred=y_pred[~mask]
    assert len(y_true) > 0, 'No data'
    ACC, PRE ,REC, CLASS_ratio,used_cutoffs =[], [], [],[],[]
    if not b:
        b=min(y_pred)
    if not e:
        e=max(y_pred)
    cutoffs=np.linspace(b,e, num=50)
    for cutoff in cutoffs:
        #cutoff= 5
        if good_class =='<':
            t_label=(y_true < cutoff)
            p_label=(y_pred < cutoff)
        else:
            t_label=(y_true > cutoff)
            p_label=(y_pred > cutoff)
        acc=100*sum(t_label==p_label)/len(t_label)
        TP= sum((p_label==1) & (t_label==1))
        FP= sum((p_label==1) & (t_label==0))
        FN= sum((p_label==0) & (t_label==1))
        #precision = TP/(TP + FP)
        #What proportion of positive identifications was actually correct?
        if (TP + FP) > 0 and (TP + FN)>0:
            pre=100*TP/(TP + FP)#sum((p_label==1) & (t_label==p_label))/sum(p_label==1)
            #recall = TP/(TP + FN)
            #What proportion of actual positives was identified correctly?
            rec=100*TP/(TP + FN) #sum((p_label==1) & (t_label==p_label))/sum(t_label==1)
            ACC.append(acc)
            PRE.append(pre)
            REC.append(rec)
            CLASS_ratio.append(100*sum(t_label)/len(y_true))
            used_cutoffs.append(cutoff)
        
    if results_dict is not None:
        results_dict[title][f' varying threshold. positive {good_class} threshold']= {'threshold':used_cutoffs,
                                                                             'Accuracy':ACC,
                                                                             'Precision':PRE,
                                                                             'Recall':REC,
                                                                             'Positive ratio':CLASS_ratio}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=used_cutoffs, y=ACC,
                        mode='lines',
                        name=f'Accuracy',
                        line=dict(color=colors[0%len(colors)], width=line_width)))
    fig.add_trace(go.Scatter(x=used_cutoffs, y=PRE,
                        mode='lines',
                        name=f'Precision',
                        line=dict(color=colors[1%len(colors)], width=line_width)))
    fig.add_trace(go.Scatter(x=used_cutoffs, y=REC,
                        mode='lines',
                        name=f'Recall',
                        line=dict(color=colors[2%len(colors)], width=line_width)))
    fig.add_trace(go.Scatter(x=used_cutoffs, y=CLASS_ratio,
                        mode='lines',
                        name=f'Positive ratio',
                        line=dict(color=colors[3%len(colors)], width=line_width)))
    xanch="left"
    xval=0.005
    if good_class =='<':
        xanch="right"
        xval=0.995
        
    fig.update_layout(title=f'Threshold variation <br><sup>Positive {good_class} threshold</sup> <br><sup>{title}</sup> ',
                    title_font_size=18,
                   xaxis_title=f'Threshold (Positive {good_class} threshold)',
                   yaxis_title="%",font=dict(size=18),height=fig_size[1], width=fig_size[0],
                    legend=dict(
                        yanchor="bottom",
                        y=0.005,
                        xanchor=xanch,
                        x=xval,
                        traceorder="normal",
                        bgcolor = 'rgba(255,255,255,0.5)',
                        font=dict(
                            size=11,
                            color="black"
                        ),
                    )
                )
   
    return fig


def plotly_enrichment(y_true,y_pred,title='',b=None, e=None , good_class= '>',colors=None,line_width=2, fig_size=(600,600), results_dict=None):
    """
    plots hit enrichment
    """
    mask=np.isnan(y_true) | np.isnan(y_pred.astype(float))
    y_true=y_true[~mask]
    y_pred=y_pred[~mask]
    assert len(y_true) > 0, 'No data'
    TPF, SF,PRE,REC =[], [], [], []
    if not b:
        b=min(y_pred)
    if not e:
        e=max(y_pred)
    cutoffs=np.linspace(b,e, num=50)
    N=len(y_true)
    for cutoff in cutoffs:
        #cutoff= 5
        if good_class =='<':
            p_label=(y_pred < cutoff)
        else:
            p_label=(y_pred > cutoff)
        
        TP= sum((p_label==1) & (y_true==1))
        FP= sum((p_label==1) & (y_true==0))
        FN= sum((p_label==0) & (y_true==1))
        #precision = TP/(TP + FP)
        #What proportion of positive identifications was actually correct?
        if TP + FP > 0 and TP + FN > 0:
            pre=TP/(TP + FP)
            rec=TP/(TP + FN) 
            PRE.append(pre)
            REC.append(rec)
        S= sum(p_label==1)
        TPF.append(TP/sum(y_true==1))
        SF.append(S/N)
               
    SF, TPF = (list(t) for t in zip(*sorted(zip(SF, TPF)))) 
    
    area=np.round(var_trapezoid_rule(SF,TPF),2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=SF, y=TPF,
                        mode='lines',
                        name=f'True Pos. Fraction: area={area}',
                        fill='tozeroy', # fill area between trace0 and trace1
                        fillcolor=f'rgba(112,128,144,0.3)',
                        line=dict(color=colors[0%len(colors)], width=line_width)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                    mode='lines',
                    showlegend=False,
                    line=dict(color=f'rgba(112,128,144,0.8)', width=2.5, dash='dash')))
    
    #fig.update_xaxes(type="log")
    fig.update_layout(title=f'Hit enrichment curve {title}',
                    title_font_size=18,
                   xaxis_title=f'Total Positive Fraction',
                   yaxis_title="True Positive Fraction",font=dict(size=18),height=fig_size[1], width=fig_size[0],
                   xaxis_range=[0.0,1.0],
                   yaxis_range=[0.0,1.05],
                    legend=dict(
                        yanchor="bottom",
                        y=0.005,
                        xanchor="right",
                        x=0.95,
                        traceorder="normal",
                        bgcolor = 'rgba(255,255,255,0.5)',
                        font=dict(
                            size=11,
                            color="black"
                        ),
                    )
                )

    fig_pr = go.Figure()
    if REC and PRE:
        REC, PRE = (list(t) for t in zip(*sorted(zip(REC, PRE))))
        pr_area=np.round(var_trapezoid_rule(REC,PRE),2)
        ap=np.round(average_precision(REC,PRE),2)
        fig_pr.add_trace(go.Scatter(x=REC, y=PRE,
                            mode='lines',
                            name=f'PR-curve: area={pr_area}, AP={ap}',
                            fill='tozeroy', # fill area between trace0 and trace1
                            fillcolor=f'rgba(112,128,144,0.3)',
                            line=dict(color=colors[0%len(colors)], width=line_width)))
        fig_pr.add_trace(go.Scatter(x=[0, 1], y=[1, 0],
                    mode='lines',
                    showlegend=False,
                    line=dict(color=f'rgba(112,128,144,0.8)', width=2.5, dash='dash')))
        fig_pr.update_layout(title=f'Precision-Recall curve {title}',
                        title_font_size=18,
                       xaxis_title=f'Recall (sensitivity)',
                       yaxis_title="Precision (PPV)",font=dict(size=18),height=fig_size[1], width=fig_size[0],
                       xaxis_range=[0.0,1.0],
                       yaxis_range=[0.0,1.05],
                        legend=dict(
                            yanchor="bottom",
                            y=0.005,
                            xanchor="left",
                            x=0.1,
                            traceorder="normal",
                            bgcolor = 'rgba(255,255,255,0.5)',
                            font=dict(
                                size=11,
                                color="black"
                            ),
                        )
                    )
    else:
        fig_pr.add_annotation(x=2, y=2,
            text="Empty figure, verify the cutoff such that the validation set contains both classes",
            showarrow=False,
            yshift=10)
        pr_area=0.0
        ap=0.0
        
    return fig,f'area={area}',fig_pr,f'(area={pr_area}, Average Precision={ap})'
    


##################################################################################################3
def df2category(df, pro, bins=None, quantile=[0.1] ,prefix='CAT', outname=None):
    if not bins:
        bins=df[pro].dropna().quantile(quantile).tolist()
        min_pro= -np.inf#df[pro].min()
        max_pro= df[pro].max()
        bins=[min_pro] + bins +[max_pro]
    bins=bins+[np.inf] # new class for nan
    labels_id=[i for i in range(len(bins) -1)]
    labels_id2name={i:f'{bins[i]} to {bins[i+1]}' for i in labels_id }
    labels_id2name[labels_id[-1]]='NaN'
    if not outname:
        outname= f'{prefix}_{pro}'
    df[outname] = pd.cut(df[pro], bins=bins,labels=labels_id)
    df[outname][df[outname].isna()]=labels_id[-1]
    print('Values count of the classes:')
    print(df[outname].value_counts())
    return labels_id2name
#df2category(df, pro='PPB_Cyprotex %Free (%)', bins=None)

def plotly_confusion_bars_from_categories(df_raw,pro1=None,pro2=None,
                                        bins1= range(1,11),color=None,#['red','yellow','green']
                  title=None,
                           leg_title='',
                           x_title=None,labelnames=None,
                            fig_size=(800,800), results_dict=None):
    '''
    Displays barplots where bins of continuous property one is set wrt to categories of property two
    
    Args:
         df_raw: dataframe containing both properties
         pro1: column name property 1
         pro2: column name property 2
        s bin1: bins of the first property
         title: title of the plot
         leg_title: legend title
         x_title: title of x-labels 
    '''
    df= df_raw.loc[(df_raw[pro1].notnull()) &(df_raw[pro2].notnull()),[pro1,pro2]].copy()
    bins1 = [min(bins1)-1000] + list(bins1) + [max(bins1)+1000]
    labels1=[f'<{np.round(bins1[1],2)}']+ [f"{np.round(bins1[i],2)}-{np.round(bins1[i+1],2)}" for i in range(1,len(bins1)-2)]+ [f'>{np.round(bins1[-2],2)}']
    #print(bins1,labels1)
    try:
        df['pro1_bin'] = pd.cut(df[pro1],bins=bins1,labels=labels1)
    except ValueError:
        df['pro1_bin'] = pd.cut(df[pro1],bins=bins1,labels=labels1,ordered=False)
    
    lims=1000
    labels2=list(labelnames.keys())
    nr_blues=len(labels1)//2
    nr_reds=len(labels1)-1*(nr_blues)
    #blue_colors = np.array([[0.,0.2,1,1]]*nr_blues)
    #blue_colors[:,3]=np.linspace(0.8, 0.1, nr_blues)
    green_colors = np.array([[0.,100.0/255,0.,1]]*nr_blues);    green_colors[:,3]=np.linspace(0.1, 0.8, nr_blues)
    red_colors = np.array([[0.9,0.0,0,1]]*nr_reds)
    red_colors[:,3]=np.linspace(0.8, 0.1, nr_reds)
    rgba_colors=np.vstack([red_colors,green_colors])
    
    if not color or len(color) !=len(labels1):
        color=rgba_colors
    colors2= {l:c for l,c in zip(labels1,color)}
    ##############
    ##############
    res = {}
    bin_counts = []
    for k,v in df.groupby(pro2):
        if len(v) < 1:continue
        res[k]=v['pro1_bin'].value_counts(normalize=True).to_dict()
        bin_counts.append(len(v))
    dftm=pd.DataFrame.from_dict(res, orient='index').reindex(columns=labels1)#.sort_values('pro2_bin')
    #pd.CategoricalIndex(dftm.columns.values,   ordered=True,      categories=labels2)
    dftm=dftm.rename(index=labelnames)
    fig = go.Figure(
        data=[go.Bar(
            name=col,
            x=dftm.index,
            y=dftm[col],
            customdata= np.stack(([int(bin_counts[index]*y_val) for index,y_val in enumerate(dftm[col])], [ labels1[col_index] for i in dftm[col] ]),-1),
            hovertemplate='<b>Cnt</b>: %{customdata[0]:d} <br><b>Rel. Cnt</b>: %{y:.2f} <br><b>Prob. bin</b>: %{customdata[1]}<br><b>Class</b>: %{x} ',
            marker_color=f'rgba({colors2[col][0]},{colors2[col][1]},{colors2[col][2]},{colors2[col][3]})',
            text=[np.round(y_val,2) if y_val>0.05 else "" for y_val in dftm[col]],
            textposition='inside',
            textfont_size=10,
            textfont_color="black"
        ) for col_index,col in enumerate(dftm.columns)])
    # Change the bar mode
    fig.update_layout(barmode='stack')
    if not x_title: x_title=f"Bins of {pro1}"
    for i,t in enumerate(bin_counts):
        fig.add_annotation(x=labels2[i], y = 1.0,
                           text = str(t),
                           showarrow = False,
                           yshift = 8,
                           font=dict(family="Courier New, monospace",
                                     size=10,
                                     color="black"
                                    )
                          )
    fig.update_layout(title=f'Predicted probability barplot <br><sup>{title}</sup>',
                    title_font_size=18,height=fig_size[1], width=fig_size[0],
                   xaxis_title=x_title,
                   yaxis_title=f'Relative counts',
                    legend=dict(
                        title_text=leg_title,
                        traceorder="normal",
                        font=dict(
                            size=11,
                            color="black"
                            )
                        )
                     )
    return fig

    
def plotly_clf_f1(y_pred_proba, y_true,title='',colors=None,line_width=2,c=0, labelname=None,youden=None,youden_val='',verbose=True,fig_size=(800,800), results_dict=None ):
    """
    Plots precision, recall and optimal thresholds for classification
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:,c], pos_label=c)
    fscore = (2 * precision * recall) / (precision + recall)
    if results_dict is not None:
        results_dict[title][f'precision']=precision
        results_dict[title][f'recall']=recall
        results_dict[title][f'thresholds']=thresholds
        results_dict[title][f'fscore']=fscore
    ix = np.nanargmax(fscore)
    if verbose: print('Class %s: Best Threshold=%f, precision=%.3f, recall=%.3f and F-Score=%.3f (uses validation set!)' % 
              (labelname,thresholds[ix],precision[ix],recall[ix],fscore[ix]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precision[:-1],
                        mode='lines',
                        name=f'Precision {labelname}',
                        line=dict(color=colors[0%len(colors)], width=line_width)))
    fig.add_trace(go.Scatter(x=thresholds, y=recall[:-1],
                        mode='lines',
                        name=f'Recall {labelname}',
                        line=dict(color=colors[1%len(colors)], width=line_width)))
    fig.add_trace(go.Scatter(x=thresholds, y=fscore[:-1],
                        mode='lines',
                        name=f'F1-score {labelname}',
                        line=dict(color=colors[2%len(colors)], width=line_width)))

    fig.add_vline(x=0.5,line=dict(color='navy', width=line_width, dash='dash'))
    fig.add_vline(x=thresholds[ix],line=dict(color=colors[3], width=line_width, dash='dash'),
                 annotation_text=f'F1={np.round(fscore[ix],3)}',
                 annotation_font_size=11,
                 annotation_position="bottom right",
                 annotation_textangle=90)
    xticks=np.linspace(0,1,5)
    xticks = xticks[np.absolute(xticks-thresholds[ix])>0.05]
    xticks=np.append(xticks, thresholds[ix])
    if youden is not None:
        xticks = xticks[np.absolute(xticks-youden)>0.05]
        xticks=np.append(xticks, youden)
        #check if thresholds overlap
        if np.abs(youden-thresholds[ix])>0.05:
            fig.add_vline(x=youden,line=dict(color=colors[4], width=line_width, dash='dash'),
                 annotation_text=f'Youdens-J={np.round(youden_val,3)}',
                 annotation_font_size=11,
                 annotation_position="bottom right",
                 annotation_textangle=90)
        else:
            fig.add_vline(x=youden,line=dict(color=colors[4], width=line_width, dash='dash'),
                 annotation_text=f'Youdens-J={np.round(youden_val,3)}',
                 annotation_font_size=11,
                 annotation_position="bottom left",
                 annotation_textangle=90)
    
    fig.update_layout(title=f'Precision, Recall and F1-score curves <br><sup>Threshold tuning {title}</sup><br> ',
                    title_font_size=18,
                    title_y=0.935,
                    title_yanchor="bottom",
                   xaxis_title="Threshold",
                   yaxis_title="Precision/Recall/F1-score",
                   xaxis_range=[0.0,1.0],
                   yaxis_range=[0.0,1.05],font=dict(size=18),height=fig_size[1], width=fig_size[0],
                    xaxis = dict(
                        tickmode = 'array',
                        tickvals = np.sort(np.round(xticks,2)),
                        ticktext = np.sort(np.round(xticks,2))
                    ),
                   legend=dict(orientation="h",
                        yanchor="bottom",
                        y=1.001,
                        xanchor="left",
                        x=-0.12,
                        traceorder="normal",
                        bgcolor = 'rgba(255,255,255,0.5)',
                        font=dict(
                            size=11,
                            color="black"
                        ),
                    ))
    return thresholds[ix],fscore[ix],fig
      
    
    
def plotly_clf_prc(y_pred_proba, y_true,title='',colors=None,line_width=2,prop_labels=None,fig_size=(800,800), results_dict=None ):
    """
    plots precision-recall curves
    """
    Youden_thresholds=[]
    Youden_vals=[]
    
    fig = go.Figure()  
    area_list=[]
    ap_list=[]
    for c in range(y_pred_proba.shape[1]):
        precision, recall, thresholds =  precision_recall_curve(y_true, y_pred_proba[:,c], pos_label=c)
        recall=recall[:-1]
        precision=precision[:-1]
        precision=precision[::-1]
        recall=recall[::-1]
        area=np.round(var_trapezoid_rule(recall,precision),2)
        area_list.append(area)
        ap=np.round(average_precision(recall,precision),2)
        ap_list.append(ap)
        fig.add_trace(go.Scatter(x=recall, y=precision,
                        mode='lines',
                        name=f'PR-curve {prop_labels[c]}: area={area}, AP={ap}',
                        line=dict(color=colors[c%len(colors)], width=line_width)))
    
    mean_a=np.round(np.mean(area_list),3)
    mean_ap=np.round(np.mean(ap_list),3)
    fig.update_layout(title=f'Precision-Recall curve <br><sup>{title}</sup><br><sup>Mean area: {mean_a}, Mean Average Precision: {mean_ap}</sup>',
                    title_font_size=18,
                   xaxis_title=f'Recall (sensitivity)',
                   yaxis_title="Precision (PPV)",font=dict(size=18),height=fig_size[1], width=fig_size[0],
                   xaxis_range=[0.0,1.0],
                   yaxis_range=[0.0,1.05],
                    legend=dict(
                        yanchor="bottom",
                        y=0.005,
                        xanchor="left",
                        x=0.1,
                        traceorder="normal",
                        bgcolor = 'rgba(255,255,255,0.5)',
                        font=dict(
                            size=11,
                            color="black"
                        ),
                    )
                )
    
    return fig,f'Mean area: {mean_a}, Mean Average Precision: {mean_ap}'

def plotly_clf_calbc(y_pred_proba, y_true,title='',colors=None,line_width=2,prop_labels=None,fig_size=(800,800),prob_cutoffs=None, results_dict=None ):
    """
    plots calibration curves
    """
    
    fig = go.Figure()  
    ece_list=[]
    mce_list=[]
    if prob_cutoffs is None: prob_cutoffs=np.linspace(0.0, 1.0, num=6)
    labels=[f'[{np.round(prob_cutoffs[i-1],2)},{np.round(prob_cutoffs[i],2)})' for i in range(len(prob_cutoffs))[1:] ]
    labels[-1]=labels[-1].replace(')', ']', 1)
    text_pos=["top center","bottom center","bottom left","top left","bottom right","top right"]
    for c in range(y_pred_proba.shape[1]):
        df=pd.DataFrame.from_dict({'prob':y_pred_proba[:,c],'y_true':y_true})
        df['bins'] = pd.cut(df['prob'], bins=prob_cutoffs,include_lowest=True)
        acc_bin=[]
        prob_bin=[]
        len_bin=[]
        for k,v in df.groupby('bins'):
            if len(v)<1: continue
            acc_bin.append(sum(v['y_true']==c)/len(v))
            prob_bin.append(sum(v['prob'])/len(v))
            len_bin.append(len(v))

        ece=sum(np.abs(np.array(acc_bin)-np.array(prob_bin))*np.array(len_bin))/sum(len_bin)
        mce=np.max(np.abs(np.array(acc_bin)-np.array(prob_bin)))
        ece_list.append(ece)
        mce_list.append(mce)
        fig.add_trace(go.Scatter(x=prob_bin, y=acc_bin,
                        mode='lines+markers+text',
                        hovertemplate =
                        '<b>Group probability </b>: %{x:.2f}<br>'+
                        '<b>Observed ratio in group</b>: %{y:.2f}<br>'+
                        '<b>Nb group samples</b>: %{text}',
                        text = len_bin,
                        textposition=text_pos[c%len(text_pos)],
                        textfont=dict(
                            size=10,
                            color=colors[c%len(colors)]
                        ),
                        name=f'{prop_labels[c]}: ECE={np.round(ece,2)}, MCE={np.round(mce,2)}',
                        line=dict(color=colors[c%len(colors)], width=line_width)))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                mode='lines',
                showlegend=False,
                line=dict(color=f'rgba(112,128,144,0.8)', width=2.5, dash='dash')))
    ece_m=np.round(np.mean(ece_list),3)
    mce_m=np.round(np.mean(mce_list),3)
    fig.update_layout(title=f'Reliability diagram <br><sup>{title}</sup><br><sup>Mean Expected Calibration Error: {ece_m}, Mean Maximum Calibration Error: {mce_m}</sup>',
                    title_font_size=18,
                   xaxis_title=f'Probability Group',
                   yaxis_title="Observed ratio",font=dict(size=18),height=fig_size[1], width=fig_size[0],
                   xaxis_range=[-0.05,1.05],
                   yaxis_range=[-0.05,1.05],
                    xaxis = dict(
                        tickmode = 'array',
                        tickvals = np.array(prob_cutoffs[1:])-(1/(2*(len(prob_cutoffs)-1))),
                        ticktext = np.array(labels),
                        tickfont=dict(
                            size=10,
                        ),
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.995,
                        xanchor="left",
                        x=0.005,
                        traceorder="normal",
                        bgcolor = 'rgba(255,255,255,0.5)',
                        font=dict(
                            size=11,
                            color="black"
                        ),
                    )
                )
    
    return fig,f'Mean Expected Calibration Error: {ece_m}, Mean Maximum Calibration Error: {mce_m}'

def plotly_clf_auc(y_pred_proba, y_true,title='',colors=None,line_width=2,prop_labels=None,fig_size=(800,800), results_dict=None ):
    """
    plots roc-auc curves
    """
    Youden_thresholds=[]
    Youden_vals=[]
    
    fig = go.Figure()    
    for c in range(y_pred_proba.shape[1]):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:,c], pos_label=c)
        J = tpr - fpr
        ix = np.argmax(J)
        print('Class %s: Best Threshold=%f, Recall=%.3f (uses validation set!), Youdens J=%.3f (uses validation set!)' % (prop_labels[c],thresholds[ix], tpr[ix], J[ix]))
        Youden_thresholds.append(thresholds[ix])
        Youden_vals.append(J[ix])
        roc_auc = auc(fpr, tpr)
        if results_dict is not None:
            results_dict[title][f' AUC-ROC class {prop_labels[c]} vs Rest']=roc_auc
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC curve class {prop_labels[c]} vs Rest (area = %0.2f)'% roc_auc,
                        line=dict(color=colors[c%len(colors)], width=line_width)))

    if y_pred_proba.shape[1]>2:
        ovoroc= roc_auc_score(y_true, y_pred_proba[:,:],multi_class='ovo')
    else:
        ovoroc= roc_auc_score(y_true==0, y_pred_proba[:,0])
    
    if results_dict is not None:
        results_dict[title][f' AUC-ROC']=ovoroc
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                    mode='lines',
                    showlegend=False,
                    line=dict(color=f'rgba(112,128,144,0.8)', width=2.5, dash='dash')))
    fig.update_layout(title=f'AUC-ROC curves <br><sup>{title}</sup><br><sup>Mean One vs One AUC: {np.round(ovoroc,3)}</sup>',
                    title_font_size=18,
                   xaxis_title="False Positive Rate",
                   yaxis_title="True Positive Rate",
                   xaxis_range=[0.0,1.0],
                   yaxis_range=[0.0,1.05],font=dict(size=18),height=fig_size[1], width=fig_size[0],
                   legend=dict(
                        yanchor="bottom",
                        y=0.005,
                        xanchor="right",
                        x=0.995,
                        traceorder="normal",
                        bgcolor = 'rgba(255,255,255,0.5)',
                        font=dict(
                            size=11,
                            color="black"
                        ),
                    )
                     )
    
    return Youden_thresholds,Youden_vals,fig,f'Mean One vs One AUC: {np.round(ovoroc,3)}'


def plotly_enrichment_clf(y_pred_proba,y_true,title='',b=None, e=None ,prop_labels=None,colors=None,line_width=2, fig_size=(600,600), results_dict=None):
    """
    plots hit enrichment curves for classification
    """
    mask=np.isnan(y_true) 
    y_true=y_true[~mask]
    assert len(y_true) > 0, 'No data'
    fig = go.Figure()
    area_list=[]
    for c in range(y_pred_proba.shape[1]):
        TPF, SF =[], []
        if not b:
            b=0
        if not e:
            e=1
        cutoffs=np.linspace(b,e, num=50)
        N=len(y_true)
        for cutoff in cutoffs:
            p_label=(y_pred_proba[:,c] > cutoff)

            TP= sum((p_label==1) & (y_true==c))
            S= sum(p_label==1)
            TPF.append(TP/sum(y_true==c))
            SF.append(S/N)

        SF, TPF = (list(t) for t in zip(*sorted(zip(SF, TPF))))
        area=np.round(var_trapezoid_rule(SF,TPF),2)
        area_list.append(area)
        fig.add_trace(go.Scatter(x=SF, y=TPF,
                            mode='lines',
                            name=f'True Pos. Fraction {prop_labels[c]}: area={area}',
                            line=dict(color=colors[c%len(colors)], width=line_width)))
    mean_area=np.round(np.mean(area_list),3)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                    mode='lines',
                    showlegend=False,
                    line=dict(color=f'rgba(112,128,144,0.8)', width=2.5, dash='dash')))
    
    #fig.update_xaxes(type="log")
    fig.update_layout(title=f'Hit enrichment curve <br><sup>{title}</sup><br><sup>Mean area: {mean_area}</sup>',
                    title_font_size=18,
                   xaxis_title=f'Total Positive Fraction',
                   yaxis_title="True Positive Fraction",font=dict(size=18),height=fig_size[1], width=fig_size[0],
                    legend=dict(
                        yanchor="bottom",
                        y=0.005,
                        xanchor="right",
                        x=0.95,
                        traceorder="normal",
                        bgcolor = 'rgba(255,255,255,0.5)',
                        font=dict(
                            size=11,
                            color="black"
                        ),
                    )
                )
    return fig,f'Mean area: {mean_area}'

def make_plotly_confusion_matrix(cf,
                          labelnames='auto',
                          cmap='Blues',
                          title=None,
                          fig_size=(800,800)):

    if cf.shape[1]==2 :
        group_labels = ["{}<br> <br>".format(value) for value in ['True Negative','False Positive','False Negative','True Positive']]
    else:
        group_labels = ['' for i in range(cf.size)]
    group_counts = ["Cnt: {0:0.0f}<br>".format(value) for value in cf.flatten()]
    group_percentages = ["Glob.: {0:.2%}<br>".format(value) for value in cf.flatten()/np.sum(cf)]
    row_precentages= ["Row: {0:.2%}".format(value) for value in (cf/cf.sum(axis=1, keepdims=True)).flatten()] 
    box_labels = [f"{v1}{v2}{v3}{v4}".strip() for v1, v2, v3,v4 in zip(group_labels,group_counts,group_percentages,row_precentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])
    accuracy  = np.trace(cf) / float(np.sum(cf))
    
    size_font=np.maximum(22-2*cf.shape[1],10)       
    
    fig=go.Figure(data=go.Heatmap(z=cf,
                   x=labelnames,
                   y=labelnames, text=box_labels,
                   texttemplate="%{text}",
                   textfont={"size":size_font},
                   colorscale=cmap))
    fig.update_layout(title=f'Confusion Matrix <br><sup>{title}</sup><br><sup><b>Accuracy: {np.round(accuracy,3)}</b></sup>',
                    title_font_size=18,
                   xaxis_title='Predicted label',
                    yaxis=dict(autorange='reversed'),
                   yaxis_title='True label',
                  font=dict(size=size_font),height=fig_size[1], width=fig_size[0])
    return fig,f'Overall accuracy: {np.round(accuracy,3)}'
    

def plotly_clf_confusion(y_pred, y_true,title='',prop_labels=None,cmap=None,fig_size=(800,800), results_dict=None ):
    cm=confusion_matrix(y_true, y_pred)
    if results_dict is not None:
        results_dict[title][f' confusion matrix']=cm
    return make_plotly_confusion_matrix(cm, labelnames=list(prop_labels.values()),cmap=cmap,title=title,fig_size=fig_size)
    
def plotly_clf_matrix(y_pred, y_true,title='',labels=None,prop_labels=None,cmap='PiYG',fig_size=(800,800), results_dict=None ):
    clf_report = classification_report(y_true,
                                   y_pred,
                                   labels=labels,
                                   target_names=prop_labels,
                                   output_dict=True)
    # .iloc[:-1, :] to exclude support
    renamed_report={prop_labels[k] if k in labels else k:v for k,v in clf_report.items()}
    if results_dict is not None:
        results_dict[title][f' classification report']=renamed_report
    y_values=[k for k,v in renamed_report.items()]
    x_values=[k for k,v in renamed_report[prop_labels[0]].items()]
    accuracy=np.round(renamed_report['accuracy'],3)
    x_values.remove('support')
    y_values.remove('accuracy')
    z_values=[]
    for y in y_values[::-1]:
        z_row=[]
        if y=='accuracy':
            z_values.append([None,None,np.round(renamed_report[y],3)])
        else:
            for x in x_values:
                z_row.append(np.round(renamed_report[y][x],3))
            z_values.append(z_row)
    fig=go.Figure(data=go.Heatmap(z=z_values,
                   x=x_values,
                   y=y_values[::-1], text=z_values,
                   texttemplate="%{text}",
                   textfont={"size":18},
                   colorscale=cmap,
                   zmin=0.3,zmax=1.0))
    fig.update_layout(title=f'Classification report <br><sup>{title}</sup><br><sup><b>Accuracy: {accuracy}</b></sup><br>',
                    title_font_size=18,font=dict(size=18),height=fig_size[1], width=fig_size[0])
    fig.update_xaxes(side="top")
    return fig,f'Overall accuracy: {np.round(accuracy,3)}'