"""implementation of the matplotlib visualization

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
from sklearn.metrics import roc_auc_score,roc_curve, auc, classification_report           

from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def generate_distance_matrix_lowerdiagonal(dists,nfps):
    """
    generate lowerdiagonal distance matrix from list of distances
    """
    dmat=np.zeros((nfps,nfps))
    counter=0
    for i in range(1, nfps):
        values=dists[counter:counter+i]
        counter+=i
        dmat[i,:i]=values
        dmat[nfps-i-1,nfps-i:]=values
    return dmat


def plot_confusion_bars_from_continuos(df_raw,pro1=None,pro2=None, 
                    bins2=None,bins1= range(1,11),color=None,#['red','yellow','green']
                    figsize=(6, 6),
                  title=None,
                           leg_title='',
                           x_title=None, ax=None):
    '''
    Displays barplots where bins of one property is set wrt to bins of the other property
    
    Args:
         df_raw: dataframe containing both properties
         pro1: column name property 1
         pro2: column name property 2
         bins2: bins of the second property
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
        try:
            df['pro1_bin'] = pd.cut(df[pro1],bins=bins1,labels=labels1,ordered=False)
        except ValueError:
            return
            
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
            return
            
    ##############
    ##############
    res = {}
    bin_counts = []
    for k,v in df.groupby('pro1_bin'):
        if len(v) < 1: continue
        res[k]=v['pro2_bin'].value_counts(normalize=True).to_dict()
        bin_counts.append(len(v))
    
    dftm=pd.DataFrame.from_dict(res, orient='index').reindex(columns=labels2)#.sort_values('pro2_bin')
    #pd.CategoricalIndex(dftm.columns.values,   ordered=True,      categories=labels2)
    if ax:
        try:
            dftm.plot(kind="bar", stacked=True, color=colors2, ax=ax) 
        except:
            return
    else:
        ax=dftm.plot(kind="bar", stacked=True, color=colors2,figsize=figsize) 
    ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1,fontsize=9,title =leg_title )
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        #print(height)
        if height < 0.05:
            continue
        x, y = p.get_xy() 
        ax.text(x+width/2, 
                y+height/2, 
                '{:.0f} %'.format(100*height), 
                horizontalalignment='center', 
                verticalalignment='center',fontsize=9, rotation=90)
    ax.set_ylabel(f'counts',fontsize=10)
    if not x_title: x_title=f"Bins of {pro1}"
    ax.set_xlabel(x_title,fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    if not title: title=pro2
    ax.set_title(title,fontsize=12)
    pos = -0.2
    for b in bin_counts:
        st=str(b)#+ ';{}%'.format(round(100*b/np.sum(bin_counts),1))
        #st=str(b)+ 'cpds;{}%'.format(round(100*b/np.sum(bin_counts),1))
        ax.text(pos,1.01,st.center(4),fontsize=9)
        pos = pos+1.0

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
    """
    compute dynamic binned error
    """
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

def plot_reg_model_with_error(y_pred, y_true,title='', extend_axis_lim=0, ax=None, b=None, e=None ,metrics='all',alpha=0.35, mask_scatter=False,bins=10,prop_cliffs=None, leave_grp_out=None, bin_window_average=True ):
    """
    Scatterplot with moving average error and distinction between different points
    """
    #metrics="RMSE = %.5f , R2 = %.2f " % (np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred))
    mask=np.isnan(y_true) | np.isnan(y_pred.astype(float))
    y_true=y_true[~mask]
    y_pred=y_pred[~mask]
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
    #handles=[]
    #leg_labels=[]
    if not mask_scatter:
        stratified_mask=np.zeros(len(y_true))>0
        if prop_cliffs is not None:
            ax.scatter(y_true[prop_cliffs], y_pred[prop_cliffs],s=30, label='prop.-cliffs', color='darkmagenta', marker='P', alpha=alpha+0.15)
            #handles.append(propcliffs)
            #leg_labels.append('prop.-cliffs')
            stratified_mask[prop_cliffs]=True
        if leave_grp_out is not None:
            ax.scatter(y_true[leave_grp_out], y_pred[leave_grp_out],s=25, label='leave-group-out', color='darkgreen', marker='D', alpha=alpha)
            #handles.append(grp_out)
            #leg_labels.append('leave-group-out')
            stratified_mask[leave_grp_out]=True
        if len(y_true[~stratified_mask])>0:
            ax.scatter(y_true[~stratified_mask], y_pred[~stratified_mask],s=25, label='stratified', color='red', marker=".", alpha=alpha)
            #handles.append(starti)
            #leg_labels.append('stratified')
    #ax.legend(leg_labels, loc=4, frameon=True)     
    ax.plot(line_X, line_X, color='blue', label='Ideal', linewidth=lw, linestyle='--', alpha=0.6)
    #handles.append(ideall)
    #leg_labels.append('Ideal')
    if bin_window_average:
        m, err_mean,err_mean_neg,err_mean_pos, err_std=get_dyn_err(y_true,y_pred, bins=bins)
    else:
        m, err_mean = get_moving_err_mean(y_true,y_pred, window=np.minimum(bins,int(len(y_true)/2+1/2)))
    err=err_mean 
    #ax.plot(m, m, color='blue', linewidth=lw, linestyle='--')
    #ax.fill_between(m, m , m - 2*err, color='blue', alpha=0.1)
    #ax.fill_between(m, m , m - 1*err, color='blue', alpha=0.2)
    #plt.plot( m,m + 1*err , linewidth=1 , color='gray',linestyle='--', alpha=0.3)
    errl,=ax.plot( m,m - 1*err , linewidth=lw, color='blue', label='Moving average error',linestyle='-.', alpha=0.6)
    #handles.append(errl)
    #leg_labels.append('Moving average error')
    #ax.legend(handles,leg_labels, loc=4, frameon=True)   
    ax.legend(loc=4, frameon=True)
    #plt.plot( m,m + 1*err_mean_pos , linewidth=2 , color='gray',linestyle='--')#, alpha=0.4)
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted")
    ax.set_title(title)#+ f':{metrics}')
    ax.text(b,e, metrics,fontsize=12)


def plot_reg_model(y_pred, y_true,title='', extend_axis_lim=0, ax=None, b=None, e=None ,metrics='all' ):
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
    if metrics=='R2':
        metrics= "MAE=%.2f\nR^2= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), r2_score(y_true, y_pred))
    elif metrics== 'P_Corr':
        metrics= "MAE= %.2f\nP_Corr= %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0])
    else:
        metrics= "MAE = %.2f\nP_Corr = %.2f\nR^2 = %.2f" % (round(mean_absolute_error(y_true, y_pred),3), pearsonr(y_true, y_pred)[0], r2_score(y_true, y_pred))
    if not b:
        b=min(y_true.min(), y_pred.min())-extend_axis_lim
    if not e:
        e=max(y_true.max(), y_pred.max()) +extend_axis_lim
    line_X = np.linspace(b, e)
    lw = 1
    ax.scatter(y_true, y_pred, color='red', marker="o",  s=18, label='Inliers', alpha=0.4,edgecolors='black')

    ax.fill_between(line_X, line_X - 2*mae,line_X + 2*mae,color='blue', alpha=0.1)
    ax.fill_between(line_X, line_X - mae,line_X + mae,color='blue', alpha=0.25)
 
    ax.plot(line_X, line_X+mae, linewidth=lw,color='blue', alpha=0.3)
    ax.plot(line_X, line_X+2*mae, linewidth=lw,color='blue', alpha=0.3)
    ax.plot(line_X, line_X, linewidth=lw,color='blue', alpha=0.3)
    ax.plot(line_X, line_X-mae, linewidth=lw,color='blue', alpha=0.3)
    ax.plot(line_X, line_X-2*mae, linewidth=lw,color='blue', alpha=0.3)

    ax.legend(['',f'{fold1}% w. 1 fold MAE',f'{fold2}% w. 2 folds MAE'], loc='upper left', fontsize=10)
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted")
    ax.set_title(title)#+ f':{metrics}')
    xl=ax.get_xlim()
    yl=ax.get_ylim()
    px=xl[0]+0.6*(xl[1]-xl[0])
    py=yl[0]+0.1*(yl[1]-yl[0])
    ax.text(px,py, metrics,fontsize=14)

###############################################################################################

def plot_acc_pre_for_reg(y_true,y_pred,title='' ,b=None, e=None , good_class= '>', ax=None):
    """
    plot with precision, recall, accuracy and active ratio for varying threshold
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
        if (TP + FP) >0 and (TP + FN) > 0:
            pre=100*TP/(TP + FP)#sum((p_label==1) & (t_label==p_label))/sum(p_label==1)
            #recall = TP/(TP + FN)
            #What proportion of actual positives was identified correctly?
            rec=100*TP/(TP + FN) #sum((p_label==1) & (t_label==p_label))/sum(t_label==1)
            ACC.append(acc)
            PRE.append(pre)
            REC.append(rec)
            CLASS_ratio.append(100*sum(t_label)/len(y_true))
            used_cutoffs.append(cutoff)
    if ax:
        ax.plot(used_cutoffs, ACC, color='blue', linewidth=2, label='Accuracy')
        ax.plot(used_cutoffs, PRE, color='red', linewidth=2, label='Precision')
        ax.plot(used_cutoffs, REC, color='green', linewidth=2, label='Recall')
        ax.plot(used_cutoffs, CLASS_ratio, color='black', linewidth=2, label='Active ratio')
        ax.set_xlabel(f'Threshold (good ones {good_class} threshold)')
        ax.set_ylabel('%')
        ax.set_title(f'{title}')
        ax.legend(loc="lower right")
    else:
        plt.plot(used_cutoffs, ACC, color='blue', linewidth=2, label='Accuracy')
        plt.plot(used_cutoffs, PRE, color='red', linewidth=2, label='Precision')
        plt.plot(used_cutoffs, REC, color='green', linewidth=2, label='Recall')
        plt.plot(used_cutoffs, CLASS_ratio, color='black', linewidth=2, label='Active ratio')

        plt.xlabel(f'Threshold (good ones {good_class} threshold)')
        plt.ylabel('%')
        plt.title(f'{title}')
        plt.legend(loc="lower right")


##################################################################################################3
def df2category(df, pro, bins=None, quantile=[0.1] ,prefix='CAT', outname=None):
    """
    legacy code
    """
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

def plot_confusion_bars_from_categories(df_raw,pro1=None,pro2=None,
                                        bins1= range(1,11),color=None,#['red','yellow','green']
                    figsize=(6, 6),
                  title=None,
                           leg_title='',
                           x_title=None, ax=None,labelnames=None):
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
    if ax:
        dftm.plot(kind="bar", stacked=True, color=colors2, ax=ax) 
    else:
        ax=dftm.plot(kind="bar", stacked=True, color=colors2,figsize=figsize) 
    ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1,fontsize=9,title =leg_title )
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        #print(height)
        if height < 0.05:
            continue
        x, y = p.get_xy() 
        ax.text(x+width/2, 
                y+height/2, 
                '{:.0f} %'.format(100*height), 
                horizontalalignment='center', 
                verticalalignment='center',fontsize=9, rotation=90)
    ax.set_ylabel(f'counts',fontsize=10)
    if not x_title: x_title=f"Bins of {pro1}"
    ax.set_xlabel(x_title,fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    if not title: title=pro2
    ax.set_title(title,fontsize=12)
    pos = -0.2
    for b in bin_counts:
        st=str(b)#+ ';{}%'.format(round(100*b/np.sum(bin_counts),1))
        #st=str(b)+ 'cpds;{}%'.format(round(100*b/np.sum(bin_counts),1))
        ax.text(pos,1.01,st.center(4),fontsize=9)
        pos = pos+1.0  
    
def plot_clf_f1(y_pred_proba, y_true,title='', ax=None,colors=None,line_width=2,c=0, labelname=None,youden=None,youden_val='',verbose=True ):
    """
    Plots precision, recall and optimal thresholds for classification
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:,c], pos_label=c)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.nanargmax(fscore)
    if verbose: print('Class %s: Best Threshold=%f, precision=%.3f, recall=%.3f and F-Score=%.3f (uses validation set!)' % 
              (labelname,thresholds[ix],precision[ix],recall[ix],fscore[ix]))
    ax.plot(
        thresholds,
        precision[:-1],
        color=colors[0],
        lw=line_width,
        label=f'Precision {labelname}',
    )
    ax.plot(
        thresholds,
        recall[:-1],
        color=colors[1],
        lw=line_width,
        label=f'Recall {labelname}',
    )
    ax.plot(
        thresholds,
        fscore[:-1],
        color=colors[2],
        lw=line_width,
        label=f'F1-score {labelname}',
    )
    ax.axvline(x=0.5, color='navy', linestyle='--')
    ax.axvline(x=thresholds[ix], color=colors[3], linestyle='--')
    ax.text(thresholds[ix]+0.01,0.02,f'F1={np.round(fscore[ix],3)}',rotation=90,color=colors[3],fontsize=10)
    xticks=np.linspace(0,1,5)
    xticks = xticks[np.absolute(xticks-thresholds[ix])>0.05]
    xticks=np.append(xticks, thresholds[ix])
    if youden is not None:
        ax.axvline(x=youden, color=colors[4], linestyle='--')
        xticks = xticks[np.absolute(xticks-youden)>0.05]
        xticks=np.append(xticks, youden)
        #check if thresholds overlap
        if np.abs(youden-thresholds[ix])>0.05:
            ax.text(youden+0.01,0.015,f'Youdens-J={np.round(youden_val,3)}',rotation=90,color=colors[4],fontsize=10)
        else:
            ax.text(youden-0.03,0.015,f'Youdens-J={np.round(youden_val,3)}',rotation=90,color=colors[4],fontsize=10)

    ax.legend(loc=0)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Precision/Recall/F1-score")
    ax.set_title(title)
    
    ax.set_xticks(np.sort(np.round(xticks,2)))
    return thresholds[ix],fscore[ix]
      
    
def plot_clf_auc(y_pred_proba, y_true,title='', ax=None,colors=None,line_width=2,prop_labels=None ):
    """
    plots roc-auc curves
    """
    Youden_thresholds=[]
    Youden_vals=[]
    for c in range(y_pred_proba.shape[1]):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:,c], pos_label=c)
        J = tpr - fpr
        ix = np.argmax(J)
        print('Class %s: Best Threshold=%f, Recall=%.3f (uses validation set!), Youdens J=%.3f (uses validation set!)' % (prop_labels[c],thresholds[ix], tpr[ix], J[ix]))
        Youden_thresholds.append(thresholds[ix])
        Youden_vals.append(J[ix])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=colors[c%len(colors)],
            lw=line_width,
            label=f'ROC curve class {prop_labels[c]} vs Rest (area = %0.2f)' % roc_auc,
        )

    if y_pred_proba.shape[1]>2:
        ovoroc= roc_auc_score(y_true, y_pred_proba[:,:],multi_class='ovo')
    else:
        ovoroc= roc_auc_score(y_true==0, y_pred_proba[:,0])
    ax.legend(loc="lower right")
    ax.plot([0, 1], [0, 1], color="navy", lw=line_width, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f'OnevOne AUC {title}: {np.round(ovoroc,2)}')
    return Youden_thresholds,Youden_vals

def make_confusion_matrix(cf,
                          labelnames='auto',
                          cmap='Blues',
                          title=None,
                          ax=None):
    """
    creates confusion matrix
    """
    if cf.shape[1]==2 :
        group_labels = ["{}\n\n".format(value) for value in ['True Negative','False Positive','False Negative','True Positive']]
    else:
        group_labels = ['' for i in range(cf.size)]
    group_counts = ["Cnt: {0:0.0f}\n".format(value) for value in cf.flatten()]
    group_percentages = ["Glob.: {0:.2%}\n".format(value) for value in cf.flatten()/np.sum(cf)]
    row_precentages= ["Row: {0:.2%}".format(value) for value in (cf/cf.sum(axis=1, keepdims=True)).flatten()] 
    box_labels = [f"{v1}{v2}{v3}{v4}".strip() for v1, v2, v3,v4 in zip(group_labels,group_counts,group_percentages,row_precentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])
    accuracy  = np.trace(cf) / float(np.sum(cf))
    
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=True,xticklabels=labelnames,yticklabels=labelnames,ax=ax)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title+" accuracy={:0.3f}".format(accuracy))
    

def plot_clf_confusion(y_pred, y_true,title='', ax=None,prop_labels=None,cmap=None ):
    cm=confusion_matrix(y_true, y_pred)
    make_confusion_matrix(cm, labelnames=list(prop_labels.values()),cmap=cmap,title=title,ax=ax)

def plot_clf_matrix(y_pred, y_true,title='', ax=None,labels=None,prop_labels=None,cmap='PiYG' ):
    clf_report = classification_report(y_true,
                                   y_pred,
                                   labels=labels,
                                   target_names=prop_labels,
                                   output_dict=True)
    """
    creates colored classification report
    """
    # .iloc[:-1, :] to exclude support
    renamed_report={prop_labels[k] if k in labels else k:v for k,v in clf_report.items()}
    sns.heatmap(pd.DataFrame(renamed_report).iloc[:-1, :].T, annot=True,ax=ax,vmin=0.3, vmax=1.0, square=True, cmap=cmap)
    ax.set_title(title)
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')