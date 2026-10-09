"""Regenerate the result figures and tables from the saved out-of-fold predictions.

Inputs (produced by the pipeline scripts, not tracked in Git):
    results/predictions_MIMIC_IV.pkl, results/predictions_eICU.pkl   (Part 1 / Part 2)
    results/stress_results_incremental.csv                           (Part 1B)
Outputs:
    figures/mnar_stress_test.pdf (MNAR stress test)   figures/risk_coverage.pdf (risk-coverage)
    figures/decision_curves.pdf (decision curves)    figures/reliability.pdf (reliability)
    results/paper_tables/*.csv  (main metrics, DeLong/Wilcoxon tests, operating-point metrics, 5-fold DCA)
Usage:  python scripts/make_paper_outputs.py [results_dir] [figures_dir]
"""
import sys, os
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = sys.argv[2] if len(sys.argv) > 2 else "figures"
TAB = os.path.join(RES, "paper_tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, norm, rankdata, t as tdist
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, f1_score
O_FIG = FIG + os.sep; O_TAB = TAB + os.sep
def load_preds(fn): return pickle.load(open(os.path.join(RES, fn), "rb"))["fold_predictions"]

# ======================================================================
# Risk-coverage (Fig. 4), DCA (Fig. 5), operating-point metrics
# ======================================================================
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
C={"JUCO":"#2196F3","XGBoost":"#FF5722","LightGBM":"#4CAF50","MissForest+XGB":"#9C27B0","JUCO-XGB":"#03A9F4","Treat All":"#888888"}
D={}
for ds,fn in [('MIMIC-IV','predictions_MIMIC_IV.pkl'),('eICU','predictions_eICU.pkl')]:
    D[ds]=load_preds(fn)
def ent(p): p=np.clip(p,1e-12,1); return -(p*np.log(p)).sum(1)
def rc(P,y):
    e=(P.argmax(1)!=y).astype(float)[np.argsort(ent(P),kind='stable')]; n=len(e)
    return np.arange(1,n+1)/n, np.cumsum(e)/np.arange(1,n+1)
grid=np.linspace(0.05,1,400)
models=['JUCO','XGBoost','LightGBM','MissForest+XGB']
# ---- risk-coverage
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
for ax,ds in zip(axs,D):
    for m in models:
        rs=np.array([np.interp(grid,*rc(f[m],f['y_true'])) for f in D[ds]])
        mu=rs.mean(0); ax.plot(grid,mu,color=C[m],lw=2.4 if m=='JUCO' else 1.5,ls='-' if m=='JUCO' else '--',label=m)
        if m=='JUCO': ax.fill_between(grid,rs.min(0),rs.max(0),color=C[m],alpha=.15,lw=0)
    err=np.mean([(f['JUCO'].argmax(1)!=f['y_true']).mean() for f in D[ds]])
    ax.axhline(err,color='#F44336',ls=':',lw=1.3,label=f'JUCO, no deferral ({err:.3f})')
    ax.axvline(.9,color='#999',lw=.8,ls=':')
    ax.set_xlabel('Coverage'); ax.set_ylabel('Selective risk (error on retained cases)'); ax.set_xlim(.05,1.0); ax.set_ylim(0,err*1.25)
    ax.grid(alpha=.3,ls=':'); ax.set_title(ds,fontsize=10)
axs[0].legend(fontsize=8,loc='upper left',frameon=False)
plt.tight_layout(); plt.savefig(O_FIG+'risk_coverage.pdf'); plt.savefig(O_FIG+'risk_coverage.png',dpi=200); plt.close()
# ---- DCA all 5 folds
th=np.round(np.arange(0.01,0.50,0.01),2); dm=['JUCO','JUCO-XGB','XGBoost','LightGBM','MissForest+XGB']
res={}
for ds in D:
    rows=[]
    acc={m:np.zeros(len(th)) for m in dm+['Treat All']}
    for f in D[ds]:
        y=f['y_true'];n=len(y);tp_all=(y==1).sum()/n;fp_all=(y==0).sum()/n
        acc['Treat All']+=tp_all-fp_all*th/(1-th)
        for m in dm:
            p=f[m][:,1]
            for i,t in enumerate(th):
                pr=p>=t; acc[m][i]+=((pr&(y==1)).sum()/n)-((pr&(y==0)).sum()/n)*t/(1-t)
    df=pd.DataFrame({'Threshold':th,**{m:acc[m]/5 for m in dm},'Treat All':acc['Treat All']/5,'Treat None':0.0})
    df.to_csv(O_TAB+f"dca_{ds.replace('-','_')}_5fold.csv",index=False); res[ds]=df
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
for ax,ds in zip(axs,D):
    df=res[ds]
    for m in dm+['Treat All']:
        ax.plot(df.Threshold,df[m],color=C[m],lw=2.4 if m=='JUCO' else 1.5,ls='-' if m in('JUCO','Treat All') else '--',label=m)
    ax.axhline(0,color='#bbb',lw=1,label='Treat None'); ax.set_ylim(-.02,df.JUCO.max()*1.25); ax.set_xlim(.01,.49)
    ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit'); ax.grid(alpha=.3,ls=':'); ax.set_title(ds,fontsize=10)
axs[0].legend(fontsize=8,frameon=False)
plt.tight_layout(); plt.savefig(O_FIG+'decision_curves.pdf'); plt.savefig(O_FIG+'decision_curves.png',dpi=200); plt.close()
# ---- metrics table
def st(P,y,t):
    pr=P[:,1]>=t;tp=(pr&(y==1)).sum();fp=(pr&(y==0)).sum();fn=((~pr)&(y==1)).sum();tn=((~pr)&(y==0)).sum()
    return tp/(tp+fn),tn/(tn+fp),tp/(tp+fp)
rows=[]
for ds in D:
    for m in ['JUCO','JUCO-XGB','XGBoost','LightGBM','MissForest+XGB']:
        for t in (.5,.2):
            v=np.array([st(f[m],f['y_true'],t) for f in D[ds]])
            rows.append({'Dataset':ds,'Model':m,'Threshold':t,**{k:f"{v[:,i].mean():.3f} ± {v[:,i].std():.3f}" for i,k in enumerate(['Sensitivity','Specificity','Precision'])}})
M=pd.DataFrame(rows); M.to_csv(O_TAB+'operating_point_metrics.csv',index=False); print(M.to_string())
# at 90% coverage summary
for ds in D:
    for m in models:
        r=[rc(f[m],f['y_true']) for f in D[ds]]; print(ds,m,'Risk@90',np.mean([x[1][np.argmin(abs(x[0]-.9))] for x in r]).round(4))
for ds in D: print(ds,'DCA 5-fold @0.10/0.20:',res[ds].set_index('Threshold').loc[[0.1,0.2],['JUCO','XGBoost','Treat All']].round(4).to_dict('index'))

# ======================================================================
# Reliability diagrams (Fig. D1)
# ======================================================================
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
C={"JUCO":"#2196F3","XGBoost":"#FF5722","LightGBM":"#4CAF50","MissForest+XGB":"#9C27B0"}
def ece(P,y,nb=15):
    c=P.max(1);ok=(P.argmax(1)==y).astype(float);e=np.linspace(0,1,nb+1);s=0
    for i in range(nb):
        m=(c>e[i])&(c<=e[i+1])
        if m.sum(): s+=m.sum()/len(y)*abs(ok[m].mean()-c[m].mean())
    return s
def bins(P,y,nb=15):
    c=P.max(1);ok=(P.argmax(1)==y).astype(float);e=np.linspace(0,1,nb+1);out=[]
    for i in range(nb):
        m=(c>e[i])&(c<=e[i+1])
        if m.sum()>=30: out.append((c[m].mean(),ok[m].mean(),m.sum()))
    return np.array(out)
fig,axs=plt.subplots(2,4,figsize=(12,6),sharex=True,sharey=True)
for r,(ds,fn) in enumerate([('MIMIC-IV','predictions_MIMIC_IV.pkl'),('eICU','predictions_eICU.pkl')]):
    fp=load_preds(fn)
    for c_,m in enumerate(C):
        ax=axs[r,c_]; P=np.vstack([f[m] for f in fp]); y=np.concatenate([f['y_true'] for f in fp])
        b=bins(P,y); e=np.mean([ece(f[m],f['y_true']) for f in fp])
        ax.plot([.5,1],[.5,1],'k--',lw=1,alpha=.5)
        ax.plot(b[:,0],b[:,1],'-',color=C[m],lw=1.5); ax.scatter(b[:,0],b[:,1],s=18+60*np.log10(b[:,2])/np.log10(b[:,2].max()),color=C[m],zorder=3)
        if r==0: ax.set_title(m,fontsize=10)
        if c_==0: ax.set_ylabel(f'{ds}\nObserved accuracy')
        if r==1: ax.set_xlabel('Mean predicted confidence')
        ax.set_xlim(.5,1.0);ax.set_ylim(.3,1.0);ax.grid(alpha=.3,ls=':')
        print(ds,m,round(e,4))
plt.tight_layout();plt.savefig(O_FIG+'reliability.pdf');plt.savefig(O_FIG+'reliability.png',dpi=200)

# ======================================================================
# MNAR stress test (Fig. 3)
# ======================================================================
matplotlib.use('Agg');import matplotlib.pyplot as plt
R=RES+os.sep
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
d=pd.read_csv(R+'stress_results_incremental.csv');d['r']=d.missing_rate.str.rstrip('%').astype(int)
C={"JUCO":"#2196F3","XGBoost":"#FF5722","LightGBM":"#4CAF50","MissForest+XGB":"#9C27B0","MeanImp+XGB":"#FF9800","JUCO-XGB":"#03A9F4"}
fig,ax=plt.subplots(figsize=(6.5,4.3))
for m,c in C.items():
    g=d.groupby('r')[m+'_AUROC'].agg(['mean','min','max'])
    ax.plot(g.index,g['mean'],'-o' if m=='JUCO' else '--s',color=c,lw=2.6 if m=='JUCO' else 1.4,ms=4,label=m)
    if m=='JUCO': ax.fill_between(g.index,g['min'],g['max'],color=c,alpha=.15,lw=0)
    print(m,round(g['mean'].iloc[-1]-g['mean'].iloc[0],4))
ax.set_xlabel('Additional MNAR masking rate in training data (%)');ax.set_ylabel('AUROC (MIMIC-IV, unmasked test data)')
ax.grid(alpha=.3,ls=':');ax.legend(fontsize=8,frameon=False,ncol=2,loc='lower left')
plt.tight_layout();plt.savefig(O_FIG+'mnar_stress_test.pdf');plt.savefig(O_FIG+'mnar_stress_test.png',dpi=200)

# ======================================================================
# Main metrics table
# ======================================================================
def ent(p): p=np.clip(p,1e-12,1); return -(p*np.log(p)).sum(1)
def ece(P,y,nb=15):
    c=P.max(1);ok=(P.argmax(1)==y).astype(float);e=np.linspace(0,1,nb+1);s=0
    for i in range(nb):
        m=(c>e[i])&(c<=e[i+1])
        if m.sum(): s+=m.sum()/len(y)*abs(ok[m].mean()-c[m].mean())
    return s
def met(P,y):
    pr=P.argmax(1);e=(pr!=y).astype(float)[np.argsort(ent(P),kind='stable')];n=len(e)
    cov=np.arange(1,n+1)/n;r=np.cumsum(e)/np.arange(1,n+1)
    return dict(Accuracy=accuracy_score(y,pr),F1_macro=f1_score(y,pr,average='macro'),AUROC=roc_auc_score(y,P[:,1]),ECE=ece(P,y),Brier=brier_score_loss(y,P[:,1]),AURC=np.trapezoid(r,cov),Risk90=r[np.argmin(abs(cov-.9))])
rows=[]
for ds,fn in [('MIMIC-IV','predictions_MIMIC_IV.pkl'),('eICU','predictions_eICU.pkl')]:
    fp=load_preds(fn)
    for m in [k for k in fp[0] if k not in('fold','y_true')]:
        d=pd.DataFrame([met(f[m],f['y_true']) for f in fp])
        rows.append({'Dataset':ds,'Model':m,**{k:f"{d[k].mean():.4f} ± {d[k].std(ddof=0):.4f}" for k in d}})
R=pd.DataFrame(rows);R.to_csv(O_TAB+'main_results_from_pkl.csv',index=False);print(R.to_string())

# ======================================================================
# DeLong / Wilcoxon tests
# ======================================================================
from sklearn.metrics import roc_auc_score
def midrank(x):
    return (rankdata(x,method='average'))
def fast_delong(y,sa,sb):
    pos=y==1;m=pos.sum();n=(~pos).sum();out=[]
    for s in (sa,sb):
        tx=midrank(s[pos]);ty=midrank(s[~pos]);tz=midrank(s)
        out.append((tz[pos].mean()*0+ (tz[pos].sum()/m - (m+1)/2)/n, (tz[pos]-tx)/n, 1-(tz[~pos]-ty)/m))
    aucs=np.array([o[0] for o in out]);v10=np.vstack([o[1] for o in out]);v01=np.vstack([o[2] for o in out])
    S=np.cov(v10)/m+np.cov(v01)/n;d=aucs[0]-aucs[1];var=S[0,0]+S[1,1]-2*S[0,1]
    z=d/np.sqrt(var);p=2*(1-norm.cdf(abs(z)));ci=(d-1.96*np.sqrt(var),d+1.96*np.sqrt(var))
    return aucs[0],aucs[1],d,z,p,ci
rows=[];wrows=[]
for ds,fn in [('MIMIC-IV','predictions_MIMIC_IV.pkl'),('eICU','predictions_eICU.pkl')]:
    fp=load_preds(fn)
    y=np.concatenate([f['y_true'] for f in fp])
    bl=[k for k in fp[0] if k not in('fold','y_true','JUCO')]
    S={k:np.concatenate([f[k][:,1] for f in fp]) for k in ['JUCO']+bl}
    alpha=0.05/len(bl)
    for b in bl:
        a,c,d,z,p,ci=fast_delong(y,S['JUCO'],S[b])
        fa=np.array([roc_auc_score(f['y_true'],f['JUCO'][:,1]) for f in fp]);fb=np.array([roc_auc_score(f['y_true'],f[b][:,1]) for f in fp])
        diff=fa-fb;sd=diff.std(ddof=1);hw=tdist.ppf(.975,4)*sd/np.sqrt(5)
        pw=wilcoxon(fa,fb).pvalue
        rows.append({'Dataset':ds,'Comparison':f'JUCO vs {b}','AUROC_JUCO_pooled':round(a,4),'AUROC_baseline_pooled':round(c,4),'Delta':f'{d:+.4f}','DeLong_95CI':f'[{ci[0]:+.4f}, {ci[1]:+.4f}]','DeLong_z':round(z,2),'DeLong_p':f'{p:.2e}','Bonferroni_alpha':round(alpha,4),'Significant_DeLong':'Yes' if p<alpha else 'No','Folds_JUCO_higher':f'{(diff>0).sum()}/5','Mean_fold_delta':f'{diff.mean():+.4f}','t95CI_fold_delta':f'[{diff.mean()-hw:+.4f}, {diff.mean()+hw:+.4f}]','Wilcoxon_p':round(pw,4)})
R=pd.DataFrame(rows);R.to_csv(O_TAB+'statistical_tests_from_pkl.csv',index=False)
print(R[['Dataset','Comparison','AUROC_JUCO_pooled','AUROC_baseline_pooled','Delta','DeLong_95CI','DeLong_z','DeLong_p','Significant_DeLong','Folds_JUCO_higher','Wilcoxon_p']].to_string())
