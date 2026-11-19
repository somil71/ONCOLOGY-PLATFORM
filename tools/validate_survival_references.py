"""Independent library comparisons for the survival evaluator (optional lifelines)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from src.survival_metrics import CensoringDistribution, harrell_c, horizon_metrics, kaplan_meier, uno_c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-dir',type=Path)
    parser.add_argument('--sksurv-dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path('RUN REPORT/survival-metric-reference-checks.json'))
    args=parser.parse_args()
    if args.dependency_dir:sys.path.insert(0,str(args.dependency_dir))
    import lifelines
    from lifelines import KaplanMeierFitter
    from lifelines.utils import concordance_index
    import sklearn
    from sklearn.metrics import roc_auc_score
    rng=np.random.default_rng(789)
    for _ in range(100):
        time=rng.integers(1,20,size=50);event=rng.integers(0,2,size=50);risk=rng.integers(0,10,size=50)
        np.testing.assert_allclose(harrell_c(time,event,risk),concordance_index(time,-risk,event),atol=1e-12,rtol=0)
        fit=KaplanMeierFitter().fit(time,event_observed=event)
        actual=kaplan_meier(time,event);times=[row['time'] for row in actual]
        np.testing.assert_allclose([r['survival'] for r in actual],fit.predict(times),atol=1e-12,rtol=0)
        confidence=fit.confidence_interval_.loc[times]
        np.testing.assert_allclose([r['lower_95'] for r in actual],confidence.iloc[:,0],atol=1e-12,rtol=0)
        np.testing.assert_allclose([r['upper_95'] for r in actual],confidence.iloc[:,1],atol=1e-12,rtol=0)
    for _ in range(100):
        time=rng.uniform(1,20,size=50);event=rng.integers(0,2,size=50);risk=rng.uniform(0,1,size=50)
        cases=(event==1)&(time<=10);controls=time>10;known=cases|controls
        censor=CensoringDistribution(time,event);weights=np.zeros(50)
        weights[cases]=1/censor.at(time[cases]);weights[controls]=1
        expected=roc_auc_score(cases[known],risk[known],sample_weight=weights[known])
        np.testing.assert_allclose(horizon_metrics(time,event,risk,time,event,10)['ipcw_auc'],expected,rtol=0,atol=1e-12)
    result={'status':'passed','seed':789,'randomized_harrell_comparisons':100,
        'randomized_km_survival_and_loglog_ci_comparisons':100,'randomized_weighted_auc_comparisons':100,
        'absolute_tolerance':1e-12,'lifelines_version':lifelines.__version__,'sklearn_version':sklearn.__version__,
        'dependency_directory':str(args.dependency_dir) if args.dependency_dir else None,
        'limitations':'Independent arithmetic checks, not external model validation; IPCW Brier and reverse-KM ties also have analytic regression examples.'}
    if args.sksurv_dir:
        sys.path.insert(0,str(args.sksurv_dir))
        import sksurv
        from sksurv.nonparametric import CensoringDistributionEstimator
        from sksurv.metrics import brier_score, cumulative_dynamic_auc, concordance_index_ipcw
        rng=np.random.default_rng(101)
        for _ in range(100):
            train_time=rng.integers(1,21,100);train_event=rng.integers(0,2,100)
            train_time[-1]=25;train_event[-1]=1
            time=rng.integers(1,19,50);event=rng.integers(0,2,50)
            risk=rng.uniform(0,1,50);probability=rng.uniform(0,1,50)
            reference_train=np.array(list(zip(train_event.astype(bool),train_time)),dtype=[('event','?'),('time','f8')])
            reference_test=np.array(list(zip(event.astype(bool),time)),dtype=reference_train.dtype)
            reference_censor=CensoringDistributionEstimator().fit(reference_train)
            np.testing.assert_allclose(CensoringDistribution(train_time,train_event).at(np.arange(1,20)),
                reference_censor.predict_proba(np.arange(1,20)),atol=1e-12,rtol=0)
            ours=horizon_metrics(time,event,risk,train_time,train_event,10,probability)
            auc,_=cumulative_dynamic_auc(reference_train,reference_test,risk,[10])
            _,brier=brier_score(reference_train,reference_test,probability[:,None],[10])
            np.testing.assert_allclose(ours['ipcw_auc'],auc[0],atol=1e-12,rtol=0)
            np.testing.assert_allclose(ours['ipcw_brier'],brier[0],atol=1e-12,rtol=0)
            np.testing.assert_allclose(uno_c(time,event,risk,train_time,train_event,10),
                concordance_index_ipcw(reference_train,reference_test,risk,tau=10)[0],atol=1e-12,rtol=0)
        result.update({'sksurv_version':sksurv.__version__,'sksurv_dependency_directory':str(args.sksurv_dir),
            'randomized_reverse_km_tied_censor_comparisons':100,'randomized_sksurv_ipcw_auc_comparisons':100,
            'randomized_sksurv_ipcw_brier_comparisons':100,'randomized_sksurv_uno_comparisons':100,
            'limitations':'Independent arithmetic parity, including tied event/censoring times; not external model validation.'})
    from lifelines import CoxPHFitter
    from src.survival_models import RegularizedCox
    import pandas as pd
    rng=np.random.default_rng(20261118)
    covariates=rng.normal(size=(100,3))
    times=rng.exponential(np.exp(-(covariates@np.array([.4,-.2,.1]))))*10+.001
    events=rng.binomial(1,.7,100)
    model=RegularizedCox(0).fit(covariates,times,events)
    table=pd.DataFrame(covariates,columns=['a','b','c']);table['time']=times;table['event']=events
    reference=CoxPHFitter().fit(table,duration_col='time',event_col='event',
        fit_options={'precision':1e-12,'r_precision':1e-12})
    np.testing.assert_allclose(model.coef_,reference.params_.to_numpy(),atol=1e-6,rtol=0)
    horizons=model.event_times_[::10]
    np.testing.assert_allclose(model.predict_survival(covariates,horizons),
        reference.predict_survival_function(table,times=horizons).T.to_numpy(),atol=1e-6,rtol=0)
    result.update({'cox_fit_reference':'lifelines, no tied event times and zero regularization',
        'cox_fit_coefficient_max_absolute_difference':float(np.max(np.abs(model.coef_-reference.params_.to_numpy()))),
        'cox_fit_survival_at_event_times_max_absolute_difference':float(np.max(np.abs(model.predict_survival(covariates,horizons)-reference.predict_survival_function(table,times=horizons).T.to_numpy()))),
        'cox_fit_absolute_tolerance':1e-6})
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
