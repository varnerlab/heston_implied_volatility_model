"""Check causal features, response cutoffs and shrinkage in the small comparison."""
from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from small_stock_models import FEATURES,feature_history,fit_ridge,predict_daily_drift


class SmallStockTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.closes=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0,.01,(400,3)),axis=0)),
            index=pd.bdate_range('2020-01-01',periods=400),columns=['GS','LLY','SPY'])

    def test_future_prices_do_not_change_origin_features(self):
        full,_=feature_history(self.closes,.94)
        prefix,_=feature_history(self.closes.iloc[:250],.94)
        cols=['date','ticker','variance',*FEATURES]
        pd.testing.assert_frame_equal(full[full.date<=self.closes.index[249]][cols].reset_index(drop=True),prefix[cols])

    def test_missing_close_is_not_an_accumulated_daily_return(self):
        data=self.closes.copy();data.iloc[200,0]=np.nan
        history,variance=feature_history(data,.9)
        self.assertEqual(variance.GS.iloc[199],variance.GS.iloc[200])
        self.assertEqual(variance.GS.iloc[199],variance.GS.iloc[201])
        a=history[(history.ticker=='GS')&(history.date==data.index[201])].iloc[0]
        self.assertEqual(a.stock_feature_age,(data.index[201]-data.index[199]).days)

    def test_training_cutoff_excludes_later_responses(self):
        history,_=feature_history(self.closes,.94)
        cutoff=self.closes.index[250]
        model=fit_ridge(history,'GS',cutoff,1.)
        perturbed=history.copy()
        perturbed.loc[perturbed.response_date>cutoff,'response']=1e9
        self.assertEqual(model,fit_ridge(perturbed,'GS',cutoff,1.))
        self.assertLessEqual(pd.Timestamp(model['last_response_date']),cutoff)

    def test_zero_drift_limit_and_training_scale(self):
        history,_=feature_history(self.closes,.94)
        model=fit_ridge(history,'LLY',self.closes.index[250],np.inf)
        self.assertTrue(np.array_equal(predict_daily_drift(history,model),np.zeros(len(history))))
        changed=history.copy()
        changed.loc[changed.date>self.closes.index[250],FEATURES]=1e6
        self.assertEqual(model,fit_ridge(changed,'LLY',self.closes.index[250],np.inf))


if __name__=='__main__':unittest.main()
