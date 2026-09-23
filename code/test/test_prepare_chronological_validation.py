"""Checks for chronology and quote eligibility before market outcome scoring."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd

spec=importlib.util.spec_from_file_location('prepare',Path(__file__).resolve().parents[1]/'scripts/prepare_chronological_validation.py')
prepare=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

class ChronologyTests(unittest.TestCase):
    def test_missing_capture_is_not_a_shorter_horizon(self):
        self.assertEqual(prepare.advance_session('2026-08-19',1),pd.Timestamp('2026-08-20'))
        self.assertEqual(prepare.advance_session('2026-07-02',1),pd.Timestamp('2026-07-06'))
    def test_quote_filter_rejects_crossed_and_unavailable_size(self):
        data=pd.DataFrame(dict(strike=[100.]*4,und_close=[100.]*4,bid=[1.]*4,
            ask=[2.,.5,2.,float('nan')],bid_size=[1.,1.,0.,1.],ask_size=[1.]*4))
        self.assertEqual(prepare.valid_quotes(data).tolist(),[True,False,False,False])

if __name__=='__main__':unittest.main()
