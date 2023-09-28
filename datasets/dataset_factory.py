from datasets.synthetic.mvd_fair_dataset import MVDFairDataset
from datasets.synthetic.ccd_fair_dataset import CCDFairDataset
from datasets.synthetic.ds_ccd_fair_dataset import DSCCDFairDataset
from datasets.synthetic.correlated_ccd_fair_dataset import CorrelatedCCDFairDataset

class DatasetFactory():
    def get_dataset(self, type, **kwargs):
        if type == 'mvd':
            return MVDFairDataset(**kwargs)
        elif type == 'ccd':
            return CCDFairDataset(**kwargs)
        elif type == 'ds_ccd':
            return DSCCDFairDataset(**kwargs)
        elif type == 'corr':
            return CorrelatedCCDFairDataset(**kwargs)
        else:
            raise ValueError('No such dataset.')