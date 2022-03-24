from .mvd_fair_dataset import MVDFairDataset
from .ccd_fair_dataset import CCDFairDataset
from .ds_ccd_fair_dataset import DSCCDFairDataset


class DatasetFactory():
    def get_dataset(self, type, **kwargs):
        if type == 'mvd':
            return MVDFairDataset(**kwargs)
        elif type == 'ccd':
            return CCDFairDataset(**kwargs)
        elif type == 'ds_ccd':
            return DSCCDFairDataset(**kwargs)
        else:
            raise ValueError('No such dataset.')