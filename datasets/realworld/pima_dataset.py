import os
import pandas as pd
from aif360.datasets import BinaryLabelDataset


class PimaDataset(BinaryLabelDataset):
    def __init__(self, label_names=['Class'],
                 protected_attribute_names=['Age'],
                 favorable_label=1
                 ):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(file_dir, '../..', 'data', 'raw', 'pima',
                                 'pima-indians-diabetes.csv')
        df = pd.read_csv(file_path)

        # Quantize age
        def age_quantizer(x):
            if x <= 45:
                return 1
            else:
                return 0
        df['Age'] = df['Age'].astype(int).apply(age_quantizer)

        super(PimaDataset, self).__init__(
            df=df, label_names=label_names,
            protected_attribute_names=protected_attribute_names,
            favorable_label=favorable_label,
            unfavorable_label=1-favorable_label)

    def __len__(self):
        return self.features.shape[0]