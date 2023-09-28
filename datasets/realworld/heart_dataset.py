import os
import pandas as pd
from aif360.datasets import BinaryLabelDataset


class HeartDataset(BinaryLabelDataset):
    def __init__(self, label_names=['cardio'],
                 protected_attribute_names=['age'],
                 favorable_label=1
                 ):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(file_dir, '../..', 'data', 'raw', 'heart',
                                 'cardio_train.csv')
        df = pd.read_csv(file_path, sep=';')
        df.set_index(keys=['id'], drop=True, inplace=True)
        # Quantize age
        def age_quantizer(x):
            if x <= 45*365:
                return 1
            else:
                return 0
        df['age'] = df['age'].astype(int).apply(age_quantizer)

        for s in ['gender', 'age']:
            if s not in protected_attribute_names:
                df.drop(columns=[s], inplace=True)

        super(HeartDataset, self).__init__(
            df=df, label_names=label_names,
            protected_attribute_names=protected_attribute_names,
            favorable_label=favorable_label,
            unfavorable_label=1-favorable_label)

    def __len__(self):
        return self.features.shape[0]

    @property
    def __n_features__(self):
        return len(self.feature_names + self.label_names +
                   self.protected_attribute_names)