from aif360.datasets import BinaryLabelDataset

class RealWorldBase(BinaryLabelDataset):

    def __init__(self, *args, **kwargs):
        super(RealWorldBase, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.features.shape[0]

    @property
    def __n_features__(self):
        return len(self.feature_names + self.label_names +
                   self.protected_attribute_names)
