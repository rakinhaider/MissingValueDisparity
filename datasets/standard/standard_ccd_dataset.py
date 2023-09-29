from scipy import sparse
from datasets.synthetic.ccd_fair_dataset import CCDFairDataset
from datasets.standard.standard_missing_strategies import *


class StandardCCDDataset(CCDFairDataset):
    def __init__(self, std_dataset, **kwargs):
        df, _ = std_dataset.convert_to_dataframe()

        instance_weight_names = std_dataset.\
            __dict__.get('instance_weight_names', None)

        self.complete_df = df
        super(StandardCCDDataset, self).__init__(
            n_samples=len(df), standard=std_dataset, complete_df=df,
            label_names=std_dataset.label_names,
            protected_attribute_names=std_dataset.protected_attribute_names,
            instance_weights_name=instance_weight_names,
            instance_names=std_dataset.instance_names,
            metadata=std_dataset.metadata,
            favorable_label=std_dataset.favorable_label,
            unfavorable_label=std_dataset.unfavorable_label, **kwargs)

    @property
    def privileged_groups(self):
        # TODO: Bad code below. No need to have standard in the class. Rethink.
        return self.standard.privileged_groups

    @property
    def unprivileged_groups(self):
        return self.standard.unprivileged_groups

    def generate_missing_matrix(self, strategy=0, **kwargs):
        rng = kwargs['rng']
        df = self.complete_df
        std_dataset = self.standard
        pa_names = std_dataset.protected_attribute_names
        label_names = std_dataset.label_names
        uic, pic = self.unpriv_ic_prob, self.priv_ic_prob
        strategy_names = {0: 'by_most_corr_col',
                          1: 'by_column',
                          2: 'rand_one_col_by_samples',
                          3: 'rand_many_col_by_samples'}
        strategy_name = strategy_names[strategy]

        if strategy == 1:
            missing_matrix = missing_single_col_by_sample(
                df, uic, pic, pa_names, label_names, rng
            )
        elif strategy == 2:
            missing_matrix = missing_by_top_corr_column(df, uic, pic, pa_names,
                                                        label_names, rng)
        elif strategy == 3:
            missing_matrix = missing_all_cols_by_sample(
                df, uic, pic, pa_names, label_names, rng
            )
        # TODO: Remove the following unused code segment.
        # elif strategy == ??:
        #     col_name = kwargs.get('missing_column_name')
        #     if col_name is None:
        #         raise ValueError("Must define column "
        #                          "for missing by column strategy")
        #     missing_matrix = missing_by_column(
        #         df, uic, pic, pa_names, col_name, rng
        #     )

        return sparse.csr_matrix(missing_matrix)

