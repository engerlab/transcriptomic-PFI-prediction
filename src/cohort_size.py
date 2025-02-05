import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from miRNA_pipe import miRNA_load_data
from clinical_pipe import clinical_load_data
from mRNA_pipe import mRNA_load_data



if __name__ == '__main__':
    mRNA_id, mRNA_data, mRNA_gene_names, mRNA_matched_labels, mRNA_matched_times  = mRNA_load_data()
    miRNA_id, miRNA_data, miRNA_gene_names, miRNA_matched_labels, miRNA_matched_times  = miRNA_load_data()

    clinical_df = clinical_load_data()

    print(clinical_df.shape)
    print(mRNA_id.shape)
    print(miRNA_id.shape)


    mRNA_set = set(mRNA_id)
    miRNA_set = set(miRNA_id)
    clinical_set = set(clinical_df['case_id'].values)


    clin_m = set.intersection(mRNA_set, clinical_set)
    print(len(clin_m))

    clin_mi = set.intersection(miRNA_set, clinical_set)
    print(len(clin_mi))
    

    m_mi = set.intersection(miRNA_set, mRNA_set)
    print(len(m_mi))

    overlap = set.intersection(mRNA_set, miRNA_set, clinical_set)
    print(len(overlap))
