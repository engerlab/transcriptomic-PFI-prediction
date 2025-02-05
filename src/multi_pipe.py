from train import regression_evaluate, rsf_train, mean_predictions, reorder
from sksurv.util import Surv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from miRNA_pipe import miRNA_load_data
from clinical_pipe import clinical_load_data, impute_scale
from mRNA_pipe import mRNA_load_data
from util import n_equal_slices, t_test_feature_selection, TruSight170, pearson_feature_selection
from sksurv.metrics import concordance_index_censored
import tqdm

def keep_overlap(ids, keep_ids, *args) -> list:
    '''function for reshaping ids and associated data contained in args to be compatable with
    another set of ids
    
    :param iterable ids: a list of ids associated to the data in args
    :param iterable keep_ids: a subset of ids that are to be retained
    :param iterable args: a list of lists of the same shape as ids that contains associated data'''
    
    bool_arr = [True if id in keep_ids else False for id in ids]
    
    res = []
    for arg in args:
        res.append(arg[bool_arr])

    return keep_ids[:], *res 


def early_fusion(DataFrames, train_bool) -> list:
    '''Take in all the preprocessed data and perform early fusion and training
    
    :param pd.DataFrame DataFrames: list of DataFrames containing the different datatypes, each dataframe must contain the patient ids, event_labels, and times so they can be merged together
    :param np.ndarray train_bool: A matching boolean array where True indicates a patient in the train set and False a patient in the test set
    :return: returns the sksurv censored_cindex result'''

    combined_data = DataFrames[0]

    for df in DataFrames[1:]:
        
        combined_data = pd.merge(combined_data, df.drop(['time', 'label'], axis=1), on='case_id', how='left')


    data = combined_data.loc[:, ~combined_data.columns.isin(['case_id', 'label', 'time','primary_diagnosis_Squamous cell carcinoma, spindle cell'])].values
    # print(combined_data.loc[:, ~combined_data.columns.isin(['case_id', 'label', 'time','primary_diagnosis_Squamous cell carcinoma, spindle cell'])].columns.tolist())

    labels = combined_data['label'].values
    times = combined_data['time'].values

    train_set = data[train_bool]
    train_labels = labels[train_bool]
    train_times = times[train_bool]
    test_set = data[~train_bool]
    test_times = times[~train_bool]
    test_labels = labels[~train_bool]

    print(train_set.shape)
    print(test_set.shape)

    survival_train = Surv.from_arrays(event=train_labels, time=train_times)

    grid_search = rsf_train(train_set, survival_train)

    res = regression_evaluate(grid_search, test_set, test_labels, test_times)


    return res

def late_fusion(DataFrames, train_bool) -> list:
    '''function to implement late fusion of different data types and return the resulting c-index on the 
    test set, returns the result from sksurv.metrics.concordance_index_censored
    
    :param list DataFrames: a list of pd.DataFrame objects in a preprocessed form, each should contain the preprocessed data and columns 'label', 'time', 'case_id' 
    :param list train_bool: a bool array to split the data into train and test sets'''

    models = []
    shapes = []
    

    combined_data = DataFrames[0]
    # first we need to merged so that we have all the data properly matchedup
    for df in DataFrames[1:]:
        combined_data = pd.merge(combined_data, df.drop(['time', 'label'], axis=1), on='case_id', how='left')

    for df in tqdm.tqdm(DataFrames):

        data = df.loc[:, ~df.columns.isin(['case_id', 'label', 'time','primary_diagnosis_Squamous cell carcinoma, spindle cell'])].values
        labels = df['label'].values
        times = df['time'].values

        train_set = data[train_bool]
        train_labels = labels[train_bool]
        train_times = times[train_bool]
        

        survival_train = Surv.from_arrays(event=train_labels, time=train_times)
        grid_search = rsf_train(train_set, survival_train)


        

        shapes.append(data.shape[1])
        models.append(grid_search.best_estimator_)
        
   
    # extract all matched data for testing
    data = combined_data.loc[:, ~combined_data.columns.isin(['case_id', 'label', 'time','primary_diagnosis_Squamous cell carcinoma, spindle cell'])].values
    test_set = data[~train_bool]
    test_times = times[~train_bool]
    test_labels = labels[~train_bool]
    
    print(train_set.shape)
    print(test_set.shape)

    risk_scores = mean_predictions(models, test_set, shapes)
    c_index = concordance_index_censored(test_labels == 1, test_times, risk_scores)
    
    return c_index

def load_all() -> dict:
    '''
    function to load and reorder the data so that the cases in each dataframe appear in the same order from the three data modalities
    
    :return: dictionary of the three datatypes
    '''

    mRNA_id, mRNA_data, mRNA_gene_names, mRNA_matched_labels, mRNA_matched_times  = mRNA_load_data()
    miRNA_id, miRNA_data, miRNA_gene_names, miRNA_matched_labels, miRNA_matched_times  = miRNA_load_data()

    clinical_df = clinical_load_data()

    mRNA_set = set(mRNA_id)
    miRNA_set = set(miRNA_id)
    clinical_set = set(clinical_df['case_id'].values)


    overlap = set.intersection(mRNA_set, miRNA_set, clinical_set)
    
    # drop cases and data not overlapping, 
    mRNA_id, mRNA_data, mRNA_matched_labels, mRNA_matched_times = keep_overlap(mRNA_id, np.array(list(overlap)), mRNA_data, mRNA_matched_labels, mRNA_matched_times)
    miRNA_id, miRNA_data, miRNA_matched_labels, miRNA_matched_times = keep_overlap(miRNA_id, np.array(list(overlap)), miRNA_data, miRNA_matched_labels, miRNA_matched_times)
    _, clinical_df = keep_overlap(clinical_df['case_id'].values, list(overlap), clinical_df)

    # match the case order so that all the train and set tests are consistent between data types
    ref_order = clinical_df['case_id'].values

    mRNA_reorder = reorder(mRNA_id, ref_order)
    mRNA_id = mRNA_id[mRNA_reorder]
    mRNA_data = mRNA_data[mRNA_reorder]
    mRNA_matched_labels = mRNA_matched_labels[mRNA_reorder]
    mRNA_matched_times = mRNA_matched_times[mRNA_reorder]

    
    miRNA_reorder = reorder(miRNA_id, ref_order)
    miRNA_id = miRNA_id[miRNA_reorder]
    miRNA_data = miRNA_data[miRNA_reorder]
    miRNA_matched_labels = miRNA_matched_labels[miRNA_reorder]
    miRNA_matched_times = miRNA_matched_times[miRNA_reorder]

    res_dict = {'mirna': [miRNA_id, miRNA_data, miRNA_gene_names, miRNA_matched_labels, miRNA_matched_times],
                'mrna': [mRNA_id, mRNA_data, mRNA_gene_names, mRNA_matched_labels, mRNA_matched_times],
                'clinical': clinical_df}
    
    return res_dict


def multi_training(data_dict, seed: int, fusion_type: str, datatypes: list) -> list:
    
    
    miRNA_id, miRNA_data, miRNA_gene_names, miRNA_matched_labels, miRNA_matched_times = data_dict['mirna']
    mRNA_id, mRNA_data, mRNA_gene_names, mRNA_matched_labels, mRNA_matched_times = data_dict['mrna']
    clinical_df = data_dict['clinical']
    
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    random_order = np.arange(0,len(clinical_df))
    rs.shuffle(random_order)
    slices = n_equal_slices(len(clinical_df), 5)

    c_vals = []

    for bound in slices:

        
        test_set = random_order[bound[0]:bound[1]]
        

        test_bool = np.array([True if i in test_set else False for i in range(len(random_order))], dtype=bool)
        train_bool = ~test_bool
        


        mRNA_df = TruSight170(train_bool, mRNA_id, mRNA_data, mRNA_gene_names, mRNA_matched_labels, mRNA_matched_times)
        miRNA_df = pearson_feature_selection(train_bool, miRNA_id, miRNA_data, miRNA_gene_names, miRNA_matched_labels, miRNA_matched_times, 100)
        scaled = impute_scale(clinical_df.loc[:, ~clinical_df.columns.isin(['case_id', 'label', 'time'])], train_bool)
        print(mRNA_df.shape)
        print(miRNA_df.shape)
        print(scaled.shape)
        # print(clinical_df[['case_id', 'label', 'time']].values.shape)

        labeled = np.concatenate((scaled, clinical_df[['case_id', 'label', 'time']].values), axis=1)
        clinical_df = pd.DataFrame(data=labeled, columns=clinical_df.columns)

        processed_data = {'mrna': mRNA_df, 'mirna': miRNA_df, 'clinical': clinical_df}

        arg = []
        for datatype in datatypes:
            arg.append(processed_data[datatype])

        if fusion_type == 'early':
            res = early_fusion(arg, train_bool)
        if fusion_type == 'late':
            res = late_fusion(arg, train_bool)
        print(res[0])
        c_vals.append(res[0])

    return c_vals

def main():
    seed = 5
    fusion_type = 'late'
    combinations = [['clinical', 'mrna'],['clinical', 'mirna'], ['clinical', 'mirna', 'mrna'], ['mirna', 'mrna']]

    data_dict = load_all()

    for datatypes in combinations:
        print(datatypes)
        res = multi_training(data_dict, seed, fusion_type, datatypes)
        print(f"{np.mean(res)} +/- {2*np.std(res)/np.sqrt(5)}")
        print("======================================================")

if __name__ == '__main__':
    main()
    