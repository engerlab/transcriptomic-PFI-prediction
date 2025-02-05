import numpy as np
import pandas as pd
import tqdm
import glob
from util import t_test_feature_selection, n_equal_slices, cox_feature_selection, PCA_feature_selection, pearson_feature_selection
from train import regression_evaluate, rsf_train, c_index_scorer
from sklearn.inspection import permutation_importance
from sksurv.util import Surv

def miRNA_load_data():
    miRNA_files = glob.glob('data/miRNA/downloads/primary_tumor/*/*mirnas*.txt')
    filemap = pd.read_csv('data/miRNA/downloads/primary_tumor/miRNA_primary_sample_sheet.tsv', header=0, sep='\t')
    
    # get the PFI labels
    label = np.genfromtxt('labels/PFI.csv', delimiter=',', unpack=True, usecols=[2], skip_header=4)
    time = np.genfromtxt('labels/PFI.csv', delimiter=',', unpack=True, usecols=[3], skip_header=4)
    case_id = np.genfromtxt('labels/PFI.csv', delimiter=',', unpack=True, usecols=[0], dtype=str, skip_header=4)

    # match the labels to the data read from miRNA files
    id_list, data_list, gene_names = gdc_read_miRNA(miRNA_files, filemap, case_id)
    matched_labels = np.array([label[list(case_id).index(id)] if id in case_id else np.nan for id in id_list])
    matched_times = np.array([time[list(case_id).index(id)] if id in case_id else np.nan for id in id_list])
    
    
    
    id_list = np.array(id_list, dtype=str)
    data_list = np.log2(np.array(data_list) + 1)

    is_nan = np.isnan(matched_times)


    return id_list[~is_nan], data_list[~is_nan], gene_names, matched_labels[~is_nan], matched_times[~is_nan]


def gdc_read_miRNA(miRNA_files, filemap, case_id):
    id_list = []
    data_list = []
    gene_names = np.genfromtxt(miRNA_files[0], delimiter='\t', skip_header=1, usecols=0, unpack=True, dtype=str) # initialize

    for i,file in tqdm.tqdm(enumerate(miRNA_files)):
        try:
            id = filemap.loc[filemap['File Name'] == file.split('/')[-1]]['Case ID'].values[0]
        except:
            print(file)
            filemap.loc[filemap['File Name'] == file.split('/')[-1]]['Case ID'].values
            print('unmapped file')
            break

        # multiple files per some patients, just keep the first one or patient not in the 2yr survival set 
        if id in id_list or id not in case_id:
            continue

        

        id_list.append(id)

        data = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=2, unpack=True)
        # store the protein coding genes
        data_list.append(data)

        old_names = gene_names
        gene_names = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=0, unpack=True, dtype=str)

        if not np.all(old_names == gene_names):
            print(old_names)
            print(gene_names)
            raise ValueError("Order of gene names changed")
        
    return id_list, data_list, gene_names
    
def main(seed):
    feature_selection_algorithms = [pearson_feature_selection]
    feature_kwargs = [{"K":100}]
    id_list, data_list, gene_names, matched_labels, matched_times  = miRNA_load_data()
    out_cvals = []
    important_features = []
    feature_importance = []

    for feature_selection,feature_kwarg in zip(feature_selection_algorithms, feature_kwargs):
        c_vals = []

        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
        random_order = np.arange(0,len(id_list))
        rs.shuffle(random_order)
        slices = n_equal_slices(len(id_list), 5)
        

        
        for bound in slices:

            
            
            test_set = random_order[bound[0]:bound[1]]
            print(test_set.shape)

            test_bool = np.array([True if i in test_set else False for i in range(len(id_list))], dtype=bool)
            train_bool = ~test_bool

            


            # feature_df = t_test_feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, 100)
            feature_df = feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, **feature_kwarg)
            important_features += list(feature_df.loc[:, ~feature_df.columns.isin(['case_id', 'label', 'time'])].columns)
            data = feature_df.loc[:, ~feature_df.columns.isin(['case_id', 'label', 'time'])].values
            print(f"data shape = {data.shape}")
            labels = feature_df['label'].values
            times = feature_df['time'].values

            train_set = data[train_bool]
            train_labels = labels[train_bool]
            train_times = times[train_bool]
            test_set = data[~train_bool]
            test_times = times[~train_bool]
            test_labels = labels[~train_bool]

            survival_train = Surv.from_arrays(event=train_labels, time=train_times)
            survival_test = Surv.from_arrays(event=test_labels, time=test_times)
            grid_search = rsf_train(train_set, survival_train)

            res = regression_evaluate(grid_search, test_set, test_labels, test_times)
            result = permutation_importance(grid_search.best_estimator_, test_set, survival_test, n_repeats=10, random_state=42, scoring=c_index_scorer)
            feature_importance += list(result.importances_mean)
            print(res[0])
            c_vals.append(res[0])
        print('==========================')
        out_cvals.append(c_vals)

    feature_res =[[i,j] for i,j in zip(important_features, feature_importance)]
    feature_df = pd.DataFrame(data=feature_res, columns=['feature', 'mean_importance'])
    feature_df.to_csv('data/miRNA_feature_importance.csv')

    return out_cvals


if __name__ == '__main__':

    res = main(5)

    print(f"{np.mean(res, axis=1)} +/- {2*np.std(res, axis=1)/np.sqrt(5)}")
