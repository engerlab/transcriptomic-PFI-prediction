import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import glob
from sksurv.ensemble import RandomSurvivalForest
from util import t_test, t_test_feature_selection, n_equal_slices, TruSight15, TruSight170, pearson_feature_selection, cox_feature_selection, PCA_feature_selection
from train import regression_evaluate, rsf_train, grid_cross_validate, drop_suffix_from_key, c_index_scorer
from sksurv.util import Surv
import tqdm

def gdc_read_mRNA(mRNA_files, filemap, case_id):
  id_list = []
  data_list = []
  gene_names = np.genfromtxt(mRNA_files[0], delimiter='\t', skip_header=6, usecols=1, unpack=True, dtype=str) # initialize


  # order of genes is the same in every file (if not an Error is raised)
  # load in the gene type for each gene from a random file
  gene_type = np.genfromtxt(mRNA_files[0], delimiter='\t', unpack=True, skip_header=6, usecols=2, dtype=str)

  # find the genes that are not protein coding
  coding = gene_type == "protein_coding"



  for i,file in tqdm.tqdm(enumerate(mRNA_files)):
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

      data = np.genfromtxt(file, delimiter='\t', skip_header=6, usecols=8, unpack=True)
      # store the protein coding genes
      data_list.append(data[coding])

      old_names = gene_names
      gene_names = np.genfromtxt(file, delimiter='\t', skip_header=6, usecols=1, unpack=True, dtype=str)

      if not np.all(old_names == gene_names):
          print(old_names)
          print(gene_names)
          raise ValueError("Order of gene names changed")
      

  gene_names = gene_names[coding]
  

  return id_list, data_list, gene_names

def mRNA_load_data(num=None):
    if num is None:
        mRNA_files = glob.glob('data/mRNA/downloads/primary_tumor/*/*rna_seq*.tsv')
    else:
        # for testing purposed
        mRNA_files = glob.glob('data/mRNA/downloads/primary_tumor/*/*rna_seq*.tsv')[:num]

    filemap = pd.read_csv('data/mRNA/downloads/primary_tumor/mRNA_primary_tumor_sample_sheet.2024-07-22.tsv', header=0, sep='\t')

    # get the PFI labels
    label = np.genfromtxt('labels/PFI.csv', delimiter=',', unpack=True, usecols=[2], skip_header=4)
    time = np.genfromtxt('labels/PFI.csv', delimiter=',', unpack=True, usecols=[3], skip_header=4)
    case_id = np.genfromtxt('labels/PFI.csv', delimiter=',', unpack=True, usecols=[0], dtype=str, skip_header=4)

    # match the labels to the data read from mRNA files
    id_list, data_list, gene_names = gdc_read_mRNA(mRNA_files, filemap, case_id)
    matched_labels = np.array([label[list(case_id).index(id)] if id in case_id else np.nan for id in id_list])
    matched_times = np.array([time[list(case_id).index(id)] if id in case_id else np.nan for id in id_list])
    
    
    
    id_list = np.array(id_list, dtype=str)
    data_list = np.log2(np.array(data_list) + 1)

    is_nan = np.isnan(matched_times)


    return id_list[~is_nan], data_list[~is_nan], gene_names, matched_labels[~is_nan], matched_times[~is_nan]
    


def single_nested_feature_selection(seed):

    feature_selection_algorithms = [TruSight170]
    feature_kwargs = [{},{"K":20},{"K":20},{"alphas":[0.22]},{"K":20},{},{}]
    id_list, data_list, gene_names, matched_labels, matched_times  = mRNA_load_data()
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
            test_bool = np.array([True if i in test_set else False for i in range(len(id_list))], dtype=bool)
            train_bool = ~test_bool
            
            feature_df = feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, **feature_kwarg)
            # feature_df = pearson_feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, 20)
            # feature_df = t_test_feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, 20)
            # feature_df = TruSight170(train_bool, id_list, data_list, gene_names, matched_labels, matched_times)
            important_features += list(feature_df.loc[:, ~feature_df.columns.isin(['case_id', 'label', 'time'])].columns)
            data = feature_df.loc[:, ~feature_df.columns.isin(['case_id', 'label', 'time'])].values
            labels = feature_df['label'].values
            times = feature_df['time'].values
            train_set = data[train_bool]
            train_labels = labels[train_bool]
            train_times = times[train_bool]
            test_set = data[~train_bool]
            test_times = times[~train_bool]
            test_labels = labels[~train_bool]
            print(f"data shape = {data.shape}")
            survival_train = Surv.from_arrays(event=train_labels, time=train_times)
            survival_test = Surv.from_arrays(event=test_labels, time=test_times)
            grid_search = rsf_train(train_set, survival_train)


            res = regression_evaluate(grid_search, test_set, test_labels, test_times)

            result = permutation_importance(grid_search.best_estimator_, test_set, survival_test, n_repeats=10, random_state=42, scoring=c_index_scorer)
            feature_importance += list(result.importances_mean)
            print(res[0])
            c_vals.append(res[0])
        out_cvals.append(c_vals)
    
    feature_res =[[i,j] for i,j in zip(important_features, feature_importance)]
    feature_df = pd.DataFrame(data=feature_res, columns=['feature', 'mean_importance'])
    feature_df.to_csv('data/mRNA_feature_importance.csv')

    print(f"{np.mean(out_cvals, axis=1)} +/- {2*np.std(out_cvals, axis=1)/np.sqrt(5)}")
    return 

def doubly_nested_feature(seed):

    feature_selection_algorithms = [ pearson_feature_selection, t_test_feature_selection, PCA_feature_selection, cox_feature_selection,
                                    TruSight170, TruSight15]
    feature_kwargs = [{},{},{},{},{},{}]
    feature_grids = [{"feature__K":[20,100,400]},{"feature__K":[20,100,400]},{"feature__K":[20,100,150]},{"feature__alphas":[[0.05],[0.10],[0.21]]},{},{}]
    id_list, data_list, gene_names, matched_labels, matched_times  = mRNA_load_data()
    out_cvals = []
    for feature_selection,feature_kwarg,feature_grid in zip(feature_selection_algorithms, feature_kwargs, feature_grids):
        
        print(feature_selection.__name__)
        print('=====================================')

        c_vals = []

        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
        random_order = np.arange(0,len(id_list))
        rs.shuffle(random_order)
        slices = n_equal_slices(len(id_list), 5)

        for bound in slices:

            
            test_set = random_order[bound[0]:bound[1]]
            test_bool = np.array([True if i in test_set else False for i in range(len(id_list))], dtype=bool)
            train_bool = ~test_bool

            
            rsf = RandomSurvivalForest(random_state=42, n_jobs=-1)

            model_grid = {
            "n_estimators": [25,75,200],
            "min_samples_split": [2,10],
            "min_samples_leaf": [1,15,30],
            }

            # concatenate grids
            param_grid = {**model_grid, **feature_grid}


            grid_search = grid_cross_validate(rsf, 
                                              id_list[train_bool], 
                                              data_list[train_bool], 
                                              gene_names, 
                                              matched_labels[train_bool], 
                                              matched_times[train_bool],
                                              feature_selection,
                                              param_grid=param_grid,
                                              )
            
            temp_params = {param_space[9:]:value for param_space,value in grid_search.best_params_.items() if 'feature__' in param_space}
            # temp_params = drop_suffix_from_key(grid_search.best_params_, 'feature__')
            print(grid_search.best_params_)
            feature_df = feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, **feature_kwarg, **temp_params)
            data = feature_df.loc[:, ~feature_df.columns.isin(['case_id', 'label', 'time'])].values
            labels = feature_df['label'].values
            times = feature_df['time'].values

            test_set = data[~train_bool]
            test_times = times[~train_bool]
            test_labels = labels[~train_bool]
            print(f"data shape = {data.shape}")
            res = regression_evaluate(grid_search, test_set, test_labels, test_times)
            print(res[0])
            c_vals.append(res[0])

        
        out_cvals.append(c_vals)

    print(f"{np.mean(out_cvals, axis=1)} +/- {2*np.std(out_cvals, axis=1)/np.sqrt(5)}")
    
    return 

def main():
    single_nested_feature_selection(5)


if __name__ == '__main__':
    
    main()
    # print(f"{np.mean(res, axis=1)} +/- {2*np.std(res, axis=1)/np.sqrt(5)}")
    