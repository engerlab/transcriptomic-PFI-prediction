import numpy as np
from scipy import stats, optimize
import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mrmr import mrmr_classif
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.decomposition import PCA


# Function to remove outliers using the IQR method not in use
def remove_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    if IQR == 0:
       return series
    
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR
    return series.loc[(series >= lower_bound) & (series <= upper_bound)]

def PCA_feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, K=20) -> pd.DataFrame:
    df = pd.DataFrame(data_list, columns=gene_names)

    
    scaler = StandardScaler()
    scaler.fit(df[train_bool].to_numpy())
    scaled_df = pd.DataFrame(scaler.transform(df.to_numpy()), columns=gene_names)
    pca = PCA(n_components=K)
    n_features = scaled_df.columns.shape[0]
    pca.fit(scaled_df[train_bool].values)

    feature_selected = pca.transform(scaled_df.values)
    
    pca_feature_selected = pd.DataFrame(data=feature_selected)
    kwarg = {'case_id':id_list, 'label':matched_labels, 'time':matched_times}
    pca_feature_selected = pca_feature_selected.assign(**kwarg)
    return pca_feature_selected

def pearson_feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, K=20) -> pd.DataFrame:
    '''
    Feature selection using pearson test to find correlations with survival time

    :param np.ndarray train_bool: N length boolean array where True indicates a sample in the train cohort and False indicates a sample in the test cohort
    :param np.ndarray id_list: N length array of patient barcodes
    :param np.ndarray data_list; (N,M) length array containing the N samples and there corresponding gene expressions
    :param np.ndarray gene_names: M length array containing the corresponding gene_names for the data_list
    :param np.ndarray matched_labels: N length array containing 1 if the patient experienced an event and 0 if not
    :param np.ndarray matched_times: N length array containing the time to event or censorship for each patient
    :param np.ndarray int K: the top K features to take
    :return: return the feature selected DataFrame with ids, matched_labels, and matched_times included
    :rtype: pd.DataFrame
    '''

    df = pd.DataFrame(data_list, columns=gene_names)

    
    scaler = StandardScaler()
    scaler.fit(df[train_bool].to_numpy())
    scaled_df = pd.DataFrame(scaler.transform(df.to_numpy()), columns=gene_names)

    n_features = scaled_df.columns.shape[0]

    # keep only those that experience an event
    keep = matched_labels[train_bool] == 1
    reshaped_arr = matched_times.reshape(len(matched_times),1)
    p_scores = stats.pearsonr(scaled_df.values[train_bool][keep], reshaped_arr[train_bool][keep], axis=0).pvalue
    # for feature_index in range(n_features-1):
    #     feature_population = scaled_df.iloc[:, feature_index].values
    #     p_score = stats.pearsonr(feature_population[keep], matched_times[keep]).pvalue
    #     p_scores.append(p_score)
    # argsort from lowest p value to highest
    order = np.argsort(p_scores)
    # select the top 20 features
    top = order[:K]

    pearson_feature_selected = scaled_df.iloc[:, top]

    kwarg = {'case_id':id_list, 'label':matched_labels, 'time':matched_times}
    pearson_feature_selected = pearson_feature_selected.assign(**kwarg)
 
    return pearson_feature_selected


def cox_feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, **kwargs) -> pd.DataFrame:
    '''
    Feature selection using Coxnet to find hazardous features

    :param np.ndarray train_bool: N length boolean array where True indicates a sample in the train cohort and False indicates a sample in the test cohort
    :param np.ndarray id_list: N length array of patient barcodes
    :param np.ndarray data_list; (N,M) length array containing the N samples and there corresponding gene expressions
    :param np.ndarray gene_names: M length array containing the corresponding gene_names for the data_list
    :param np.ndarray matched_labels: N length array containing 1 if the patient experienced an event and 0 if not
    :param np.ndarray matched_times: N length array containing the time to event or censorship for each patient
    :param np.ndarray int K: the top K features to take
    :return: return the feature selected DataFrame with ids, matched_labels, and matched_times included
    :rtype: pd.DataFrame
    '''

    df = pd.DataFrame(data_list, columns=gene_names)
    scaler = StandardScaler()
    scaler.fit(df[train_bool].to_numpy())
    scaled_df = pd.DataFrame(scaler.transform(df.to_numpy()), columns=gene_names)

    missing = matched_times[train_bool] < 0

    # Convert training labels to the format required by scikit-survival
    survival_train = np.array([(event, time) for event, time in zip(matched_labels[train_bool][~missing], matched_times[train_bool][~missing])],
                            dtype=[('event', '?'), ('time', '<f8')])
    
    if 'alphas' not in kwargs.keys():
        kwargs['alphas'] = [0.2]

    coxnet = CoxnetSurvivalAnalysis(**kwargs)  
    coxnet.fit(scaled_df[train_bool][~missing].values, survival_train)
    retained_indices = np.abs(coxnet.coef_) > 0
    
    cox_feature_selected = scaled_df.iloc[:, retained_indices]

    kwarg = {'case_id':id_list, 'label':matched_labels, 'time':matched_times}
    cox_feature_selected = cox_feature_selected.assign(**kwarg)
    return cox_feature_selected

# code taken from https://bitbucket.org/GowriSrinivasa/analysisofrna-seqdata/src/master/
def t_test(controls_feature_population, cases_feature_population, mudiff=0, outliers=True):
  #controls_feature_population = controls_feature_population.transform(lambda x: np.log(x+1)/np.log(2))
  #cases_feature_population = cases_feature_populationtransform(lambda x: np.log(x+1)/np.log(2))

  controls_feature_std = controls_feature_population.std()
  cases_feature_std = cases_feature_population.std()

  if not outliers:
    controls_feature_population = remove_outliers(controls_feature_population)
    cases_feature_population = remove_outliers(cases_feature_population)

  controls_feature_mean = controls_feature_population.mean()
  cases_feature_mean = cases_feature_population.mean()

  controls_feature_counts = controls_feature_population.count()
  cases_feature_counts = cases_feature_population.count()

  pooledSE = np.sqrt(controls_feature_std**2/controls_feature_counts + cases_feature_std**2/cases_feature_counts)
  z = ((controls_feature_mean - cases_feature_mean) - mudiff)/pooledSE
  pval = 2*(1 - stats.norm.cdf(abs(z)))
  
  return pval


def t_test_feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, K=20) -> pd.DataFrame:
    '''
    Feature selection using the statistical t-test on the matched_labels

    :param np.ndarray train_bool: N length boolean array where True indicates a sample in the train cohort and False indicates a sample in the test cohort
    :param np.ndarray id_list: N length array of patient barcodes
    :param np.ndarray data_list; (N,M) length array containing the N samples and there corresponding gene expressions
    :param np.ndarray gene_names: M length array containing the corresponding gene_names for the data_list
    :param np.ndarray matched_labels: N length array containing 1 if the patient experienced an event and 0 if not
    :param np.ndarray matched_times: N length array containing the time to event or censorship for each patient
    :param np.ndarray int K: the top K features to take
    :return: return the feature selected DataFrame with ids, matched_labels, and matched_times included
    :rtype: pd.DataFrame
    '''

    df = pd.DataFrame(data_list, columns=gene_names)

    
    scaler = StandardScaler()
    scaler.fit(df[train_bool].to_numpy())
    scaled_df = pd.DataFrame(scaler.transform(df.to_numpy()), columns=gene_names)
    
    controls = scaled_df[train_bool].loc[matched_labels[train_bool] == 1, :]
    cases = scaled_df[train_bool].loc[matched_labels[train_bool] == 0, :]

    # n_features = scaled_df.columns.shape[0]


    # p_scores = []
    # for feature_index in range(n_features-1):
    #     controls_feature_population = controls.iloc[:, feature_index]
    #     cases_feature_population = cases.iloc[:, feature_index]
    #     p_score = t_test(controls_feature_population, cases_feature_population)
    #     p_scores.append(p_score)
    p_scores = stats.ttest_ind(controls.values, cases.values, axis=0, equal_var=False).pvalue
    # argsort from lowest p value to highest
    order = np.argsort(p_scores)
    # select the top 20 features
    top = order[:K]

    t_test_feature_selected = scaled_df.iloc[:, top]

    kwarg = {'case_id':id_list, 'label':matched_labels, 'time':matched_times}
    t_test_feature_selected = t_test_feature_selected.assign(**kwarg)
 
    return t_test_feature_selected


def mrmr_feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times, K=20) -> pd.DataFrame:
    '''
    Feature selection using the mrmr on the matched_labels

    :param np.ndarray train_bool: N length boolean array where True indicates a sample in the train cohort and False indicates a sample in the test cohort
    :param np.ndarray id_list: N length array of patient barcodes
    :param np.ndarray data_list; (N,M) length array containing the N samples and there corresponding gene expressions
    :param np.ndarray gene_names: M length array containing the corresponding gene_names for the data_list
    :param np.ndarray matched_labels: N length array containing 1 if the patient experienced an event and 0 if not
    :param np.ndarray matched_times: N length array containing the time to event or censorship for each patient
    :param np.ndarray int K: the top K features to take
    :return: return the feature selected DataFrame with ids, matched_labels, and matched_times included
    :rtype: pd.DataFrame
    '''


    df = pd.DataFrame(data_list, columns=gene_names)
    keep = df.nunique().values > 1 

    df = df.loc[:, keep]
    gene_names=gene_names[keep]

    Y = pd.Series(matched_labels)

    scaler = StandardScaler()
    scaler.fit(df[train_bool].to_numpy())
    scaled_df = pd.DataFrame(scaler.transform(df.to_numpy()), columns=gene_names)

    selected_features = mrmr_classif(X=scaled_df[train_bool], y=Y[train_bool], K=K)
    feature_selected = scaled_df.loc[:, scaled_df.columns.isin(selected_features)]

    kwarg = {'case_id':id_list, 'label':matched_labels, 'time':matched_times}
    feature_selected = feature_selected.assign(**kwarg)


    return feature_selected


def TruSight15(train_bool, id_list, data_list, gene_names, matched_labels, matched_times) -> pd.DataFrame:
    '''
    Feature selection using the TruSight15 gene set from illumina

    :param np.ndarray train_bool: N length boolean array where True indicates a sample in the train cohort and False indicates a sample in the test cohort
    :param np.ndarray id_list: N length array of patient barcodes
    :param np.ndarray data_list; (N,M) length array containing the N samples and there corresponding gene expressions
    :param np.ndarray gene_names: M length array containing the corresponding gene_names for the data_list
    :param np.ndarray matched_labels: N length array containing 1 if the patient experienced an event and 0 if not
    :param np.ndarray matched_times: N length array containing the time to event or censorship for each patient
    :return: return the feature selected DataFrame with ids, matched_labels, and matched_times included
    :rtype: pd.DataFrame
    '''

    df = pd.DataFrame(data_list, columns=gene_names)
    scaler = StandardScaler()
    scaler.fit(df[train_bool].to_numpy())
    scaled_df = pd.DataFrame(scaler.transform(df.to_numpy()), columns=gene_names)

    trusight15 = ['AKT1', 'BRAF', 'EGFR', 'ERBB2', 'FOXL2', 'GNA11', 'GNAQ', 'KIT', 'KRAS',
                  'MET', 'NRAS', 'PDGFRA', 'PIK3CA', 'RET', 'TP53']
    
    feature_selected = scaled_df.loc[:, scaled_df.columns.isin(trusight15)]
    kwarg = {'case_id':id_list, 'label':matched_labels, 'time':matched_times}
    feature_selected = feature_selected.assign(**kwarg)

    return feature_selected

def TruSight170(train_bool, id_list, data_list, gene_names, matched_labels, matched_times) -> pd.DataFrame:
    '''
    Feature selection using the TruSight170 RNA gene set from illumina

    :param np.ndarray train_bool: N length boolean array where True indicates a sample in the train cohort and False indicates a sample in the test cohort
    :param np.ndarray id_list: N length array of patient barcodes
    :param np.ndarray data_list; (N,M) length array containing the N samples and there corresponding gene expressions
    :param np.ndarray gene_names: M length array containing the corresponding gene_names for the data_list
    :param np.ndarray matched_labels: N length array containing 1 if the patient experienced an event and 0 if not
    :param np.ndarray matched_times: N length array containing the time to event or censorship for each patient
    :return: return the feature selected DataFrame with ids, matched_labels, and matched_times included
    :rtype: pd.DataFrame
    '''

    df = pd.DataFrame(data_list, columns=gene_names)
    scaler = StandardScaler()
    scaler.fit(df[train_bool].to_numpy())
    scaled_df = pd.DataFrame(scaler.transform(df.to_numpy()), columns=gene_names)

    trusight170 = ['AKT2','BRCA2','CHEK1','ERCC2','FGF5','FGF14','FGFR4','MDM4','NRG1','RAF1',
                  'ALK','CCND1','CHEK2','ESR1','FGF6','FGF19','JAK2','MET','PDGFRA','RET',
                  'AR','CCND3','EGFR','FGF1','FGF7','FGF23','KIT','MYC','PDGFRB','RICTOR',
                  'ATM','CCNE1','ERBB2','FGF2','FGF8','FGFR1','KRAS','MYCL1','PIK3CA','RPS6KB1',
                  'BRAF','CDK4','ERBB3','FGF3','FGF9','FGFR2','LAMP1','MYCN','PIK3CB','TFRC',
                  'BRCA1','CDK6','ERCC1','FGF4','FGF10','FGFR3','MDM2','NRAS','PTEN']
    
    feature_selected = scaled_df.loc[:, scaled_df.columns.isin(trusight170)]
    kwarg = {'case_id':id_list, 'label':matched_labels, 'time':matched_times}
    feature_selected = feature_selected.assign(**kwarg)

    return feature_selected


def storey_qvalue(p_values, lambda_val=0.5):
    """
    Compute q-values using Storey's method.
    
    :param np.ndarray p_values: List of p-values (unordered).
    :param float lambda_val: A value between 0 and 1 to estimate the false positive rate pi_0 (default = 0.5).
    :returns: Numpy array of q-values corresponding to the input p-values.
    """
    # Convert the list of p-values to a numpy array if it's not already
    p_values = np.array(p_values)
    
    # Step 1: Sort the p-values
    sorted_pvals = np.sort(p_values)
    sorted_index = np.argsort(p_values)
    
    # Step 2: Estimate pi_0
    pi_0 = np.mean(p_values > lambda_val) / (1 - lambda_val)
    
    # Step 3: Compute the q-values
    m = len(p_values)  # Total number of tests
    q_values = np.zeros(m)
    min_qval = 1.0
    
    for i in range(m-1, -1, -1):
        pval = sorted_pvals[i]
        qval = pi_0 * pval * m / (i + 1)
        min_qval = min(min_qval, qval)
        q_values[i] = min_qval
    
    # Step 4: Reorder the q-values to match the original p-value order
    q_values = q_values[np.argsort(sorted_index)]
    
    return q_values

def storey_q_feature_selection(controls, cases, dataset, lambda_val, q_threshold):
  n_features = len(dataset.columns)
  pvals = []
  retain_features = []
  
  for feature_index in tqdm.tqdm(range(n_features-1)):
    try:
      controls_feature_population = controls.iloc[:, feature_index]
      cases_feature_population = cases.iloc[:, feature_index]
      p_score = t_test(controls_feature_population, cases_feature_population)
      pvals.append(p_score)
      retain_features.append(feature_index)
    except Exception as e:
      print(e)
      break

  q_values = storey_qvalue(pvals, lambda_val=lambda_val)
  # done this way so that an error in the for loop is robust to the changing size of q_values array
  retained_indices = np.array(retain_features)[q_values < q_threshold]

  dataset_feature_selected = dataset.iloc[:, retained_indices]
  
  return dataset_feature_selected





def copeland_method(votes, names):
    # Number of candidates (representatives)
    n_candidates = len(names)

    # Initialize win and draw counts for each candidate
    score = np.zeros(n_candidates)

    # Compare each candidate against every other candidate
    for i in range(n_candidates):
        for j in range(i + 1, n_candidates):
            votes_for_i, votes_for_j = 0, 0
            # Compare how many voters prefer candidate i over candidate j
            for vote in votes:
                
                if names[i] in vote:
                   rank1 = vote.index(names[i])
                else:
                   rank1 = len(vote) + 1

                if names[j] in vote:
                   rank2 = vote.index(names[j])
                else:
                   rank2 = len(vote) + 1
                if  rank1 < rank2:
                    votes_for_i += 1
                else:
                    votes_for_j += 1
            # Update scores based on the comparison results
            if votes_for_i > votes_for_j:
                score[i] += 1  # Candidate i wins
            elif votes_for_j > votes_for_i:
                score[j] += 1  # Candidate j wins
            else:
                score[i] += 0.5  # Tie, both get half a point
                score[j] += 0.5

    # Rank candidates by score (higher is better)
    ranking = np.argsort(-score)
    result = ranking.tolist()

    return result

def label_patients(ids, statuses, dtfs, dtds, causes, cutoff):
    '''
    function to label patients for a classification problem, patients who die before the cutoff are labeled as 1
    and patients who die after the cutoff are labeled as 0. Patients who are censored before the cutoff are dropped 
    from the set. 

    :param list ids: unique ids for each patient, in our case this is usually case_submitter_id
    :param list status: the vital status of the patients at time of last followup
    :param list dtf: the days to last follow up of that patient
    :param list dtd: the days to death of the patient, if patient did not die it should be nan
    :param list cause: the cause of death, right now this is not being used
    :param int cutoff: the time in days that is used as the cutoff for classification
    '''


    labels = []
    days_to_event = []
    censoring = []
    # going to start by ignoring the censored data
    # so patients that have vital status of alive but their last follow up was less than two years we will remove them from training and testing
    # we lose ~130 cases this way
    for i,id in enumerate(ids):
        
        # row = combined_data.loc[combined_data['case_submitter_id'] == id, ['vital_status', 'days_to_last_follow_up', 'days_to_death', 'patient_death_reason']]
        # print(combined_data.loc[combined_data['case_id'] == id, ['vital_status', 'days_to_last_follow_up', 'days_to_death']])
        # try:
        #     status = row['vital_status'].item()
        #     dtd = row['days_to_death'].item()
        #     dtf = row['days_to_last_follow_up'].item()
        #     cause = row['patient_death_reason'].item()
        # except KeyError:
        #     print(row)
        #     break

        # if cause == 'Other non-malignant disease':
        #     continue
        status = statuses[i]
        dtd = dtds[i]
        dtf = dtfs[i]
        

        if status == 'Dead' and dtd < cutoff:
            labels.append((id, 1, dtd))
            days_to_event.append(dtd)
            censoring.append(1)

        elif status == 'Dead' and (dtd > cutoff):
            labels.append((id, 0, dtd))
            days_to_event.append(dtd)
            censoring.append(1)

        elif status == 'Alive' and dtf > cutoff:
            labels.append((id, 0, dtf))
            days_to_event.append(dtf)
            censoring.append(0)

        elif status == 'Alive' and dtf < cutoff:
            days_to_event.append(dtf)
            censoring.append(0)

        else:
            days_to_event.append(np.nan)
            censoring.append(0)

    return labels, days_to_event, censoring

def overlap(hist1, hist2, bin_width) -> float:
    '''calculate the overlap between two discrete probability distributions
    
    :param np.ndarray hist1: array of histogram heights
    :param np.ndarray hist2: array of histogram heights
    :param np.ndarray bin_width: the bin widths, this normalizes the answer to 1
    '''

    return np.sum(np.min(np.array([hist1,hist2]), axis=0)*bin_width)

def overlap_ranking(cases, controls):

    n_features = len(controls.columns)
    similarities = []

    for feature_index in tqdm.tqdm(range(n_features-1)):
        controls_feature = controls.iloc[:, feature_index].values
        cases_feature = cases.iloc[:, feature_index].values

        min_feature = np.min(np.concatenate((cases_feature, controls_feature)))
        max_feature = np.max(np.concatenate((cases_feature, controls_feature)))

        hist1, bin_edges = np.histogram(controls_feature, range=(min_feature, max_feature), bins=len(controls_feature)//15, density=True)
        hist2, bin_edges = np.histogram(cases_feature, range=(min_feature, max_feature), bins=len(controls_feature)//15, density=True)
        

        similarity = overlap(hist1, hist2, bin_edges[1:] - bin_edges[:-1])
        similarities.append(similarity)

    return similarities

def Available_Data(data):    
    series_names = []
    available_data = []
    for series_name, series in data.items():
        series_names.append(series_name)
        available_data.append(100 - 100*series.isna().sum()/(series.notna().sum() + series.isna().sum()))

    available_data = np.array(available_data)
    series_names = np.array(series_names)
    # print(series_names)

    i_order = np.argsort(available_data)[::-1]
    available_data = available_data[i_order]
    series_names = series_names[i_order]
    return available_data, series_names

def n_equal_slices(length, N) -> list:
    '''
    return N equal slices for a list of a given length 

    :param int length: the number of samples to be split
    :param int N: the number of subgroups to create
    :return: returns a list of slices of shape (N,2) each slices selects a subgroup
    '''


    slices = []
    slice_width = length//N
    for i in range(N):
        if i != N-1:
            slices.append((i*slice_width, (i+1)*slice_width))
        else:
            slices.append((i*slice_width, length))

    return slices

# used for 2 year classification
def log_rank_split(threshold, labels, times, risk_scores):
    '''calculate the log-rank score between two groups split according to a risk threshold

    :param float threshold: threshold of risk for splitting the patients into two groups
    :param pd.dataframe patients: dataframe of patients containing their time to event info under the label 'days_to_event' and censorship status under 'censored', 1 being uncensored and 0 being right censored
    :param np.ndarray risk_scores: an array of the predicted risk scores for each patient
    '''
    

    labels1, times1 = labels[risk_scores>=threshold], times[risk_scores>=threshold]
    labels2, times2 = labels[risk_scores<threshold], times[risk_scores<threshold]
    
    X = stats.CensoredData(
    uncensored = times1[labels1==1],
    right = times1[labels1==0]
    )
    Y = stats.CensoredData(
        uncensored = times2[labels2==1],
    right = times2[labels2==0]
    )

    return stats.logrank(x=X, y=Y).pvalue

def optimal_risk_split(labels, times, risk_scores):

    min_risk = np.min(risk_scores)
    max_risk = np.max(risk_scores)
    median_risk = np.median(risk_scores)
    print(median_risk)
    threshold = optimize.minimize_scalar(log_rank_split, median_risk, args=(labels, times, risk_scores), bounds=[min_risk,max_risk], tol=1e-6)
    
    
    return threshold.x

def kaplan_splitting(train_labels, train_times, train_risk_scores, test_labels, test_times, test_risk_scores):

    patients=pd.DataFrame(data=np.array([test_labels, test_times, test_risk_scores]).T, columns=['censored', 'days_to_event', 'risk'])
    threshold = optimal_risk_split(train_labels, train_times, train_risk_scores)

    group1 = patients.loc[patients['risk'] >= threshold]
    group2 = patients.loc[patients['risk'] < threshold]

    # xgroup1, sgroup1, cgroup1 = kaplan_meier_estimator(group1['censored']==1, group1['days_to_event'], conf_type='log-log')
    # xgroup2, sgroup2, cgroup2 = kaplan_meier_estimator(group2['censored']==1, group2['days_to_event'], conf_type='log-log')
    # pvalue = log_rank_split(threshold, test_labels, test_times, test_risk_scores)

    # ret_df
    return group1, group2


if __name__ == '__main__':

    features1 = np.array([[1, 2, 3], [0.75, 3, 6], [0.5,3,5,]])
    features2 = np.array([[1, 4, 3], [0.75, 5, 6], [0.5,3,7,]])
    times = np.array([[1],[2],[3]])
    # print(times.shape)
    # print(features1.shape)
    # print(stats.ttest_ind(features1, features2, axis=0, equal_var=False).pvalue)
    # print(stats.pearsonr(features1, times, axis=0))
    np.random.seed(0)
    censored = np.random.randint(0,2, size=100)
    risk_score = np.random.random(100)
    survival_time = np.random.random(100)
    data = np.array([censored, survival_time]).T

    print(optimal_risk_split(censored, survival_time, risk_score))

