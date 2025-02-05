import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sksurv.ensemble import RandomSurvivalForest
import warnings 
import datetime
import json
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sksurv.metrics import concordance_index_censored
from dataclasses import dataclass
import plotly.graph_objects as go
from scipy import stats
from util import n_equal_slices
from sksurv.util import Surv
import itertools

# used for classification, not in use anymore
@dataclass
class TestResult:
    '''class for storing accuracy, F1, precision, recall, and support of the test cohort'''
    F1: float
    test_accuracy: float
    train_score: float
    train_accuracy: float
    precision: float
    recall: float
    auc: float
    support: list
    cov: np.ndarray
    

    def __str__(self):
        
        if self.cov is None:
            self.cov = np.zeros((10,10))

        err = 2 * np.sqrt(np.diag(self.cov)) / np.sqrt(len(self.cov))

        # the order of the indexing is determined by the order of cov in train_with_seeds
        st = ''
        st += f"mean training score = {self.train_score} +/- {err[0]}\n"
        st += f"mean training accuracy = {self.train_accuracy} +/- {err[9]}\n"
        st += f"mean test accuracy = {self.test_accuracy} +/- {err[1]}\n"
        st += f"test AUC = {self.auc} +/- {err[8]}\n"
        st += f"F1: {self.F1} +/- {err[[2,3]]}\n"
        st += f"precision: {self.precision} +/- {err[[4,5]]}\n"
        st += f"recall: {self.recall} +/- {err[[6,7]]}\n"
        st += f"support: {self.support}"

        return st

# this was used when training 2 year classification but since moving to Regression has been abandoned
def train_with_seeds(model, training, testing, param_grid=None, search_method=GridSearchCV, seeds=None, model_kwargs={}, search_kwargs={}, verbose=False):
    '''
    function to train models with multiple seeds and return the errors and mean performance metric. 
    
    :param sklearn.model model: the sklearn model to be trained
    :param tuple training: a tuple containing the training set and labels, (training_set, labels) 
    :param tuple testing: a tuple containing the test set and labels, (test_set, labels)
    :param dict param_grid: dict containing the search space for hyperparameters of the model, see sklearn.model_selection.
    :param sklearn.model_selection search_method: the search strategy to be used, for example GridSearchCV
    :param list seeds: the different seeds to be used
    :param bool verbose: include extra information in the return
    :param dict model_kwargs: kwarg arguments to be provided to model, model_instance = model(*args, **model_kwargs)
    :param dict search_kwargs: kwarg arguments to be provided to search, search = search_method(*args, **search_kwargs)
    :raise ValueError: if param_grid or seeds are not provided
    '''

    if param_grid is None:
        raise ValueError("param_grid kwarg is missing")

    if seeds is None:
        raise ValueError("seeds kwarg is missing")
    
    

    train_scores = []
    test_scores = []
    F1s = np.ndarray((len(seeds),2))
    recalls = np.ndarray((len(seeds),2))
    precisions = np.ndarray((len(seeds),2))
    AUCs = np.ndarray(len(seeds))
    train_accuracy = np.ndarray(len(seeds))

    
    for i,seed in enumerate(seeds):
        model_instance = model(random_state=seed, **model_kwargs)

        
        searcher = search_method(model_instance, param_grid, **search_kwargs)

        

        searcher.fit(*training)
        predict = searcher.predict(testing[0])
        train_predict = searcher.predict(training[0])

        precision, recall, F1, support = precision_recall_fscore_support(testing[1], predict)

        if hasattr(searcher.best_estimator_, 'predict_proba'):
            auc = roc_auc_score(testing[1], searcher.predict_proba(testing[0])[:,1])
        else:
            auc = np.nan

        AUCs[i] = auc
        F1s[i] = F1
        recalls[i] = recall
        precisions[i] = precision
        train_accuracy[i] = accuracy_score(training[1], train_predict)
        train_scores.append(searcher.best_score_)
        test_scores.append(accuracy_score(testing[1], predict))

    
    print(len(recalls))
    
    if len(seeds) > 1:
        cov = np.cov(np.stack((train_scores, test_scores, *F1s.T, *recalls.T, *precisions.T, AUCs.T, train_accuracy), axis=0))
    else:
        cov = None
    mean_train = np.mean(train_scores)
    mean_train_accuracy = np.mean(train_accuracy)
    mean_test = np.mean(test_scores)
    mean_F1 = np.mean(F1s, axis=0)
    mean_recall = np.mean(recalls, axis=0)
    mean_precision = np.mean(precisions, axis=0)
    mean_AUC = np.mean(AUCs)
    
    # order determined by TestResult class definitions
    test_res = TestResult(mean_F1, mean_test, mean_train, mean_train_accuracy, mean_precision, mean_recall, mean_AUC, support, cov)
    if not verbose:
        return test_res, cov, searcher.best_params_, searcher
    
    if verbose:
        dic = dict(test_res=test_res,
                   cov=cov,
                   params=searcher.best_params_,
                   train_scores=train_scores,
                   test_scores=test_scores,
                   searcher=searcher)
        return dic


def reorder_labels(old_labels, old_case_order, new_case_order):
    '''
    take in labels and corresponding case id order as well as a desired new case order,
    resort the labels so the correspond the new case order
    
    :param list old_labels: The old labels matching the old case order
    :param list old_case_order: a list of case_ids that are matched to the old labels list
    :param list new_case_order: a list of the same case_ids but in the desired order
    :return the labels reordered to match the new_case_order
    :raise ValueError: if case order list are not the same size
    :raise ValueError: if cases in the two lists do not match
    '''

    if len(old_case_order) > len(new_case_order):
        warnings.warn("There are more cases in the old case order than the new, this will result in cases being dropped")
    
    elif len(old_case_order) < len(new_case_order):
        raise ValueError("The size of the new case order is larger than the old, it cannot be remapped")

    # make sure that the casing (a=>A) is the same
    n_case_order = [case.upper() for case in new_case_order]
    o_case_order = [case.upper() for case in old_case_order]
    new_label_order = np.ndarray(len(new_case_order))

    for i,case in enumerate(n_case_order):
        # provide more descriptive error message
        try:
            j = o_case_order.index(case)
        except ValueError:
            raise ValueError('the cases in the two case orders do not match')
        
        new_label_order[i] = old_labels[j]
        

    return new_label_order

def reorder(old_case_order, new_case_order):
    '''
    take in labels and corresponding case id order as well as a desired new case order,
    return the mapping from the old to new case order
    
    
    :param list old_case_order: a list of case_ids that are matched to the old labels list
    :param list new_case_order: a list of the same case_ids but in the desired order
    :return the labels reordered to match the new_case_order
    :raise ValueError: if case order list are not the same size
    :raise ValueError: if cases in the two lists do not match
    '''

    if len(old_case_order) > len(new_case_order):
        warnings.warn("There are more cases in the old case order than the new, this will result in cases being dropped")
    
    elif len(old_case_order) < len(new_case_order):
        raise ValueError("The size of the new case order is larger than the old, it cannot be remapped")

    # make sure that the casing (a=>A) is the same
    n_case_order = [case.upper() for case in new_case_order]
    o_case_order = [case.upper() for case in old_case_order]
    new_label_order = np.ndarray(len(old_case_order), dtype=int)

    for i,case in enumerate(n_case_order):
        # provide more descriptive error message
        try:
            j = o_case_order.index(case)
        except ValueError:
            raise ValueError('the cases in the two case orders do not match')
        
        new_label_order[i] = j
        

    return new_label_order


def make_kaplan_plot(group1, group2, **kwargs):
    

    xgroup1, sgroup1, cgroup1 = group1
    xgroup2, sgroup2, cgroup2 = group2

    # since no events happen in the first two years i have to add a point at x = 0 to make the plot look more continuous
    xgroup2 = np.concatenate(([0], xgroup2))
    sgroup2 = np.concatenate(([1], sgroup2))
    cgroup2_0 = np.concatenate(([1], cgroup2[0]))
    cgroup2_1 = np.concatenate(([1], cgroup2[1]))

    # since no events happen in the first two years i have to add a point at x = 0 to make the plot look more continuous
    xgroup1 = np.concatenate(([0], xgroup1))
    sgroup1 = np.concatenate(([1], sgroup1))
    cgroup1_0 = np.concatenate(([1], cgroup1[0]))
    cgroup1_1 = np.concatenate(([1], cgroup1[1]))

    # plotting
    fig = go.Figure()

    
    fig.add_trace(go.Scatter(x=xgroup1/365, y=sgroup1, mode="lines", line_shape='hv', line_color='black', name='high risk', **kwargs))
    fig.add_trace(go.Scatter(x=np.concatenate([xgroup1, xgroup1[::-1]])/365, y=np.concatenate((cgroup1_0, cgroup1_1[::-1])),
                            fill='toself', showlegend=False, line_shape='vh', line_color='skyblue', **kwargs))

    fig.add_trace(go.Scatter(x=np.concatenate([xgroup2, xgroup2[::-1]])/365, y=np.concatenate((cgroup2_0, cgroup2_1[::-1])),
                            fill='toself', showlegend=False, line_shape='vh', line_color='#90ee90', **kwargs))
    fig.add_trace(go.Scatter(x=xgroup2/365, y=sgroup2, mode="lines", line_shape='hv', line_color='green', name='low risk', **kwargs))


    fig.update_layout(yaxis_title="Survival fraction", xaxis_title="years", legend_xanchor='right', 
                    yaxis_showgrid=True, xaxis_showgrid=True, font_size=26)

    
    
    return fig

# used for 2 year classification
def log_rank(predicted_survivor, predicted_death):
    '''calculate the log-rank score between two groups
    
    :param pd.DataFrame predicted_survivor: DataFrame of patients the model predicted as surviving, has to contain columns 'censored' and 'days_to_event'
    :param pd.DataFrame predicted_death: DataFrame of patients the model predicted as dying, has to contain columns 'censored' and 'days_to_event'
    :return: log rank test statistics and p-value
    '''

    X = stats.CensoredData(
    uncensored = predicted_survivor.loc[predicted_survivor['censored']==1, 'days_to_event'].values,
    right = predicted_survivor.loc[predicted_survivor['censored']==0, 'days_to_event'].values
    )
    Y = stats.CensoredData(
        uncensored = predicted_death.loc[predicted_death['censored']==1, 'days_to_event'].values,
        right = predicted_death.loc[predicted_death['censored']==0, 'days_to_event'].values
    )

    return stats.logrank(x=X, y=Y)


def score_AUC(model, X, y_true):
    pred = model.predict_proba(X)[:,1]
    return roc_auc_score(y_true, pred)

def rsf_train(train_set, survival_train):

    rsf = RandomSurvivalForest(random_state=42, n_jobs=-1)

    param_grid = {
    "n_estimators": np.arange(50,500,100),
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": np.arange(1,30,10)
    }   

    grid_search = GridSearchCV(
        estimator=rsf,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring=c_index_scorer,  # Custom C-index scorer
        n_jobs=-1  # Use all processors
    )

    # Fit the model with grid search
    grid_search.fit(train_set, survival_train)


    return grid_search

def regression_evaluate(grid_search, test_set, test_labels, test_times):
    # Predict risk scores on the test set
    risk_scores_test = grid_search.predict(test_set)

    # Calculate C-index on the test set
    c_index = concordance_index_censored(test_labels == 1, test_times, risk_scores_test)

    return c_index
    
def c_index_scorer(estimator, X, y):
    # Predict risk scores on the validation set
    risk_scores = estimator.predict(X)
    # Unpack survival data (censoring status and time)
    event, time = y['event'], y['time']
    # Calculate C-index
    return concordance_index_censored(event, time, risk_scores)[0]

def mean_predictions(models, X, shapes):
    scores = np.ndarray((len(models), X.shape[0]))
    start = 0
    for i, model in enumerate(models):
        scores[i] = model.predict(X[:,start:start+shapes[i]])
        start += shapes[i]
        print(start)

    
    return np.mean(scores, axis=0)


def grid_cross_validate(model, id_list, data_list, gene_names, matched_labels, matched_times, feature_selection, 
                        param_grid={}, feature_args=[], feature_kwargs={}, seed=432, refit=True):


    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    random_order = np.arange(0,len(data_list))
    rs.shuffle(random_order)
    slices = n_equal_slices(len(data_list), 5)

    # we have to split the param_grid into things targeted at feature_selection and those targeted at the model
    # for item in param_grid.items():
    #     print(item)

    feature_grid = {param_space:value for param_space,value in param_grid.items() if 'feature__' in param_space}
    model_grid = {param_space:value for param_space,value in param_grid.items() if 'feature__' not in param_space}

    key_order, feature_param_space = get_param_space(feature_grid)
    best_score = 0
    for param in feature_param_space:
        scores = []
        for bound in slices:

            val_set = random_order[bound[0]:bound[1]]
            val_bool = np.array([True if i in val_set else False for i in range(len(data_list))], dtype=bool)
            train_bool = ~val_bool

            # train_bool, id_list, data_list, gene_names, matched_labels, matched_times, alphas=[0.12]
            # print(key_order)
            feature_params = {}
            
            for i,key in enumerate(key_order):
                feature_params[key] = param[i]
            # print(feature_params)
            temp_params = drop_suffix_from_key(feature_params, 'feature__')
            feature_df = feature_selection(train_bool, id_list, data_list, gene_names, matched_labels, matched_times,
                                           *feature_args, **feature_kwargs, **temp_params)

            data = feature_df.loc[:, ~feature_df.columns.isin(['case_id', 'label', 'time'])].values
            labels = feature_df['label'].values
            times = feature_df['time'].values
            train_set = data
            train_labels = labels
            train_times = times
            

            survival_train = Surv.from_arrays(event=train_labels, time=train_times)
            # exclude -1 from being used as test set
            fold = [-1 if x else 0 for x in train_bool]
            ps = PredefinedSplit(test_fold=fold)
            grid_search = GridSearchCV(
                                        estimator=model,
                                        param_grid=model_grid,
                                        cv=ps,  # single fold
                                        scoring=c_index_scorer,  # Custom C-index scorer
                                        n_jobs=-1,  # Use all processors
                                        refit=False # we refit at the end with feature selection
                                    )

            # Fit the model with grid search
            grid_search.fit(train_set, survival_train)

            scores.append(grid_search.best_score_)
            
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_grid = grid_search
            best_feature_params = {param_space:value for param_space,value in feature_params.items()}
            best_grid.best_params_ = {**grid_search.best_params_, **best_feature_params}
            
    if refit:
        temp_params = drop_suffix_from_key(best_feature_params, 'feature__')
        feature_df = feature_selection(np.ones(len(id_list), dtype=bool), id_list, data_list, gene_names, matched_labels, matched_times,
                                           *feature_args, **feature_kwargs, **temp_params)
        survival_train = Surv.from_arrays(event=feature_df['label'].values, time=feature_df['time'].values)
        data = feature_df.loc[:, ~feature_df.columns.isin(['case_id', 'label', 'time'])].values
        model.fit(data, survival_train)
        
        best_grid.best_estimator_ = model
        best_grid._is_fitted = True
        best_grid.refit = True
        best_grid

    return best_grid
    

class TestModel():
    def fit(self, X, Y):
        return np.random.random()
            

def get_param_space(param_dict):
    lists = param_dict.values()
    return list(param_dict.keys()), list(itertools.product(*lists))

def drop_suffix_from_key(input_dict, suffix):
    return {key[len(suffix):]:value for key,value in input_dict.items() if suffix in key[:len(suffix)]}
    


if __name__ == '__main__':
    param_grid = {'k':[2,4,1],
                  }


    keys, param_space = get_param_space(param_grid)
    print(param_space)
