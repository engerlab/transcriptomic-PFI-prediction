import numpy as np
import pandas as pd
from scipy import stats as st
import xml.etree.ElementTree as ET
import glob
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.inspection import permutation_importance
from train import regression_evaluate, rsf_train, c_index_scorer, log_rank
from util import n_equal_slices, optimal_risk_split, kaplan_splitting
from sksurv.util import Surv


namespaces = {
    'hnsc': 'http://tcga.nci/bcr/xml/clinical/hnsc/2.7',
    'clin_shared': 'http://tcga.nci/bcr/xml/clinical/shared/2.7',
    'shared': 'http://tcga.nci/bcr/xml/shared/2.7',
    'shared_stage': 'http://tcga.nci/bcr/xml/clinical/shared/stage/2.7',
    'hnsc_nte': 'http://tcga.nci/bcr/xml/clinical/hnsc/shared/new_tumor_event/2.7/1.0',
    'nte': 'http://tcga.nci/bcr/xml/clinical/shared/new_tumor_event/2.7',
    'rx': 'http://tcga.nci/bcr/xml/clinical/pharmaceutical/2.7',
    'rad': 'http://tcga.nci/bcr/xml/clinical/radiation/2.7',
    'follow_up_v1.0': 'http://tcga.nci/bcr/xml/clinical/hnsc/followup/2.7/1.0',
    'admin': 'http://tcga.nci/bcr/xml/administration/2.7',
    'follow_up_v4.8':"http://tcga.nci/bcr/xml/clinical/hnsc/followup/2.7/4.8"
}

def recursive_pathing(root, path, ret) -> str:
    '''
    recursively add all children of a starting path to a set


    '''

    if path == '':
        path = root.tag.split('}')[-1]
    else:
        path += ':' + root.tag.split('}')[-1]

    # If the element has no children, add its path to the result
    if len(root) == 0:
        ret.add(path)
        return
    
    # Recurse for each child element
    for element in root:
        recursive_pathing(element, path, ret)

# Improved iterative gathering function to handle multiple entries
def gather_values_from_path(root, path, namespaces) -> str:
    # Split the path into components (tags)
    tags = path.split(':')
    
    # Initialize the current element as root
    current_elements = [root]  # Start with a list containing the root
    

    # Iterate through the path components to locate the desired elements
    for tag in tags:
        next_elements = []
        found_any = False
        
        # Check each namespace prefix for the given tag
        for current_element in current_elements:
            for prefix, url in namespaces.items():
                # Find all matches for this tag using the correct namespace
                elements = current_element.findall(f'{prefix}:{tag}', namespaces)
                
                if elements:
                    next_elements.extend(elements)  # Collect all found elements
                    found_any = True

        # If we found no elements, return 'N/A'
        if not found_any:
            return 'N/A'

        # Move to the next level with the found elements
        current_elements = next_elements

    # At this point, current_elements contains all matching elements at the end of the path
    if not current_elements:  # If no elements found
        return 'N/A'

    # Collect texts from all current elements and concatenate them with ' | '
    values = [el.text.strip() for el in current_elements if el.text is not None]
    return ' | '.join(values).replace(',', '') if values else 'N/A'


def make_header(filepath, target_file) -> None:
    '''make a header by taking all of the taking all of the data from an example file,
    the example file should include as many variables as possible'''

    with open("data/Clinical/Clinical_data/f881286d-4814-4f1a-9feb-25e5eb4f4f6b/nationwidechildrens.org_clinical.TCGA-HD-7229.xml", 'r') as f:
        data = f.read()

    # Parse the XML data
    root = ET.fromstring(data)

    ret = set()
    for patient in root.findall(f"hnsc:patient", namespaces):
        for element in patient:
            # ret is changed in place here
            recursive_pathing(element, '', ret)



    header = ','.join(ret)
    with open(target_file, 'w') as cf:
        cf.write(header)


def fill_file(filepath, clinical_files) -> None:
    '''fill the savefile with the data read from the clinical files
    
    :param str filepath: path to the save file
    :param list clinical_files: list of paths to the clinical.xml files from which the data is read'''


    header_list = list(np.genfromtxt(filepath, delimiter=',', max_rows=1, dtype=str))
    header = ','.join(header_list)

    # rewrite the header
    with open(filepath, 'w') as cf:
        cf.write(header + '\n')

        for i, file in enumerate(clinical_files):
            track_cols = 0

            with open(file) as f:
                data = f.read()

                # Parse the XML file
                root = ET.fromstring(data)

                for name in header_list:
                    
                    
                    patient = root.find(f"hnsc:patient", namespaces)
                    data = gather_values_from_path(patient, name, namespaces)


                    # Write to CSV, managing commas correctly
                    if track_cols < len(header_list) - 1:
                        cf.write(data + ',')
                        track_cols += 1
                    else:
                        cf.write(data + '\n')
                        track_cols += 1

                if track_cols != len(header_list):
                    print(f"filename = {file}, number_cols={track_cols}")
                    raise ValueError("the number of cols added does not equal the number of rows in the header")
    


def extract_clinical_info(save_file) -> None:
    '''Algorithm to extract all variables from the clinical.xml files and save it 
    to save_file
    
    :param str save_file: path to where extracted data is to be saved'''

    
    clinical_files = glob.glob("data/Clinical/Clinical_data/*/*.xml")
    make_header("data/Clinical/Clinical_data/f881286d-4814-4f1a-9feb-25e5eb4f4f6b/nationwidechildrens.org_clinical.TCGA-HD-7229.xml", save_file)
    fill_file(save_file, clinical_files)

def load_into_df(file) -> pd.DataFrame:
    '''take the path to the created csv file containing all parameters and load it into a dataframe,
    also simplify the naming scheme.
    
    :param str file: path to .csv with all the information'''


    data = pd.read_csv(file, sep=',', header=0, engine='python', skiprows=[1])

    i = 1
    while i < 4:
        for name in data.columns:
            try:
                last = ':'.join(name.split(':')[-i:])
                
                # reduce to the simplest name where possible
                # radiation:measure cannot be reduced further since pharam:measure exists
                # but ajcc:clinical_N can be reduced to just clinical_N
                if sum(last in col for col in data.columns) == 1:
                    data.rename(columns={name: last}, inplace=True)
            except IndexError:
                pass

        i += 1
    
    return data


def filter_df(data, threshold) -> pd.DataFrame:
    '''take in a data frame and filter it based on the missing values
    
    :param pd.DataFrame data: dataframe of variables'''


    for i, res in enumerate(data[['radiation_dosage', 'postoperative_rx_tx']].values):
        dose = res[0]
        post = res[1]
        # take the dose from the first followup, otherwise we could inadvertanly tell that a patient must
        # live a long time since no patient would receive 24000 Gy in a single treatment
        if dose is not np.nan:
            # sometimes value is 'N/A'
            try:
                first_dose = float(dose.split('|')[0].strip())
            except ValueError:
                first_dose = np.nan

        else:
            first_dose = np.nan 

        data.iloc[i, data.columns.get_loc('radiation_dosage')] = first_dose

        # Again take only postoperative from first followup
        if post is not np.nan:
            
            first_post = post.split('|')[0].strip()
            
        else:
            first_post = np.nan
        
        data.iloc[i, data.columns.get_loc('postoperative_rx_tx')] = first_post

    # change all the dosages to cGy
    data.loc[data['radiation:units'] == "Gy", "radiation_dosage"] = data.loc[data['radiation:units'] == "Gy", "radiation_dosage"] * 100
    data.loc[data['radiation:units'] == "Gy", "radiation:units"] = "cGy"

    # removing the data points with 100 cGy or less because I don't trust them (they have NaN or mistaken units)
    data.loc[data.radiation_dosage < 100, ["radiation:units", "radiation_dosage", "numfractions"]] = np.nan
    data.loc[data['radiation_therapy'] == 0, 'radiation_dosage'] = 0

    keep_names = ['gender','anatomic_neoplasm_subdivision', 'age_at_initial_pathologic_diagnosis',
                    'lymphnode_neck_dissection', 'neoplasm_histologic_grade', 'tobacco_smoking_history', 'race', 'icd_o_3_histology',
                    'clinical_stage', 'pathologic_T', 'pathologic_N', 'pathologic_M', 'ethnicity', 'margin_status', 'pathologic_stage', 'perineural_invasion_present', 
                    'presence_of_pathological_nodal_extracapsular_spread', 'lymphovascular_invasion_present', 'radiation_dosage', 'radiation_therapy', 
                    'postoperative_rx_tx','other_dx', 'bcr_patient_barcode'
                    ]
    
    filtered_data = data.filter(keep_names)

    
    return filtered_data


enum_d = {'clinical_M': {'M0': 0, 'M1': 2, 'MX': -1, 'N/A': -1},
            'clinical_N': {'N/A':-1,
                                'N0': 0,
                                'N1': 1,
                                'N2': 2,
                                'N2a': 2,
                                'N2b': 2,
                                'N2c': 2,
                                'N3': 3,
                                'NX': -1},
            'clinical_stage': {'N/A':-1,
                                    'Stage I': 1,
                                    'Stage II': 2,
                                    'Stage III': 3,
                                    'Stage IVA': 4,
                                    'Stage IVB': 4,
                                    'Stage IVC': 4},
            'clinical_T': {'N/A':-1,
                                'T1': 1,
                                'T2': 2,
                                'T3': 3,
                                'T4': 4,
                                'T4a': 4,
                                'T4b': 5,
                                'TX': 0},
            'pathologic_M': {'N/A':-1, 'M0': 0, 'M1': 1, 'MX': 0},
            'pathologic_N': {'N0': 0,
                                'N1': 1,
                                'N2': 2,
                                'N2a': 2,
                                'N2b': 2,
                                'N2c': 2,
                                'N3': 3,
                                'NX': -1,
                                'N/A': -1},
            'pathologic_stage': {'N/A':-1,
                                      'Stage I': 1,
                                    'Stage II': 2,
                                    'Stage III': 3,
                                    'Stage IVA': 4,
                                    'Stage IVB': 4,
                                    'Stage IVC': 4},
            'pathologic_T': {'T0': 0,
                                'T1': 1,
                                'T2': 2,
                                'T3': 3,
                                'T4': 4,
                                'T4a': 4,
                                'T4b': 4,
                                'TX': -1,
                                'N/A': -1},
            'gender': {'female': 1, 'male': 0},
            'neoplasm_histologic_grade': {'N/A': -1,
                                          'G1': 1,
                                          'G2': 2,
                                          'G3': 3,
                                          'G4': 4,
                                          'GX': -1},
            'margin_status' : {'Negative':0, 'Close':1, 'Positive':2},
            'presence_of_pathological_nodal_extracapsular_spread': {'No Extranodal Extension': 0, 'Microscopic Extension': 1, 'Gross Extension': 2, 'N/A': -1},
}

def convert_to_numerical(df) -> pd.DataFrame:
    '''take in the filtered df with tabular data, convert the strings into a 
    numerical mapping'''


    to_be_dummied = ['ethnicity',
                  'icd_o_3_histology', 'race', 'anatomic_neoplasm_subdivision', 'tobacco_smoking_history', 'lymphnode_neck_dissection',
                  'perineural_invasion_present', 'postoperative_rx_tx', 'lymphovascular_invasion_present', 'other_dx',
                  'radiation_therapy' 
                  ]
    

    dummied_df = pd.get_dummies(df, columns=to_be_dummied, dtype=int).copy()

    for key in enum_d.keys():
        try:
            dummied_df[key] = dummied_df[key].apply(lambda x: enum_d[key][x] if x in list(enum_d[key].keys()) else -1)
        except KeyError:
            print(f"{key} not in keys")

    return dummied_df


def impute_scale(df, train_bool) -> pd.DataFrame:

    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    imr.fit(df[train_bool].values)

    imputed_df = pd.DataFrame(data = imr.transform(df.values), columns=df.columns)

    mms = MinMaxScaler()

    mms.fit(imputed_df[train_bool].values)

    # imputed_scaled_df = pd.DataFrame(data=mms.transform(imputed_df.values), columns=imputed_df.columns)

    return mms.transform(imputed_df.values)

def clinical_load_data() -> pd.DataFrame:
    save_file = 'data/Clinical/accumulated_data_2.csv'
    
    extract_clinical_info(save_file)
    data = load_into_df(save_file)
    print(data.shape)
    filtered_data = filter_df(data, 70)
    print(filtered_data.shape)
    numerical = convert_to_numerical(filtered_data.loc[:, ~filtered_data.columns.isin(['bcr_patient_barcode'])])
    numerical['bcr_patient_barcode'] = filtered_data['bcr_patient_barcode']

    # numerical['bcr_patient_barcode'] = numerical['bcr_patient_barcode'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    labels = pd.read_csv('labels/PFI.csv', header=3, names=['bcr_patient_barcode', 'label', 'time'], usecols=(0,2,3))
    labels = labels.dropna()

    # labels['bcr_patient_barcode'] = labels['bcr_patient_barcode'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    
    labeled_data = pd.merge(numerical, labels, left_on='bcr_patient_barcode', right_on='bcr_patient_barcode', how='left')
    labeled_data.rename(columns={'bcr_patient_barcode':'case_id'}, inplace=True)
    
    labeled_data = labeled_data[~labeled_data['label'].isna()]
    print(labeled_data.shape)
    # labeled_data.info()

    return labeled_data


def main(seed):
    c_vals = [] # store c-index for each test set
    log_ranks = []
    features = [] # store the feature importance for each test set
    feature_names = []
    data = clinical_load_data()
    print(data.columns)
    # print(data.info())
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    random_order = np.arange(0,len(data))
    rs.shuffle(random_order)
    slices = n_equal_slices(len(data), 5)
    # iterations = 30
    # for i in range(iterations):
    for i,bound in enumerate(slices):

        # rs.shuffle(random_order)
        test_set = random_order[bound[0]:bound[1]]
        # test_set = random_order[:len(random_order)//5]
        print(test_set.shape)

        test_bool = np.array([True if i in test_set else False for i in range(len(data))], dtype=bool)
        train_bool = ~test_bool
        
        scaled = impute_scale(data.loc[:, ~data.columns.isin(['case_id', 'label', 'time'])], train_bool)
        
        labels = data['label'].values
        times = data['time'].values

        train_set = scaled[train_bool]
        train_labels = labels[train_bool]
        train_times = times[train_bool]
        test_set = scaled[~train_bool]
        test_times = times[~train_bool]
        test_labels = labels[~train_bool]

        survival_train = Surv.from_arrays(event=train_labels, time=train_times)
        survival_test = Surv.from_arrays(event=test_labels, time=test_times)
        grid_search = rsf_train(train_set, survival_train)
        train_risk = grid_search.predict(train_set)
        test_risk = grid_search.predict(test_set)

        res = regression_evaluate(grid_search, test_set, test_labels, test_times)
        result = permutation_importance(grid_search.best_estimator_, test_set, survival_test, n_repeats=10, random_state=42, scoring=c_index_scorer)
        
        group1, group2 = kaplan_splitting(train_labels, train_times, train_risk, test_labels, test_times, test_risk)
        group1.to_csv(f'data/kaplan/RSF_clinical_test_set_{i}_group1.csv', index_label=False)
        group2.to_csv(f'data/kaplan/RSF_clinical_test_set_{i}_group2.csv', index_label=False)
        features += list(result.importances_mean)
        feature_names += list(data.columns[~data.columns.isin(['case_id', 'label', 'time'])])
        lr = log_rank(group1, group2).pvalue

        print(res[0])
        log_ranks.append(lr)
        c_vals.append(res[0])

    
    print(f"{np.mean(c_vals)} +/- {2*np.std(c_vals)/np.sqrt(5)}")
    print(log_ranks)
    print(f"{np.mean(log_ranks)} +/- {2*np.std(log_ranks)/np.sqrt(5)}")
    
    feature_res =[[i,j] for i,j in zip(feature_names, features)]
    feature_df = pd.DataFrame(data=feature_res, columns=['feature', 'mean_importance'])
    feature_df.to_csv('data/clinical_feature_importance.csv')

if __name__ == '__main__':
    main(5)
    