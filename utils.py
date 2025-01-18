import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
from scipy import stats
from itertools import combinations
from collections import Counter
import statsmodels.api as sm
import ast
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

# split users randomly and with stratification; userdata must be in dataframe, splits is the number of splits, stratify contains the columns for stratification
def split_users(userdf, splits, stratify = []):
    originals = []
    reproductions = []
    for i in range(splits):
        if stratify: 
            # Create a composite stratification column (tuple of values from stratify_cols)
            userdf['composite_stratify'] = userdf[stratify].apply(lambda x: tuple(x), axis=1)
            # Handle missing values
            if userdf['composite_stratify'].apply(lambda x: any(pd.isna(i) for i in x)).any():
                # Split rows into those with and without NaN in the stratification columns
                df_non_nan = userdf[userdf['composite_stratify'].apply(lambda x: all(pd.notna(val) for val in x))]
                df_nan = userdf[userdf['composite_stratify'].apply(lambda x: any(pd.isna(val) for val in x))]
                # Perform stratified train-test split on non-NaN rows
                o_non_nan, r_non_nan = train_test_split(df_non_nan, test_size=0.5, stratify=df_non_nan['composite_stratify'])
                # Split NaN rows randomly (without stratification)
                if len(df_nan) > 1:
                    o_nan, r_nan = train_test_split(df_nan, test_size=0.5)
                else:
                    o_nan = df_nan
                    r_nan = pd.DataFrame()
                # Combine the stratified and random splits
                o = pd.concat([o_non_nan, o_nan], axis=0)
                r = pd.concat([r_non_nan, r_nan], axis=0) 
            else:
                o, r = train_test_split(userdf, test_size=0.5, stratify=userdf['composite_stratify'])
        else: 
            o, r = train_test_split(userdf, test_size=0.5)
        originals.append(list(o['user']))
        reproductions.append(list(r['user']))
    return [originals, reproductions]

# create dataframe with mean and std for all search behaviours and times listed in "tab", based on "df", for all conditions in "conditions"
def show_original_results(tab,df,conditions):
    def create_condition_dataframe(tab,df,c):
        for i in range(len(tab)):
            t = tab[i]
            if i == 0:
                result = df[[c,t]].groupby([c]).agg({t: ['mean', 'std']}).reset_index()
                result = result.sort_index(axis=1)
            else:
                add = df[[c,t]].groupby([c]).agg({t: ['mean', 'std']}).reset_index()
                add = add.sort_index(axis=1)
                result = result.merge(add,on=c, how='outer')
        return result
    result = pd.DataFrame()
    for c in conditions:
        if result.empty:
            result = create_condition_dataframe(tab,df,c)
            result.rename(columns={c:'cond'}, inplace=True)
        else:
            add = create_condition_dataframe(tab,df,c)
            add.rename(columns={c:'cond'}, inplace=True)
            result = pd.concat([result, add])
    return result.reset_index(drop=True)

# for all measures listed in "tab", get p-values for conditions in "conditions"
def pvals_table(tab,df,conditions):
    def pvals_condition_column(df,cond,col):
        groups = df[[cond,col]].groupby([cond])[col].apply(list)
        group_pairs = list(combinations(groups.index, 2))
        p_values = []
        r = {}
        for g1, g2 in group_pairs:
            _, p_val = stats.mannwhitneyu([x for x in groups[g1] if not np.isnan(x)], [x for x in groups[g2] if not np.isnan(x)], alternative='two-sided')
            p_values.append(p_val)
        _, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        for i, (g1, g2) in enumerate(group_pairs):
            r[(g1,g2)] = pvals_corrected[i]
        return r
    r = {}
    for col in tab:
        for cond in conditions:
            r[col,cond] = pvals_condition_column(df,cond,col)
    df = pd.DataFrame(r)
    # combine columns with different conditions (interface, ps, cf)
    main_columns = df.columns.get_level_values(0).unique()  # Get unique main column names
    combined_cols = []
    for main_col in main_columns:
        # for each main column, stack the subcolumns and keep multi-index
        combined = df[main_col].stack().rename(main_col)
        combined_cols.append(combined)
    # concatenate the results
    final_df = pd.concat(combined_cols, axis=1, sort=False)
    final_df.index = final_df.index.reorder_levels([2, 0, 1])
    #final_df[final_df > 0.05] = np.nan # remove non-significant values 
    return final_df

# used for stratification
def level1_findings(df, tab, conditions, splits):
    effects_per_ex = []
    succ_per_repro = []
    fails_per_repro = []
    for i in range(len(splits[0])):
            # create dataframe for each "study"
            A = df[df['user'].isin(splits[0][i])] # "original"
            B = df[df['user'].isin(splits[1][i])] # "reproduction"
            # find significant effects in each study
            A_p = pvals_table(tab,A,conditions)
            B_p = pvals_table(tab,B,conditions)
            # add number of significant effects per experiment, successfully reproduced effects, and effects that were failed to reproduce to the respective lists
            effects_per_ex.append(pd.to_numeric(A_p.stack(), errors='coerce').lt(0.05).sum())
            effects_per_ex.append(pd.to_numeric(B_p.stack(), errors='coerce').lt(0.05).sum())
            succ_per_repro.append(((A_p <= 0.05) & (B_p <= 0.05)).sum().sum())
            fails_per_repro.append(((A_p > 0.05) & (B_p <= 0.05)).sum().sum() + ((A_p <= 0.05) & (B_p > 0.05)).sum().sum())
            
    print('average number of significant effects per study:',np.mean(effects_per_ex))
    print('average number of successfully reproduced effects:',np.mean(succ_per_repro))
    print('average number of effects only present in one study:',np.mean(fails_per_repro)) # note: regardless of the study being the "original" or the "reproduction"

def level2_measurements(session_df, tab, splits):
    def compare_measurements_l2(df1, df2, tab):
        r = {}
        for col in tab:
            l1 = list(df1[col])
            l2 = list(df2[col])
            # Mann Whitney U test
            _, p_val = stats.mannwhitneyu([x for x in l1 if not np.isnan(x)], [x for x in l2 if not np.isnan(x)], alternative='two-sided')
            r[col] = p_val
        # correcting p values using benjamini hochberg
        p_values = list(r.values())
        keys = list(r.keys())
        _, pvals_corrected, _, _ = sm.stats.multipletests(p_values, method='fdr_bh')
        r = dict(zip(keys, pvals_corrected))
        # return a list with corrected p-values for each metric
        return r
    count_l2_diffs = Counter()

    pvals_l2 = []
    for i in range(len(splits[0])):
        # creating dataframes for each group (original/reproduction)
        A = session_df[session_df['user'].isin(splits[0][i])]
        B = session_df[session_df['user'].isin(splits[1][i])]
        # adding list of p-values for all metrics to list for all reproductions
        pvals_l2.append(compare_measurements_l2(A,B,tab))

    # for each measure, count in how many reproductions there are significant differences between the groups
    for d in pvals_l2:
        for key, value in d.items():
            if value < 0.05: 
                count_l2_diffs[key] += 1

    # print results
    for key in tab:
        if key in count_l2_diffs.keys():
            print(f"Significant differences for '{key}': {count_l2_diffs[key]}")
        else:
            print(f"Significant differences for '{key}': 0")

def level3_times(session_df, states, splits):

    def compare_times_l3(df1, df2, states):
        r = {}
        for s in states:
            col = s.lower()+' time mean' # corresponding column name in dataframe
            A_times = list(df1[col])
            B_times = list(df2[col])
            _, p_val = stats.mannwhitneyu([x for x in A_times if not np.isnan(x)], [x for x in B_times if not np.isnan(x)], alternative='two-sided')
            r[s] = p_val
        # apply Benjamini-Hochberg correction 
        keys = list(r.keys())
        p_values = list(r.values())
        _, pvals_corrected, _, _ = sm.stats.multipletests(p_values, method='fdr_bh')
        r = dict(zip(keys, pvals_corrected))
        return r

    count_l3_times = Counter()
    pvals_l3_times = []

    for i in range(len(splits[0])):
        A = session_df[session_df['user'].isin(splits[0][i])].copy() # "original"
        B = session_df[session_df['user'].isin(splits[1][i])].copy() # "reproduction"

        pvals_l3_times.append(compare_times_l3(A,B,states))

    # for each measure, count in how many reproductions there are significant differences between the groups
    for d in pvals_l3_times:
        for key, value in d.items():
            if value < 0.05: 
                count_l3_times[key] += 1

    # print results
    for s in states:
        if s in count_l3_times.keys():
            print(f"Significant differences for '{s}': {count_l3_times[s]}")
        else:
            print(f"Significant differences for '{s}': 0")

def level3_transitions(session_df, states, splits):
    def create_transition_df(df):
        dicts = list(df['session_transition_dict']) # get transition dictionaries from sessions
        dfs = [pd.DataFrame(ast.literal_eval(d)) for d in dicts] # dictionaries to dataframes
        normalized_dfs = [df.div(df.sum(axis=1), axis=0) for df in dfs] # normalize
        concatenated_df = pd.concat(normalized_dfs)
        result_df = concatenated_df.groupby(concatenated_df.index).sum()
        result_df = result_df[states] # sort columns
        result_df = result_df.loc[states] # sort rowse
        result_df_normal = result_df.div(result_df.sum(axis=1), axis=0) # normalize
        result_df_normal = result_df_normal.fillna(0)
        result_df_normal.drop(['END'],inplace=True) # drop row "END" because there are no outgoing transitions
        return result_df_normal
    frobenius = []
    jsd = []
    ks = []

    for i in range(len(splits[0])):
        A = session_df[session_df['user'].isin(splits[0][i])] # "original"
        B = session_df[session_df['user'].isin(splits[1][i])] # "reproduction"
        lmdf1 = create_transition_df(A)
        lmdf2 = create_transition_df(B)
        M1 = lmdf1.to_numpy()
        M2 = lmdf2.to_numpy()

        # Frobenius Norm
        frobenius_norm = np.linalg.norm(lmdf1.values - lmdf2.values, 'fro')
        frobenius.append(frobenius_norm)

        # Jensen-Shannon Divergence for each row
        js_divergences = [jensenshannon(M1[j], M2[j]) for j in range(len(M1))]
        jsd.append(js_divergences)

        # Kolmogorov-Smirnov Test for each row
        ks_tests = [ks_2samp(M1[j], M2[j])[1] for j in range(len(M1))]
        ks.append(ks_tests)

    print('mean, min, and max values')
    print('Frobenius:',np.mean(frobenius),np.min(frobenius),np.max(frobenius))
    states_tab4 = ['START', 'TASK', 'SERP', 'SNIPPET', 'DOC', 'MARK', 'REVIEW'] # no query, no end 
    for i in range(len(states_tab4)):
        print(states_tab4[i])
        jsdlist = []
        for j in range(len(jsd)):
            jsdlist.append(jsd[j][i])
        print('JSD', np.mean(jsdlist),np.min(jsdlist),np.max(jsdlist))
        kslist = []
        for j in range(len(ks)):
            kslist.append(ks[j][i])
        print('KS', np.mean(kslist),np.min(kslist),np.max(kslist))