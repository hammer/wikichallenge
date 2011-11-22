#!/usr/bin/env python
"""
Implementation of Dell Zhang's solution to Wikipedia's Participation Challenge on Kaggle

Kaggle competition: http://www.kaggle.com/c/wikichallenge
Dell Zhang's solution: http://blog.kaggle.com/2011/10/26/long-live-wikipedia-dell-zhang
"""
from calendar import monthrange
from datetime import datetime
from itertools import takewhile
import cPickle as pickle
import locale
import math
import sys

import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import gaussian_process
import cv2

#
# Constants
#
locale.setlocale(locale.LC_ALL, 'en_US')
PERIODS = [1/16.0, 1/8.0, 1/4.0, 1/2.0, 1, 2, 4, 12, 36, 108]
TEST_TIMES = {'training': 116,
              'moredata': 111,
              'validation': 84,
             }
# NB: NeighborsRegressor was deprected in sklearn 0.9
# NB: mode "mean" for NeighborsRegressor does not exist
MODEL_TYPES = {
    'ols': linear_model.LinearRegression(),
    'ridge': linear_model.Ridge(),
    'lasso': linear_model.Lasso(),
    'elasticnet': linear_model.ElasticNet(),
    'sgd': linear_model.SGDRegressor(),
    'svr': svm.SVR(),
    'svr_linear': svm.SVR(kernel='linear', C=1),
    'svr_rbf': svm.SVR(kernel='rbf', C=1e3),
    '5nn': neighbors.KNeighborsRegressor(),
    '120nn': neighbors.KNeighborsRegressor(n_neighbors=120),
    'gaussian_process': gaussian_process.GaussianProcess(),
    'gbt': cv2.GBTrees()
}
GBT_WEAK_COUNT = 1000

#
# Data processing
#

def parse_timestamp(dt_str):
    """Maps a timestamp into months since 1/1/2001
    """
    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

    # months
    months = (dt.year - 2001) * 12 + (dt.month - 1)

    # days
    # NB: monthrange() returns the number of days in a particular month
    delta = dt - datetime(dt.year, dt.month, 1)
    days = delta.days + (delta.seconds + delta.microseconds/1000000.0)/(3600.0*24.0)
    days /= float(monthrange(dt.year, dt.month)[1])
    return months + days

def process_edits(dataset, time_test):
    """
    Read through the list of edits and:
    1) Extract a list of all editors who have made an edit in the 1 year prior
       to the start of the testing period, and
    2) Extract user_id, article_id, and transformed timestamp for each edit
    """
    active_editors_test = []
    processed_edits = []
    data_file = open('data/%s.tsv' % dataset)
    data_file.readline()  # header

    for line_number, line in enumerate(data_file):
        # Progress report
        if line_number % 10000 == 0: print "Processing line %s" % locale.format('%d', line_number, grouping=True)

        # Save user_id, article_id, and transformed timestamp
        attr = line.strip().split('\t')
        user = int(attr[0])
        article = int(attr[1])
        parsed_timestamp = parse_timestamp(attr[4])
        processed_edits.append([user, article, parsed_timestamp])

        # If the edit happened within 1 year of the test time, note the editor
        if parsed_timestamp >= time_test - 12 and parsed_timestamp < time_test:
            active_editors_test.append(user)

    data_file.close()
    # deduplicate active_editors
    active_editors_test = list(set(active_editors_test))
    return active_editors_test, processed_edits

def filter_and_group_edits(active_editors_test, processed_edits):
    """
    1) Filter out 1y inactive users by doing a sort-merge join
    2) Group the edits for each user
    3) Sort the edits for each user by timestamp

    Returns grouped_edits[user_id] = [(timestamp1, article1), (timestamp2, article2), ...]
    """
    grouped_edits = {}

    edits_enumerator = enumerate(processed_edits)
    edit_number, edit = edits_enumerator.next()
    for editor in active_editors_test:
        # Skip to edits by editor
        done = False
        while edit[0] != editor and not done:
            try:
                edit_number, edit = edits_enumerator.next()
                # Progress report
                if edit_number % 10000 == 0: print "Processing edit %s" % locale.format('%d', edit_number, grouping=True)
            except StopIteration:
                done = True

        # Collect all edits by editor
        grouped_edits[editor] = []
        while edit[0] == editor and not done:
            _, article_id, timestamp = edit
            grouped_edits[editor].append((timestamp, article_id))
            try:
                edit_number, edit = edits_enumerator.next()
                # Progress report
                if edit_number % 10000 == 0: print "Processing edit %s" % locale.format('%d', edit_number, grouping=True)
            except StopIteration:
                done = True

        # Sort edits by editor by timestamp in descending order
        # NB: need copysign since cmp needs an integer and int() and math.trunc() round towards 0
        # TODO: figure out if this even helps
        grouped_edits[editor].sort(cmp=lambda x, y: int(math.copysign(1, x[0] - y[0])), reverse=True)

    return grouped_edits

#
# Featurize/targetize
#

def count_edits(edits, deadline_month, months_prior):
    """Calculate the number of edits between deadline_month - months_prior and deadline_month
    """
    return len([edit for edit in edits if deadline_month - months_prior < edit[0] < deadline_month])

def count_articles(edits, deadline_month, months_prior):
    """Calculate the number of articles edited between deadline_month - months_prior and deadline_month
    """
    return len(set([edit[1] for edit in edits if deadline_month - months_prior < edit[0] < deadline_month]))

def time_between_first_and_last_edit(edits, deadline_month):
    """Calculate the time between the first edit and the last edit before the deadline_month
    """
    last_edit_index = len([edit for edit in takewhile(lambda x: x[0] > deadline_month, edits)])
    try:
        return edits[last_edit_index][0] - edits[-1][0]
    except IndexError:
        return 0.0        # No edits before deadline_month

def featurize_single_editor(edits, deadline_month):
    """Generate the 21 features for a single editor
    """
    vector = []
    # number of edits
    vector += [float(count_edits(edits, deadline_month, months_prior)) for months_prior in PERIODS]
    # number of articles
    vector += [float(count_articles(edits, deadline_month, months_prior)) for months_prior in PERIODS]
    # natural log of time between first and last edit
    vector += [math.log1p(time_between_first_and_last_edit(edits, deadline_month))]
    return vector

def featurize_all_editors(editors, grouped_edits, deadline_month):
    """Generate the 21 features for all editors
    """
    return [featurize_single_editor(grouped_edits[editor], deadline_month) for editor in editors]

def load_validation_targets_test(active_editors_test):
    """Load the known targets_test for the validation set
    """
    validation_targets_test_file = open('data/validation_solutions.csv', 'r')
    validation_targets_test_file.readline()  # header
    edits_count = {}
    for line_number, line in enumerate(validation_targets_test_file):
        # Progress report
        if line_number % 10000 == 0: print "Processing line %s" % locale.format('%d', line_number, grouping=True)
        
        (editor_str, target_str) = line.strip().split(',')
        editor = int(editor_str)
        target = int(target_str)
        if editor not in active_editors_test:
            continue
        edits_count[editor] = target
    validation_targets_test_file.close()
    return edits_count

def targetize_all_editors(editors, grouped_edits, deadline_month):
    """
    Generate the targets for all editors, where the target for a single editor is the
    natural log of the number of edits in a chosen five month period
    """
    return [math.log1p(count_edits(grouped_edits[editor], deadline_month, 5)) for editor in editors]

#
# Model fitting and predicting
#

# TODO: parameterize model fitting
def learn(featurized_data_train, targets_train, model_type='gbt'):
    """Fit the model
    """
    model = MODEL_TYPES[model_type]
    # 'subsample_portion':0.8, 'shrinkage':0.01
    model.train(featurized_data_train,
                cv2.CV_ROW_SAMPLE,
                targets_train,
                params={'weak_count':GBT_WEAK_COUNT})
    return model

def drift(active_editors_test, grouped_edits, time_train, time_test):
    """Calculate drift
    """
    average_train = sum([math.log1p(count_edits(grouped_edits[editor], time_train, 5))
                         for editor in active_editors_test])/len(active_editors_test)
    average_test = sum([math.log1p(count_edits(grouped_edits[editor], time_test, 5))
                        for editor in active_editors_test])/len(active_editors_test)
    return average_test - average_train

# TODO: parameterize estimation
def estimate(model, featurized_data_test, drift):
    forecasts = [model.predict(sample) for sample in featurized_data_test]
    return [max(y + drift, 0) for y in forecasts]

def rmsle(targets_test_predicted, targets_test):
    n = len(targets_test_predicted)
    sle = sum([math.pow(targets_test_predicted[i] - targets_test[i], 2) for i in range(n)])
    return math.sqrt(sle/n)


#
# Main
#

if __name__ == "__main__":
    dataset = sys.argv[1]
    try:
        time_test = TEST_TIMES[dataset]
        time_train = time_test - 5
    except KeyError:
        sys.exit("The dataset %s is unknown." % dataset)

    #
    # 1. Process data
    #

    print "Generating active editors and processed edits lists"
    active_editors_test, processed_edits = process_edits(dataset, time_test)
    print "Done generating active editors and processed edits lists"

    print "Sorting and pickling the active editors"
    active_editors_test.sort()
    pkl_file = open('data/%s_active_editors_test.pkl' % dataset, 'wb')
    pickle.dump(active_editors_test, pkl_file, -1)
    pkl_file.close()
    print "Done sorting and pickling the active editors"

    print "Sorting and pickling the processed edits"
    processed_edits.sort(cmp=lambda x, y: x[0] - y[0])
    pkl_file = open('data/%s_processed_edits.pkl' % dataset, 'wb')
    pickle.dump(processed_edits, pkl_file, -1)
    pkl_file.close()
    print "Done sorting and pickling the processed edits"

    print "Filtering and grouping edits"
    grouped_edits = filter_and_group_edits(active_editors_test, processed_edits)
    pkl_file = open('data/%s_grouped_edits.pkl' % dataset, 'wb')
    pickle.dump(grouped_edits, pkl_file, -1)
    pkl_file.close()
    print "Done filtering and grouping edits"

    #
    # 2. Generate featurized data and targets for training
    #

    print "Calculating elgible editors for training"
    active_editors_train = [editor for editor in active_editors_test
                            if count_edits(grouped_edits[editor], time_train, 12)]
    pkl_file = open('data/%s_active_editors_train.pkl' % dataset, 'wb')
    pickle.dump(active_editors_train, pkl_file, -1)
    pkl_file.close()
    print "Done calculating elgible editors for training"

    print "Generating featurized data for training"
    featurized_data_train = np.array(featurize_all_editors(active_editors_train, grouped_edits, time_train), dtype=np.float32)
    pkl_file = open('data/%s_featurized_data_train.pkl' % dataset, 'wb')
    pickle.dump(featurized_data_train, pkl_file, -1)
    pkl_file.close()
    print "Done generating featurized data for training"

    print "Generating targets for training"
    targets_train = np.array(targetize_all_editors(active_editors_train, grouped_edits, time_train + 5), dtype=np.float32)
    pkl_file = open('data/%s_targets_train.pkl' % dataset, 'wb')
    pickle.dump(targets_train, pkl_file, -1)
    pkl_file.close()
    print "Done generating targets for training"

    #
    # 3. Train model
    #

    print "Fitting model"
    model = learn(featurized_data_train, targets_train)
    print "Done fitting model"

    #
    # 4. Generate featurized data and targets for testing
    #

    print "Generating featurized data for testing"
    featurized_data_test = np.array(featurize_all_editors(active_editors_test, grouped_edits, time_test), dtype=np.float32)
    pkl_file = open('data/%s_featurized_data_test.pkl' % dataset, 'wb')
    pickle.dump(featurized_data_test, pkl_file, -1)
    pkl_file.close()
    print "Done fenerating featurized data for testing"

    print "Generating targets for testing"
    if dataset == 'validation':
        print "Loading known validation test targets"
        edits_count = load_validation_targets_test(active_editors_test)
        targets_test = np.array([math.log1p(edits_count[editor]) for editor in active_editors_test], dtype=np.float32)
        print "Done loading known validation test targets"
    else:
        targets_test = np.array(targetize_all_editors(active_editors_test, grouped_edits, time_test + 5), dtype=np.float32)
    pkl_file = open('data/%s_targets_test.pkl' % dataset, 'wb')
    pickle.dump(targets_test, pkl_file, -1)
    pkl_file.close()
    print "Done generating targets for testing"

    #
    # 5. Predict the targets
    #

    print "Calculating drift"
    # validation drift = -0.271349248816
    # moredata drift = 0.0112541788177
    drift = drift(active_editors_test, grouped_edits, time_train, time_test)
    print "Done calculating drift: %s" % drift

    print "Predicting the test targets"
    targets_test_predicted = estimate(model, featurized_data_test, drift)
    print "Done predicting the test targets"

    #
    # 6. Calculate loss
    #
    print "RMSLE = %.6f" % rmsle(targets_test_predicted, targets_test)


