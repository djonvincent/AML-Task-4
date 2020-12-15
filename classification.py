import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.ndimage.filters import uniform_filter1d
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from pystruct.learners import FrankWolfeSSVM, OneSlackSSVM
from pystruct.models import ChainCRF
from pystruct.models import LatentGraphCRF
from load_data import load_data
from features import power_features, PLF, onset_features, EMG_power

N_SPLITS=3 # 3 subjects in train data

parser = ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--finetune', dest='finetune', action='store_true', default=False)
args = parser.parse_args()

#DATAFRAME
x_train, y_train, x_test = load_data()

def extract_features(x_raw, x_test_raw):
    print('extracting features...')
    PF_path_train = Path('data', 'train_pf.csv')
    PF_path_test = Path('data', 'test_pf.csv')
    if PF_path_train.exists():
        pf = np.loadtxt(PF_path_train, delimiter=',')
    else:
        pf = power_features(x_raw[:, :2])
        np.savetxt(X=pf, fname=PF_path_train, delimiter=',')
    if PF_path_test.exists():
        pf_test = np.loadtxt(PF_path_test, delimiter=',')
    else:
        pf_test = power_features(x_test_raw[:, :2])
        np.savetxt(X=pf_test, fname=PF_path_test, delimiter=',')

    PLF_path_train = Path('data', 'train_plf.csv')
    PLF_path_test = Path('data', 'test_plf.csv')
    if PLF_path_train.exists():
        plf = np.loadtxt(PLF_path_train, delimiter=',')
    else:
        plf = PLF(x_raw[:, :2])
        np.savetxt(X=plf, fname=PLF_path_train, delimiter=',')
    if PLF_path_test.exists():
        plf_test = np.loadtxt(PLF_path_test, delimiter=',')
    else:
        plf_test = PLF(x_test_raw[:, :2])
        np.savetxt(X=plf_test, fname=PLF_path_test, delimiter=',')

    onsets = onset_features(x_raw[:,2]).reshape(-1,1)
    onsets_test = onset_features(x_test_raw[:,2]).reshape(-1,1)

    emg_pow = EMG_power(x_raw[:, 2]).reshape(-1,1)
    emg_pow_test = EMG_power(x_test_raw[:, 2]).reshape(-1,1)

    f_train = np.hstack((pf, emg_pow, onsets))
    f_test = np.hstack((pf_test, emg_pow_test, onsets_test))

    return f_train, f_test

def preprocess(f_train, y, f_test, k_features=None):

    #SCALING - normalise signal powers per subject
    for i in range(0, f_train.shape[0], 21600):
        f_train[i:i+21600, :11] = StandardScaler().fit_transform(f_train[i:i+21600, :11])
    for i in range(0, f_test.shape[0], 21600):
        f_test[i:i+21600, :11] = StandardScaler().fit_transform(f_test[i:i+21600, :11])
    scaler = StandardScaler().fit(f_train[:, 11:])
    f_train[:, 11:] = scaler.transform(f_train[:, 11:])
    f_test[:, 11:] = scaler.transform(f_test[:, 11:])

    #SMOOTHING
    n = 9
    f_train = uniform_filter1d(f_train, axis=0, size=n, mode='nearest')
    f_test = uniform_filter1d(f_test, axis=0, size=n, mode='nearest')

    '''
    pca = PCA(n_components=5)
    pca.fit(f_train[:, :10])
    f_train = np.hstack((pca.transform(f_train[:, :10]), f_train[:,10:]))
    f_test = np.hstack((pca.transform(f_test[:, :10]), f_test[:,10:]))
    '''

    #FEATURE SELECTION
    #x, x_test = kBest(x, y, x_test, f_classif, k_features)

    return f_train, f_test

def CV_score(x, y, model, model_args, k_features=None, report=False):
    kf = KFold(n_splits=N_SPLITS, shuffle=False)
    total_score = 0
    y_pred_all = np.zeros(y.size, dtype=int)
    for train_idx, test_idx in kf.split(x):
        x_train_kf, x_test_kf = preprocess(
            x[train_idx], y[train_idx], x[test_idx], k_features)

        print('Training model')
        y_pred = model(x_train_kf, y[train_idx], x_test_kf, model_args)
        y_pred_all[test_idx] = y_pred
        total_score += balanced_accuracy_score(y[test_idx], y_pred)
    if report:
        print(classification_report(y, y_pred_all))
    return total_score/N_SPLITS

def Chain_CRF(x, y, x_test, model_args):
    # Reshape for CRF
    #svc = SVC(class_weight='balanced', kernel='rbf', decision_function_shape='ovr')
    #svc.fit(x, y)
    #x = svc.decision_function(x)
    #x_test = svc.decision_function(x_test)
    #scaler = StandardScaler().fit(x)
    #x = scaler.transform(x)
    #x_test = scaler.transform(x_test)
    x = x[:,:10]
    x_test = x_test[:,:10]
    x = x.reshape(-1, 21600, x.shape[-1])
    x_test = x_test.reshape(-1, 21600, x.shape[-1])
    y = y.reshape(-1, 21600)
    crf = ChainCRF(directed=True)
    ssvm = OneSlackSSVM(model=crf, C=model_args['C'], max_iter=model_args['max_iter'])
    ssvm.fit(x, y)
    y_pred = np.array(ssvm.predict(x_test))
    return y_pred.flatten()

def SVM(x, y, x_test, model_args):
    svc = SVC(class_weight='balanced', kernel='rbf', C=model_args['C'],
            gamma=model_args['gamma'])
    svc.fit(x, y)
    return svc.predict(x_test)

def GBC(x, y, x_test, model_args):
    gbc = GradientBoostingClassifier(
            n_estimators = model_args['n_estimators'],
            max_depth = model_args['max_depth']
    )
    gbc.fit(x, y)
    return gbc.predict(x_test)

if args.test:
    f_train, f_test = extract_features(x_train, x_test)
    f_train, f_test = preprocess(f_train, y_train, f_test)
    print('Running on test data')
    y_pred = SVM(f_train, y_train, f_test, {'C': 0.5, 'gamma': 'auto'})
    #y_pred = Chain_CRF(f_train, y_train, f_test, {'C': 0.01, 'max_iter':1000})

    y_pred_ids = np.hstack((
        np.arange(y_pred.size).reshape(-1, 1),
        y_pred.reshape(-1, 1) + 1 # Labels start at 1
    ))
    np.savetxt(fname='y_test.csv', header='Id,y', delimiter=',', X=y_pred_ids,
            fmt=['%d', '%d'], comments='')

elif args.finetune:
    f_train, f_test = extract_features(x_train, x_test)
    scores = []
    for C in [0.5, 1, 2, 5, 10]:
        for gamma in ['scale', 'auto', 0.01, 0.001]:
            print(C, gamma)
            model = SVM
            scores.append(CV_score(f_train, y_train, model, {'C':C, 'gamma':gamma}))
            print(scores[-1])
    np.savetxt(fname='finetuning.csv', header='C,gamma,score', delimiter=',',
            X=scores, fmt=['%.1f', '%.3f', '%.5f'], comments='')
else:
    #model = GBC
    f_train, f_test = extract_features(x_train, x_test)
    #model_args = {'n_estimators': 200, 'max_depth': 3}
    model = Chain_CRF
    for C in [0.005, 0.008, 0.01]:
        print(C)
        model_args = {'C': C, 'max_iter': 1000}
        print(CV_score(f_train, y_train, model, model_args))
    #model_args = {'C': 0.1, 'max_iter': 100}
    #avg_score = CV_score(f_train, y_train, model, model_args)
    #print('Average BMAC:', avg_score)
