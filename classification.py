import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.ndimage.filters import uniform_filter1d
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from pystruct.learners import FrankWolfeSSVM
from pystruct.models import ChainCRF
from pystruct.models import LatentGraphCRF
from load_data import load_data
from features import power_features, PLF, onset_features

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

    f_train = np.hstack((pf, plf, onsets))
    f_test = np.hstack((pf_test, plf_test, onsets_test))

    return f_train, f_test

def preprocess(f_train, y, f_test, k_features=None):

    #SCALING
    scaler = StandardScaler().fit(f_train)
    f_train = scaler.transform(f_train)
    f_test = scaler.transform(f_test)

    #SMOOTHING
    n = 7
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

def CV_score(x, y, model, C, gamma, k_features=None, report=False):
    kf = KFold(n_splits=N_SPLITS, shuffle=False)
    total_score = 0
    y_pred_all = np.zeros(y.size, dtype=int)
    for train_idx, test_idx in kf.split(x):
        x_train_kf, x_test_kf = preprocess(
            x[train_idx], y[train_idx], x[test_idx], k_features)

        print('Training model')
        y_pred = model(x_train_kf, y[train_idx], x_test_kf, C, gamma)
        y_pred_all[test_idx] = y_pred
        total_score += balanced_accuracy_score(y[test_idx], y_pred)
    if report:
        print(classification_report(y, y_pred_all))
    return total_score/N_SPLITS

def Chain_CRF(x, y, x_test):
    # Reshape for CRF
    #svc = SVC(class_weight='balanced', kernel='rbf', decision_function_shape='ovr')
    #svc.fit(x, y)
    #x = svc.decision_function(x)
    #x_test = svc.decision_function(x_test)
    #scaler = StandardScaler().fit(x)
    #x = scaler.transform(x)
    #x_test = scaler.transform(x_test)
    x = x.reshape(-1, 21600, x.shape[-1])
    x_test = x_test.reshape(-1, 21600, x.shape[-1])
    y = y.reshape(-1, 21600)
    crf = ChainCRF()
    ssvm = FrankWolfeSSVM(model=crf, C=1, max_iter=10)
    ssvm = NSlackSSVM(model=crf, C=0.1, max_iter=10)
    ssvm.fit(x, y)
    y_pred = np.array(ssvm.predict(x_test))
    return y_pred.flatten()

def SVM(x, y, x_test, C, gamma):
    svc = SVC(class_weight='balanced', kernel='rbf', C=C, gamma=gamma)
    svc.fit(x, y)
    return svc.predict(x_test)

if args.test:
    x_train, x_test = preprocess(x_train, y_train, x_test)
    y_pred = np.array(model.predict(x_test)).flatten()

    y_pred_ids = np.hstack((
        np.arange(y_pred.size).reshape(-1, 1),
        y_pred.reshape(-1, 1) + 1 # Labels start at 1
    ))
    np.savetxt(fname='y_test.csv', header='Id,y', delimiter=',', X=y_pred_ids,
            fmt=['%d', '%d'], comments='')

elif args.finetune:
    f_train, f_test = extract_features(x_train, x_test)
    scores = []
    for C in [0.1, 0.5, 1, 5, 10, 20]:
        for gamma in ['auto', 'scale', 0.1, 0.01]:
            model = SVM
            scores.append(CV_score(f_train, y_train, model, C, gamma))
    np.savetxt(fname='finetuning.csv', header='C,gamma', delimiter=',',
            X=scores, fmt=['%.1f', '%.3f'], comments='')
else:
    model = SVM
    f_train, f_test = extract_features(x_train, x_test)
    avg_score = CV_score(f_train, y_train, model)
    print('Average BMAC:', avg_score)
