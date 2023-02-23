import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chembl_webresource_client.new_client import new_client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from custom_funcs import computeFP

ensemble = {
    'rf': RandomForestClassifier,
    'svm': SVC,
    'mlp': MLPClassifier
}

def deploy_model(key,smiles):
    data = []
    with open(f'models/{key}_model.pkl', 'rb') as f:
        mol = Chem.MolFromSmiles(smiles)
        ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)]
        data += [ecfp6]
        model = pickle.load(f)
        res = model.predict(data)
    return res[0]

def deploy_ensemble(smiles):
    predictions = {}
    for key in ensemble:
        pred = deploy_model(key, smiles)
        predictions[key] = pred
    return predictions

def chembl_query_df(tar,activity):
    client = new_client.activity
    activities = client.filter(target_chembl_id=tar, standard_type=activity)
    if activities == []:
        return None
    df = pd.DataFrame.from_dict(activities)
    df = df[df.standard_value.notna()]
    mol_ids = [cid for cid in df.molecule_chembl_id]
    smiles = [mol for mol in df.canonical_smiles]
    std_vals = [float(val) for val in df.standard_value]
    labels =[]
    for val in std_vals:
        if val > 1000:
            labels.append(0)
        else:
            labels.append(1)
    print('active: ', labels.count(1))
    print('inactive: ', labels.count(0))
    tuples = list(zip(mol_ids,smiles,std_vals, labels))
    df = pd.DataFrame(tuples, columns=['chembl_id', 'smiles','value', 'labels'])
    df.drop_duplicates(subset='chembl_id')
    return df



        
        
def generate_tsne(data):
    df = pd.read_csv(data)
    X = df.drop('labels', axis=1)
    y = df['labels']



    tSNE_data = TSNE(n_components=3, n_jobs=-1, verbose=1).fit_transform(X)
    tSNE_x, tSNE_y, tSNE_z = list(zip(*tSNE_data))
    tSNE_fig = plt.figure(figsize=(8,8))
    ax1 = tSNE_fig.add_axes([0,0,1,1],projection='3d')
    ax1.grid(color='white')
    ax1.set_xlabel('tSNE-1')
    ax1.set_ylabel('tSNE-2')
    ax1.set_zlabel('tSNE-3')
    ax1.scatter(tSNE_x, tSNE_y, tSNE_z, s=60, c=y, cmap='Spectral', linewidth=1, edgecolor='black')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.zaxis.label.set_color('white')
    ax1.title.set_color('white')
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.tick_params(axis='z', colors='white')
    ax1.xaxis._axinfo["grid"]['color'] =  'white'
    ax1.yaxis._axinfo["grid"]['color'] =  'white'
    ax1.zaxis._axinfo["grid"]['color'] =  'white'

    tSNE_fig.patch.set_alpha(0)
    ax1.patch.set_alpha(0)
    with open('tsne.pkl', 'wb') as f:
        pickle.dump(tSNE_fig, f)
    tSNE_fig.savefig('tsne.png')
    plt.show()


if __name__ == '__main__':
    '''
    The code commented out below is an example of how each of the models were trained and validated.
    '''
    # df = pd.read_csv('data/target_ligand_ECFP6.csv')
    # X = np.array(df.drop('labels', axis=1))
    # y = np.array(df['labels'])
    # kf = StratifiedKFold(5, random_state=999, shuffle=True)
    # params = {
    #             "hidden_layer_sizes": [100, 1000, 2000, 5000], 
    #             "max_iter": [100, 1000, 5000],
    #             "alpha": [0.1,0.01,0.001]
    #         }
    # model = MLPClassifier(random_state=999)
    # search = GridSearchCV(model, params, cv=3, n_jobs=-1,verbose=4)
    # search.fit(X,y)
    # best_params = search.best_params_
    # print(best_params)
    # for train_ind, test_ind in kf.split(X,y):
    #     X_train, X_test = X[train_ind], X[test_ind]
    #     y_train, y_test = y[train_ind], y[test_ind]
    #     model = MLPClassifier(**best_params, random_state=999)
    #     model.fit(X_train,y_train)
    #     preds = model.predict(X_test)
    #     acc = accuracy_score(y_test, preds)
    #     f1 = f1_score(y_test, preds)
    #     mcc = matthews_corrcoef(y_test,preds)
    #     print('accuracy: ',acc)
    #     print('f1 score: ', f1)
    #     print('mcc: ', mcc)
    # trained_model = MLPClassifier(**best_params, random_state=999)
    # trained_model.fit(X,y)
    # with open('mlp_model.pkl', 'wb') as f:
    #     pickle.dump(trained_model, f)
    #     print('Model dumped')
        
    pass
    
