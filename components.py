import pandas as pd
import streamlit as st
import custom_funcs as ctk
import qsar
from random import randint

def ligand_qsar():
    st.markdown('''
                **Use AI models to identify potential drug candidates for ERα**

                ''')
    with st.container():
        prot_str='1ERE,1GWR,1QKT,5WGD,5WGQ,1VJB,1XYJ'
        prot_list=prot_str.split(',')
        idx = randint(0, len(prot_list)-1)
        
        ctk.render_prot(prot_list[idx],'cartoon',True, height=200)
    with st.form(key='qsar-form'):


        smi = st.text_input('Input compound in SMILES format')
        submit_job = st.form_submit_button('Predict')
            
        
        if submit_job and not ctk.valid_smiles(smi):
            st.error('Invalid SMILES')
        elif submit_job and ctk.valid_smiles(smi):
            pred_dict = qsar.deploy_ensemble(smi)
            predictions = list(pred_dict.values())
            st.write(f'**Inputted ligand and prediction details**')
            blk=ctk.makeblock(smi)
            if blk is None: 
                st.error('Invalid SMILES')
            else:
                ctk.render_mol(blk,'stick')
            if predictions.count(1) > predictions.count(0):
                st.success('Active compound (IC50 < 1000 nM)')
            else:
                st.warning('Inactive compound (IC50 > 1000 nM)')
            with st.expander('Model vote'):
                st.write(pred_dict)
                st.write('Active is designated to 1 and inactive is designated to 0')
        with st.expander('Sample ligand SMILES'):
            df = pd.read_csv('data/target_ligand_Ki.csv')
            df.set_index('chembl_id', inplace=True)

            st.write(df)
            st.write('All activity values are given in Ki. These ligands were not included in the training set and are known ligands of ERα')
    st.write('** Try removing or adding functional groups from SMILES and testing prediction. (e.g. (=O) for ketones or (-c2ccccc2) for a benzene group)')

def about():
    st.markdown('#### Application purpose')
    st.markdown('This Streamlit application is designed to facilitate the process of drug discovery for the human estrogen receptor alpha (ERα). ERα is a protein that plays a crucial role in various physiological processes, including reproductive development and breast cancer. By predicting the activity of compounds against ERα, we can identify potential drug candidates for the treatment of diseases associated with ERα dysfunction, such as breast cancer. Above is a visualization of the crystal structure of an ERα.')

    st.markdown('#### Explore machine learning for drug discovery:')
    st.markdown(''' 
            This application uses three different machine learning models to make predictions about the activity of compounds: **random forest, 
            support vector machines, and multilayer perceptron**. The predictions are based on a majority vote, where the majority of the predicted class (active/inactive)
            from the 3 models is the outputted prediction. A cut-off activity of 1000 nM IC50 was used, where only ligands with an IC50 value below 1000 nM are considered active. 
            Lower IC50 values indicate a higher potency of the compound and thus a higher likelihood of producing an effective drug candidate. 
                ''')
    st.image('data/tsne.png', caption='This figure is a 3D visualization of the model training data produced by the tSNE dimensionality reduction method. Each point represents a ligand colour coded as blue for active and red for inactive. Each cluster of points represent ligands that share a similar chemical space.')

    
    st.markdown('''
                To make accurate predictions about the activity of chemical compounds, this application uses canonical SMILES and ECFP6 molecular fingerprints. A canonical SMILES is a string of characters that represents the unique structure of a chemical compound in a standardized way, making it easier to compare and analyze compounds. ECFP6 fingerprints are a type of molecular descriptor that encodes information about the chemical structure of a compound, including the presence and arrangement of specific atoms and bonds. These fingerprints are useful for predicting the activity of compounds against specific targets, such as ERα, as they capture important structural features that can influence how the compound interacts with the target protein. By combining these two tools with machine learning models, this application can accurately predict the activity of chemical compounds against ERα to provide valuable insights for drug discovery in a way that is easily accessible to medicinal chemists.
                ''')

def model_info():
    with st.container():
        st.markdown('''
                    **Model Metrics**
                    ---
                    ''')
        with st.container():
            st.markdown('**Random Forest**')
            acc, f1, mcc = st.columns(3)
            with acc:
                st.metric('Accuracy', value ='85.3%')
            with f1:
                st.metric('F1-score', value ='88.7%')
            with mcc:
                st.metric('Matthews Correlation Coef.', value ='67.8%')
        with st.container():
            st.markdown('**Support Vector Machine**')
            acc, f1, mcc = st.columns(3)
            with acc:
                st.metric('Accuracy', value ='87.9%')
            with f1:
                st.metric('F1-score', value ='90.7%')
            with mcc:
                st.metric('Matthews Correlation Coef.', value ='73.7%')
        with st.container():
            st.markdown('**Multilayer Perceptron**')
            acc, f1, mcc = st.columns(3)
            with acc:
                st.metric('Accuracy', value ='86.3%')
            with f1:
                st.metric('F1-score', value ='89.7%')
            with mcc:
                st.metric('Matthews Correlation Coef.', value ='71.4%')
        st.markdown('''
                    All models were individually trained and tested using stratified 5-fold cross validation. The reported metrics are the means of the CV scores. 
                    ''')
        with st.container():   
            st.markdown('''
                        **Data collection and model training** 
                        ---
                        ''')
    
            st.markdown('''
            All data was collected from the ChEMBL database. Only ligands with reported IC50 values towards ERα were included within the training set. After data cleaning steps were applied, a total of 3901 ligands were obtained. 2462 of those ligands were active and 1439 were inactive. Ligands were then featurized by converting SMILES to ECFP6 fingerprints. Models were trained on all collected ligands for deployment.
            ''')

            
        with st.container():
            st.markdown(
                '''
                **Implementation details**
                ---
                This application was implemented in Anaconda 22.11.1 virtual environment. Data collection was done using the ChEMBL Web API. Pandas 1.5.2 was used for data processing and manipulation. All estimators were implemented and trained using sci-kit learn 1.2.0. 
                Molecular featurization was applied using RDKit 2022.9.4. Grid search hyperparameter optimization was used on a subset of model hyperparameters. Below is the list of adjusted hyperparameters used in each model. All other hyperparameters were kept default.
                
                '''
            )
            params = {
                'rf': {'criterion': 'gini', 'min_samples_leaf': 10, 'n_estimators': 100},
                'svm': {'C': 1, 'kernel': 'rbf'},
                'mlp': {'alpha': 0.1, 'max_iter': 100, 'hidden_layer_sizes': 100}
            }
            st.write('**Model hyperparameters**')
            st.write(params)
            


def app_info():
    st.markdown('''
                **Author info**
                ---
                Streamlit app and classification models were developed solely by me, Gurveer Singh Shienh. My LinkedIn profile can be found at www.linkedin.com/in/gurveer-shienh-8a09a2189. Source code can be found at https://github.com/gurveershienh
                
                ''')
    
    st.markdown('''
                **Acknowledgements**
                ---
                Below is a reference to the studies this tool was inspired from:
                
                Ahmed, M., Hasani, H.J., Kalyaanamoorthy, S. et al. GPCR_LigandClassify.py; 
                a rigorous machine learning classifier for GPCR targeting compounds. 
                Sci Rep 11, 9510 (2021). https://doi.org/10.1038/s41598-021-88939-5
                
                T. Lerksuthirat, S. Chitphuk, W. Stitchantrakul, D. Dejsuphong, A.A. Malik, C. Nantasenamat, PARP1PRED: A web server for screening the bioactivity of inhibitors against DNA repair enzyme PARP-1, EXCLI Journal (2023) DOI: https://doi.org/10.17179/excli2022-5602.
                ''')
    


