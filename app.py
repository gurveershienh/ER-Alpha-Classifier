import streamlit as st
import components

def app():
    st.set_page_config(
            page_title="ER-Alpha Classifier",
    )
    
    st.markdown('''
                ### :pill: Estrogen Receptor-α Ligand Classifier 
                ---
                ''')
    tab1, tab2, tab3, tab4 = st.tabs(["Ligand Classifier", 'Synopsis', "Model Details", 'Supplementary Info'])

    with tab1:
        components.ligand_qsar()
    with tab2:
        components.about()
    with tab3:
        components.model_info()
    with tab4:
        components.app_info()
        
        
        pass

if __name__ =='__main__':
    app()
