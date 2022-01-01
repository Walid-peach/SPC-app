import streamlit as st
from scipy import stats
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import hydralit_components as hc
from PIL import Image


st.set_page_config(layout='wide',page_title='Statistical Process Control application')
# specify the primary menu definition
menu_data = [
        {'icon': "far fa-copy", 'label':"Inserer les données manuellement"},
        {'icon': "far fa-chart-bar", 'label':"Inserer un fichier"},#no tooltip message
        {'icon': "far fa-address-book", 'label':"About us"}
]
# we can override any part of the primary colors of the menu
#over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
over_theme = {'txc_inactive': '#FFFFFF'}

def navigation():
    try:
        menu_id=hc.nav_bar(
                            menu_definition=menu_data,
                            override_theme=over_theme,
                            home_name='Home',
                            hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
                            sticky_nav=True, #at the top or not
                            sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
                        )
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return menu_id

@st.cache(allow_output_mutation=True)
def get_data():
    return []

def test_normality(df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111) # 111 is equivalent to nrows=1, ncols=1, plot_number=1.
    prob = stats.probplot(df, dist=stats.norm, plot=ax1)
    ax1.set_xlabel('')
    ax1.set_title('Probability plot against normal distribution')
    plt.show()
    st.pyplot()

def dessiner_histogramme(df):
        st.subheader("l'Histogramme")
        plt.hist(df)
        plt.show()
        st.pyplot()
#Cp function
def Cp(x, usl, lsl):
    sigma = x.std()
    Cp = (usl - lsl) / (6*sigma)
    return Cp
#Cpk function
def Cpk(x, usl, lsl):
    sigma = x.std()
    m = x.mean()
    Cpu = (usl - m) / (3*sigma)
    Cpl = (m - lsl) / (3*sigma)
    Cpk = np.min([Cpu, Cpl] )
    return Cpk

if st.sidebar.button('click me too'):
  st.info('You clicked at: {}'.format(datetime.datetime.now()))

#-----------------------------------------------------------------------------------
#----------------------------------------RUN----------------------------------------
menu_id=navigation()
header = st.container()
with header:
    st.markdown("<h1 style='text-align: center; color: #ff4d4d  ;'>Bienvenue sur l'application du MSP</h1>", unsafe_allow_html=True)

if menu_id == "Home":
    st.title('MAÎTRISE STATISTIQUE DES PROCÉDÉS :')    
    st.write("<h4>Un mode de gestion qui conduit à se rendre maître des outils de production pour satisfaire les besoins du client, en agissant à temps sur les facteurs techniques et humains responsables de la qualité.</h4>", unsafe_allow_html=True)
    st.write("<h3 style='color:  #2471a3;'>Les avantages de la MSP:</h3>", unsafe_allow_html=True)
    st.write("<ul><li><h4>Anticiper les problèmes</h4></li><li><h4>Améliorer la productivité</h4></li><li><h4>Eviter le sous ou le sur contrôle, uniquement réagir quand il le faut</h4></li></ul>", unsafe_allow_html=True)
    col9, col10 = st.columns(2)
    with col9:
        st.write("<h3 style='color:  #2471a3;'>La « loi normale » ou distribution gaussienne :</h3>", unsafe_allow_html=True) 
        st.write("<h4>C'est un des fondements de la MSP. La plupart des outils qui seront mis en place sont basés sur des propriétés de la loi normale. Il est donc indispensable d'en comprendre les propriétés fondamentales.La loi normale de moyenne nulle et d'écart type unitaire est appelée loi normale centrée réduite ou loi normale standard. Parmi les lois de probabilité, les lois normales prennent une place particulière grâce au théorème central limite.</h4>", unsafe_allow_html=True) 
    with col10:
        image = Image.open('loi_norm.png')
        st.image(image, caption='Loi Normale')    
        
    
elif menu_id =="Inserer les données manuellement":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    

    data_raw = st.text_input("Entrer les données Ici!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ajouter une ligne"):
            get_data().append(data_raw)

    with col2:
        if st.button("effacer la dernière ligne"):
            get_data().pop()
    data_load_state = st.text('Les données inserées')
    st.dataframe(get_data())        
            
    # Test normality of data distribution
    
    if st.button("Tester la normalité"):
        test_normality(np.array(get_data()).astype(np.float))

    #---------------------------------------------------
    #Plot the histogram
    
    if st.button("Dessiner l'Histogramme"):    
        dessiner_histogramme(np.array(get_data()).astype(np.float))
    if st.button("Normal test valeur"):    
        st.write(stats.normaltest(np.array(get_data()).astype(np.float)))

elif menu_id =="Inserer un fichier":
    
    uploaded_file = st.file_uploader("Importer un fichier Excel")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, engine= 'openpyxl')
        st.dataframe(df)

    
    if st.button("Tester la normalité"):
        test_normality(df['Diameter'])
    
    if st.button("Dessiner l'Histogramme"):    
        dessiner_histogramme(df['Diameter'])
    
    if st.button("Normal test valeur Khi 2"):    
        sta, pval = stats.normaltest(df['Diameter'])
        coll1,coll2,coll3 = st.columns(3)
        with coll1:
            st.write("p-value = " +str(pval))
        with coll2 :
            if pval > 0.05 :
                st.write("<h3 style='color:  #ff4d4d;'>P-value > 0.05 d'où il suit la loi normale</h3>", unsafe_allow_html=True)
            else :
                st.write("<h3 style='color:  #ff4d4d;'>P-value < 0.05 d'où il suit pas la loi normale </h3>", unsafe_allow_html=True)
        
    
    
    if st.button("Calculer Cp et Cpk"):
        col3, col4,col5 = st.columns(3)
        with col3:
            cp=Cp(df['Diameter'], 74.05,73.95)
            cpk = Cpk(df['Diameter'], 74.05,73.95)
            st.write("Cp = "+str(cp))
            st.write("Cpk = "+str(cpk))
        with col4:
            if cp > 1.33 and cpk > 1.33:
                st.write("<h3 style='color:  #ff4d4d;'>D'où le procédé est capable</h3>", unsafe_allow_html=True)
            else :
                st.write("<h3 style='color:  #ff4d4d;'>D'où le procédé n'est pas capable </h3>", unsafe_allow_html=True)
        with col5:
            Rr=round(cpk/cp*100,2)
            st.write("le Rendement de reglage egale  "+str(Rr)+"%")
    
            
            