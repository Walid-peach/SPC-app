import streamlit as st
from scipy import stats
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import hydralit_components as hc


st.set_page_config(layout='wide',page_title='Statistical Process Control application')
# specify the primary menu definition
menu_data = [
        {'icon': "far fa-copy", 'label':"Insert Data manually"},
        {'icon': "far fa-chart-bar", 'label':"Insert File"},#no tooltip message
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
    st.title("Welcome to Statistical Process Control application")

if menu_id == "Home":
        st.title('Home')
        st.info('This is the home page.')     
        
    
elif menu_id =="Insert Data manually":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    data_load_state = st.text('Enter data please!')

    data_raw = st.text_input("data raw")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add row"):
            get_data().append(data_raw)

    with col2:
        if st.button("clear last row"):
            get_data().pop()
    st.dataframe(get_data())        
            
    # Test normality of data distribution
    
    if st.button("Test normality"):
        test_normality(np.array(get_data()).astype(np.float))

    #---------------------------------------------------
    #Plot the histogram
    
    if st.button("Dessiner l'Histogramme"):    
        dessiner_histogramme(np.array(get_data()).astype(np.float))
    if st.button("Normal test valeur"):    
        st.write(stats.normaltest(np.array(get_data()).astype(np.float)))

elif menu_id =="Insert File":
    
    uploaded_file = st.file_uploader("Choose a XLSX file")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, engine= 'openpyxl')
        st.dataframe(df)

    
    if st.button("Test normality"):
        test_normality(df['Diameter'])
    
    if st.button("Dessiner l'Histogramme"):    
        dessiner_histogramme(df['Diameter'])
    
    if st.button("Normal test valeur"):    
        st.write(stats.normaltest(df['Diameter']))
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Calculer Cp"):
            st.write(Cp(df['Diameter'], 74.05,73.95))
            

    with col4:
        if st.button("Calculer Cpk"):
           st.write(Cpk(df['Diameter'], 74.05,73.95))

    