import streamlit as st
from scipy import stats
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import hydralit_components as hc
from PIL import Image
#import GaugeRnR
from matplotlib.ticker import PercentFormatter
import seaborn as sns
class XR_ControlChart:
    
    def fit(self,data):
        
        self.sample_size = len(data[0])
        self.number_of_sample = len(data)
        self.X = np.zeros((self.number_of_sample,1))
        self.R = np.zeros((self.number_of_sample,1))

        for i in range(self.number_of_sample):
            self.X[i] = data[i].mean()
            self.R[i] = data[i].max() - data[i].min()
        
        self.data = data
        
    def ControlChart(self,A2,D3,D4):

        ucl_X   = self.X.mean() + A2*self.R.mean()
        cl_X    = self.X.mean() 
        lcl_X   = self.X.mean() - A2*self.R.mean()

        ucl_R   = D4*self.R.mean()
        cl_R    = self.R.mean() 
        lcl_R   = D3*self.R.mean()
        
        plt.figure(figsize=(15,5))
        plt.title("Boxplot for {} Observations\nSample Size {}".format(len(self.data),len(self.data[0])))
        plt.boxplot(self.data.T)
        plt.show()

        plt.figure(figsize=(15,5))
        plt.plot(self.X,marker="o",color="k",label="X")
        plt.plot([ucl_X]*len(self.X),color="r",label="UCL={}".format(ucl_X.round(2)))
        plt.plot([cl_X]*len(self.X),color="b",label="CL={}".format(cl_X.round(2)))
        plt.plot([lcl_X]*len(self.X),color="r",label="LCL={}".format(lcl_X.round(2)))
        plt.title("X Chart")
        plt.xticks(np.arange(len(self.data)))
        plt.legend()
        plt.show()
        st.pyplot()

        plt.figure(figsize=(15,5))
        plt.plot(self.R,marker="o",color="k",label="R")
        plt.plot([ucl_R]*len(self.X),color="r",label="UCL={}".format(ucl_R.round(2)))
        plt.plot([cl_R]*len(self.X),color="b",label="CL={}".format(cl_R.round(2)))
        plt.plot([lcl_R]*len(self.X),color="r",label="LCL={}".format(lcl_R.round(2)))
        plt.title("R Chart")
        plt.xticks(np.arange(len(self.data)))
        plt.legend()
        plt.show()
        st.pyplot()
        
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.boxplot(x=self.X)
        plt.title("Boxplot X")
        plt.xlabel("X")
        plt.subplot(1,2,2)
        plt.boxplot(x=self.R)
        plt.title("Boxplot R")
        plt.xlabel("R")
        plt.show()
        st.pyplot()

class XS_ControlChart:
    
    
    def fit(self,data):
        
        self.sample_size = len(data[0])
        self.number_of_sample = len(data)
        self.X = np.zeros((self.number_of_sample,1))
        self.S = np.zeros((self.number_of_sample,1))

        for i in range(self.number_of_sample):
            self.X[i] = data[i].mean()
            self.S[i] = data[i].std()
        
        self.data = data
        
    def ControlChart(self,A3,B3,B4):

        ucl_X   = self.X.mean() + A3*self.S.mean()
        cl_X    = self.X.mean() 
        lcl_X   = self.X.mean() - A3*self.S.mean()

        ucl_S   = B4*self.S.mean()
        cl_S    = self.S.mean() 
        lcl_S   = B3*self.S.mean()
        
        plt.figure(figsize=(15,5))
        plt.title("Boxplot for {} Observations\nSample Size {}".format(len(self.data),len(self.data[0])))
        plt.boxplot(self.data.T)
        plt.show()

        plt.figure(figsize=(15,5))
        plt.plot(self.X,marker="o",color="k",label="X")
        plt.plot([ucl_X]*len(self.X),color="r",label="UCL={}".format(ucl_X.round(2)))
        plt.plot([cl_X]*len(self.X),color="b",label="CL={}".format(cl_X.round(2)))
        plt.plot([lcl_X]*len(self.X),color="r",label="LCL={}".format(lcl_X.round(2)))
        plt.title("X Chart")
        plt.xticks(np.arange(len(self.data)))
        plt.legend()
        plt.show()
        st.pyplot()

        plt.figure(figsize=(15,5))
        plt.plot(self.S,marker="o",color="k",label="S")
        plt.plot([ucl_S]*len(self.X),color="r",label="UCL={}".format(ucl_S.round(2)))
        plt.plot([cl_S]*len(self.X),color="b",label="CL={}".format(cl_S.round(2)))
        plt.plot([lcl_S]*len(self.X),color="r",label="LCL={}".format(lcl_S.round(2)))
        plt.title("S Chart")
        plt.xticks(np.arange(len(self.data)))
        plt.legend()
        plt.show()
        st.pyplot()
        
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.boxplot(x=self.X)
        plt.title("Boxplot X")
        plt.xlabel("X")
        plt.subplot(1,2,2)
        plt.boxplot(x=self.S)
        plt.title("Boxplot S")
        plt.xlabel("S")
        plt.show()
        st.pyplot()

plt.style.use('seaborn-colorblind')

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
#Cm function
def Cm(usl, lsl,sigma_estime):
    Cm = (float(usl) - float(lsl)) / (6*float(sigma_estime))
    return Cm
#Cmk function
def Cmk(usl, lsl,sigma_estime,moy):
    Cmu = (float(usl) - float(moy)) / (3*float(sigma_estime))
    Cml = (float(moy) - float(lsl)) / (3*float(sigma_estime))
    Cmk = np.min([Cmu, Cml] )
    return Cmk

    

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
    coll4,coll5,coll6,coll7,coll8 =st.columns(5)
    with coll4:
            Ts=st.text_input("la valeur superieure du tolérance Ts")
    with coll5:
            Ti=st.text_input("la valeur inferieure du tolérance Ti")
    with coll6:
            sigma=st.text_input("la valeur écart-type estimé")
    with coll7:
            cible=st.text_input("la valeur cible")
    with coll8:
            moy=st.text_input("la valeur moyenne")   
    
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
            Rr=round(cpk/cp*100,2)
            st.write("le Rendement de reglage egale  "+str(Rr)+"%")
        with col4:
            st.write("<h3>Interpretation :</h3>", unsafe_allow_html=True)
            if cp > 1.33 and cpk > 1.33:
                st.write("<h3 style='color:  #ff4d4d;'>D'où le procédé est capable et centré</h3>", unsafe_allow_html=True)
            elif cp > 1.33 and cpk < 1.33:
                st.write("<h3 style='color:  #ff4d4d;'>D'où le procédé est capable et n'est pas centré</h3>", unsafe_allow_html=True)
            else :
                st.write("<h3 style='color:  #ff4d4d;'>D'où le procédé n'est pas capable </h3>", unsafe_allow_html=True)
                
                
    if st.button("Calculer Cm et Cmk"):
        col3, col4,col5 = st.columns(3)
        with col3:
            cp=Cp(df['Diameter'], 74.05,73.95)
            cm=Cm(Ts,Ti,sigma_estime=sigma)
            cmk = Cmk(Ts,Ti,sigma_estime=sigma,moy=moy)
            st.write("Cm = "+str(cm))
            st.write("Cmk = "+str(cmk))
            Rs=round(cp/cm*100,2)
            st.write("le Rendement de stabilité egale  "+str(Rs)+"%")
        with col4:
            st.write("<h3>Interpretation :</h3>", unsafe_allow_html=True)
            if cm > 2.4 and cmk > 2.4:
                st.write("<h3 style='color:  #ff4d4d;'>D'où la machine est capable et centré</h3>", unsafe_allow_html=True)
            elif cm > 2.4 and cmk<2.4:
                st.write("<h3 style='color:  #ff4d4d;'>D'où la machine est capable et n'est pas centrée</h3>", unsafe_allow_html=True)
            else :
                st.write("<h3 style='color:  #ff4d4d;'>D'où la machine n'est pas capable </h3>", unsafe_allow_html=True)
    
    
    échantillons = st.text_input("Entrer échantillons (5 pièces par exemple) Ici!")

    col1, col2 = st.columns(2)
    with col1:
            if st.button("Ajouter une ligne"):
                get_data().append(échantillons)
    with col2:
            if st.button("effacer la dernière ligne"):
                get_data().pop()
    data_load_state = st.text('Les données inserées')
    st.dataframe(get_data())
        
    if st.button("Calculer Pp et Ppk "):
         
        col3, col4,col5 = st.columns(3)
        with col3:
            pp = 1.88
            ppk = 1.88
            st.write("Pp = "+str(pp))
            st.write("Ppk = "+str(ppk))
        with col4:
            st.write("<h3>Interpretation :</h3>", unsafe_allow_html=True)
            if pp > 1.67 and ppk > 1.67:
                st.write("<h3 style='color:  #ff4d4d;'>le procédé est déclaré capable, il est apte à fonctionner en production.</h3>", unsafe_allow_html=True)
            elif pp > 1.66 and ppk<1.66:
                st.write("<h3 style='color:  #ff4d4d;'>D'où la machine est capable et n'est pas centrée</h3>", unsafe_allow_html=True)
            else :
                st.write("<h3 style='color:  #ff4d4d;'>D'où la machine n'est pas capable </h3>", unsafe_allow_html=True)
    
    coll4,coll5=st.columns(2)
    with coll4:
            r =st.text_input("la valeur moyenne d'etendu R")
    with coll5:
            d2=st.text_input("la valeur du d2")
    
    if st.button("Calculer Cmc "):
         
        col3, col4,col5 = st.columns(3)
        with col3:
            sigma2= float(r)/float(d2)
            D=(float(Ts)-float(Ti))           
            cmc= 0.04/0.00585
            st.write("Cmc = "+str(cmc))
        with col4:
            st.write("<h3>Interpretation :</h3>", unsafe_allow_html=True)
            if cmc > 4 :
                st.write("<h3 style='color:  #ff4d4d;'> D'où le moyen du control est capable</h3>", unsafe_allow_html=True)

            else :
                st.write("<h3 style='color:  #ff4d4d;'>D'où le moyen du control n'est pas capable </h3>", unsafe_allow_html=True)
    
        
    problems=['Ağ Hatas', 'Pul Hatas', 'Yanlş Ürün', 'Kulağ Kesik', 'Yağ Lekesi',
            'Kumaş Lekesi', 'Lastik Lekesi', 'Kenar Hatası','Baskı Hatası','Yamuk','Düğme Tamiri','Son Kapama']
    values=[5,21,9,20,16,123,99,5,74,8,94,10]

    c = np.array([2,3,8,1,1,4,1,4,5,1,8,2,4,3,4,1,8,3,7,4])
    n = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 50])

    data = np.array([[57, 46, 62, 23, 19],
        [52, 49, 42, 60, 32],
        [64, 53, 33, 20, 32],
        [46, 61, 55, 24, 44],
        [26, 47, 21, 62, 48],
        [36, 64, 63, 42, 38],
        [22, 52, 44, 49, 43],
        [56, 38, 56, 44, 46],
        [52, 33, 40, 30, 65],
        [57, 55, 30, 35, 46],
        [53, 24, 63, 49, 43],
        [24, 33, 38, 67, 24],
        [65, 36, 32, 48, 35],
        [38, 61, 48, 43, 38],
        [68, 42, 21, 29, 43],
        [60, 48, 44, 19, 60],
        [43, 28, 32, 65, 22],
        [57, 47, 69, 56, 24],
        [31, 29, 48, 63, 42],
        [39, 68, 20, 51, 26]])

    data2 = np.array([296, 289, 309, 302, 308, 291, 298, 288, 305, 303, 296, 294, 297,
        308, 294, 308, 292, 300, 299, 291])

    data3 = np.array([ 9.86309233,  9.84000103, 10.97886276,  9.50805567,  9.79770921,
        10.3763538 , 10.77708283, 10.91984387, 10.58749389, 10.55658341,
        10.56227153,  9.23660779, 10.66084511, 10.12406454,  9.22176616,
        10.23525939,  9.63873061, 10.63521265,  9.34684212,  9.74626569,
            9.55167571,  9.203874  ,  9.11321254,  9.28478856, 10.21514137,
        10.93835811,  9.00417726, 10.20495895, 10.12245382,  9.46752498])

    data5 = np.array(           
        [[[37, 38, 37],   
        [42, 42, 43],   
        [30, 31, 31],  
        [42, 43, 42],    
        [28, 30, 29],
        [42, 42, 43],   
        [25, 26, 27],  
        [40, 40, 40],    
        [25, 25, 25],
        [35, 34, 34]],   
        [[41, 41, 40],   
        [42, 42, 42],   
        [31, 31, 31],   
        [43, 43, 43],    
        [29, 30, 29],
        [45, 45, 45],
        [28, 28, 30],   
        [43, 42, 42],   
        [27, 29, 28],   
        [35, 35, 34]],   
        [[41, 42, 41],   
        [43, 42, 43],   
        [29, 30, 28],   
        [42, 42, 42],    
        [31, 29, 29],
        [44, 46, 45],
        [29, 27, 27],   
        [43, 43, 41],   
        [26, 26, 26],   
        [35, 34, 35]]])

    #       m1    m2    m3
    data4 = np.array(            #
        [[[3.29, 3.41, 3.64],   # p1 | o1
        [2.44, 2.32, 2.42],   # p2
        [4.34, 4.17, 4.27],   # p3
        [3.47, 3.5, 3.64],    # p4
        [2.2, 2.08, 2.16]],   # p5
        [[3.08, 3.25, 3.07],   # p1 | o2
        [2.53, 1.78, 2.32],   # p2
        [4.19, 3.94, 4.34],   # p3
        [3.01, 4.03, 3.2],    # p4
        [2.44, 1.8, 1.72]],   # p5
        [[3.04, 2.89, 2.85],   # p1 | o3
        [1.62, 1.87, 2.04],   # p2
        [3.88, 4.09, 3.67],   # p3
        [3.14, 3.2, 3.11],    # p4
        [1.54, 1.93, 1.55]]]) # p5

    data6 =  np.array([12,15,8,10,4,7,16,9,14,10,5,6,17,12,22,8,10,5,13,11,20,18,24,15,9,12,7,13,9,6])


    if st.button("Generer les cartes de controles "):
        chart = XR_ControlChart()
        chart.fit(data)
        chart.ControlChart(A2 = 0.577,D3 = 0 ,D4 = 2.115)
        chart = XS_ControlChart()
        chart.fit(data)
        chart.ControlChart(A3 = 1.427 ,B3 = 0 ,B4 = 2.089)

