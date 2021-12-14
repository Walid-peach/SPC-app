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
        {'icon': "far fa-copy", 'label':"Left End"},
        {'icon': "far fa-chart-bar", 'label':"Chart"},#no tooltip message
        {'icon': "far fa-address-book", 'label':"Book"}
]
# we can override any part of the primary colors of the menu
#over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
over_theme = {'txc_inactive': '#FFFFFF'}

if st.sidebar.button('click me too'):
  st.info('You clicked at: {}'.format(datetime.datetime.now()))


header = st.container()

with header:
    st.title("Welcome to Statistical Process Control application")
    hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    login_name='Logout',
    hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Enter data please!')
# Load 10,000 rows of data into the dataframe.
#data = load_data(10000)
# Notify the reader that the data was successfully loaded.
#data_load_state.text('Loading data...done!')

st.subheader('Raw data')


#disable this warning by disabling the config option: deprecation.showPyplotGlobalUse
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache(allow_output_mutation=True)
def get_data():
    return []

user_id = st.text_input("User ID")

col1, col2 = st.columns(2)
with col1:
    if st.button("Add row"):
        get_data().append(user_id)

with col2:
    if st.button("clear last row"):
        get_data().pop()
        
        
# Test normality of data distribution
fig = plt.figure()
ax1 = fig.add_subplot(111) # 111 is equivalent to nrows=1, ncols=1, plot_number=1.
prob = stats.probplot(np.array(get_data()).astype(np.float), dist=stats.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probability plot against normal distribution')
plt.show()
st.pyplot(height=800)



st.subheader('Number of pickups by hour')
st.write(np.array(get_data()).astype(np.float))
fig=plt.figure(figsize=(8,8))
plt.hist(np.array(get_data()).astype(np.float))
plt.show()
st.pyplot(fig, )


