import streamlit as st
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header = st.container()

with header:
    st.title("Welcome to Statistical Process Control application")

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
#data = load_data(10000)
# Notify the reader that the data was successfully loaded.
#data_load_state.text('Loading data...done!')

st.subheader('Raw data')
#st.write(data)

#st.subheader('Number of pickups by hour')
#hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
#st.bar_chart(hist_values)

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
st.pyplot()


#disable this warning by disabling the config option: deprecation.showPyplotGlobalUse
st.set_option('deprecation.showPyplotGlobalUse', False)
st.subheader('Number of pickups by hour')
st.write(np.array(get_data()).astype(np.float))
plt.hist(np.array(get_data()).astype(np.float))
plt.show()
st.pyplot()

