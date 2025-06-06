import streamlit as st
import pandas as pd
import joblib
from datetime import datetime



model = joblib.load('Mobile_model.pkl')
metadata = joblib.load('mobile_metadata.joblib')
feature_name = metadata['feature_name']
target_name = metadata['target_name']

DATA_FILE = 'mobile_prediction.csv'



try:
    df1 = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    df1 = pd.DataFrame(columns=pd.Series(feature_name).tolist() + ['prediction','timestamp'])

st.title('MOBILE PHONE PRICE RANGE CLASSIFICATION')
tab,tab1,tab2 = st.tabs(['Data Analysis','Auto Mobile Prediction','Manual Inputs For User'])

with tab:
    st.header('Preview of data')


    
    st.markdown("Feature Description \n- Battery_Power : The phone battery power in mah \n- Blue : If the phone has bluetooth or not \n- Clock_speed : The speed at which microprocessor executes instruction \n- Dual_sim : If the phone has Dual_sim or no \n- Fc : Front Camera mega pixel \n- Four_g : Has 4G support or no \n- Int_memory : Internal Memory in Gigabyte \n- M_dep : Mobile Depth in c \n- Mobile_wt : Weight of mobile phon \n- n_cores : Number of cores of processo \n- Pc :  Primary camera megapixel \n- Px_height : Height of the display screen in pixel \n- Px_width : Width of the display screen in pixel \n- Ram : Random Access Memory (in MB \n- Sc_h : Screen height in some unit(cm \n- Sc_w : Screen width in some unit(cm \n- Talk_time : Maximum talk time on a single battery charge (in hours \n- Three_g : 3G support or no \n- Touch_screen : Is a touch screen or no \n- Wifi : Has wifi or no \n- Price_range : The price range of the phone from low cost, medium_cost, high_cost, very_high_cost")

    uploaded_file = st.file_uploader('Choose a csv file',type=['csv'],key='upload')

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success('The file was uploaded successfully')
            st.subheader('Dataset')
            st.write(df.head())
            st.subheader('Dataset Analysis')
            st.write(df.describe())
        except Exception as e:
            st.error(f'They was an error processing the file {e}')

with tab1:
    uploaded_file1 = st.file_uploader('Choose a csv file',type=['csv'],key='upload1')
    if uploaded_file1 is not None:
        df = pd.read_csv(uploaded_file1)
        prediction = model.predict(df)
        df['prediction'] = prediction
        st.dataframe(df)
    else:
        st.warning('Please upload a CSV file to proceed.')
with tab2:
    inputs = {}
    

    col1, col2, col3 = st.columns(3)
    with col1:
        inputs['battery_power'] = st.number_input('Battery Power (mAh)', min_value=500, max_value=5000, value=2000)
    with col2:
        inputs['blue'] = st.selectbox('Bluetooth', options=[0, 1])
    with col3:
        inputs['clock_speed'] = st.number_input('Clock Speed (GHz)', min_value=0.5, max_value=3.0, value=2.0, step=0.1)

    
    col4, col5, col6 = st.columns(3)
    with col4:
        inputs['dual_sim'] = st.selectbox('Dual SIM', options=[0, 1])
    with col5:
        inputs['fc'] = st.number_input('Front Camera (MP)', min_value=0, max_value=20, value=5)
    with col6:
        inputs['four_g'] = st.selectbox('4G Support', options=[0, 1])

    
    col7, col8, col9 = st.columns(3)
    with col7:
        inputs['int_memory'] = st.number_input('Internal Memory (GB)', min_value=2, max_value=256, value=64)
    with col8:
        inputs['m_dep'] = st.number_input('Mobile Depth (cm)', min_value=0.1, max_value=1.0, value=0.5, step=0.01)
    with col9:
        inputs['mobile_wt'] = st.number_input('Mobile Weight (g)', min_value=80, max_value=300, value=180)

    col10, col11, col12 = st.columns(3)
    with col10:
        inputs['n_cores'] = st.number_input('The numbers of cores', min_value=1, max_value=8, value=3)
    with col11:
        inputs['pc'] = st.number_input('primary camera (MP)', min_value=0, max_value=20, value=3)
    with col12:
        inputs['px_height'] = st.number_input('The px height ', min_value=0, max_value=2000, value=3)

    col13, col14, col15 = st.columns(3)
    with col13:
        inputs['px_width'] = st.number_input('The px width', min_value=500, max_value=2000, value=1000)
    with col14:
        inputs['ram'] = st.number_input('ram amount (MB) ', min_value=256, max_value=4000, value=1000)
    with col15:
        inputs['sc_h'] = st.number_input('The screen height ', min_value=5, max_value=20, value=10)
    
    col16, col17, col18 = st.columns(3)
    with col16:
        inputs['sc_w'] = st.number_input('The screen width', min_value=0, max_value=20, value=10)
    with col17:
        inputs['talk_time'] = st.number_input('The talk time', min_value=2, max_value=20, value=3)
    with col18:
        inputs['three_g'] = st.selectbox('3G',options=[0,1])

    col19, col20 = st.columns(2)
    with col19:
        inputs['touch_screen'] = st.selectbox('Touch screen', options=[0,1])
    with col20:
        inputs['wifi'] =  st.selectbox('Wifi', options=[0,1])
    
    input_df = pd.DataFrame([inputs])

    submitted = st.button('predict')

    if submitted:
        prediction = model.predict(input_df)
        if prediction == 0:
            st.success(f'The price range is Low Cost')
        elif prediction == 1:
            st.success(f'The price range is Medium Cost')
        elif prediction == 2:
            st.success(f'The price range is High Cost')
        else:
            st.success(f'The price range is Very High Cost')

        def prediction():
            prediction = model.predict(input_df)
            if prediction == 0:
                p = f'Low Cost'
            elif prediction == 1:
                p = f'Medium Cost'
            elif prediction == 2:
                p = f'High Cost'
            else:
                p = f'Very High Cost'

            return p
        
        p = prediction()

        # Add to Dataframe
        new_row = {**inputs, 'prediction': p, 'timestamp': datetime.now()}
        df1 = pd.concat([df1, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to CSV
        df1.to_csv(DATA_FILE, index=False)
        st.info('Prediction saved to database')

    if st.checkbox('Show collected data'):
        st.dataframe(df1)

    if st.button('Download collected data as CSV'):
        st.download_button(
            label = 'Download CSV',
            data = df1.to_csv(index=False).encode('utf-8'),
            file_name = 'mobile_prediction.csv',
            mime='text/csv'
        )


