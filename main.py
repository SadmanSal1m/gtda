import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA

# Upload the dataset CSV file
    st.subheader('Upload Dataset')
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        # Read the CSV file
        data = pd.read_csv(file, encoding='ISO-8859-1', low_memory=False)
        
# Set page title and layout
st.set_page_config(page_title='Global Terrorism Analysis', layout='wide')

# Sidebar widgets
analysis_options = ['Prediction of Terrorist Attacks', 'Geospatial Analysis', 'Temporal Analysis',
                    'Attack Types and Targets', 'Investigation and Analysis']
analysis_choice = st.sidebar.selectbox('Select Analysis', analysis_options)

# Perform analysis based on user selection
if analysis_choice == 'Prediction of Terrorist Attacks':
    st.title('Prediction of Terrorist Attacks')

    # Select relevant features and target variable
    features = ['iyear', 'imonth', 'iday', 'country_txt', 'region', 'latitude', 'longitude',
                'attacktype1', 'targtype1', 'weaptype1', 'nkill']
    target = 'success'

    # Preprocess the data
    df = data[features + [target]].copy()
    df.dropna(inplace=True)  # Remove rows with missing values
    df = pd.get_dummies(df, columns=['country_txt', 'attacktype1'])  # One-hot encode categorical features

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target],
                                                        test_size=0.2, random_state=42)

    # Build a random forest classifier model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Input box for user to type the name of a country
    country_name = st.text_input("Enter the name of a country", "")

    # Input box for user to type the attack type
    attack_type = st.text_input("Enter the attack type", "")

    # Radio button to choose past or future prediction
    prediction_type = st.radio("Select prediction type", ("Past Years", "Future Years"))

    if prediction_type == "Past Years":
        # Slider for past years
        start_year, end_year = st.slider("Select the year range for past years", min_value=int(data['iyear'].min()),
                                         max_value=int(data['iyear'].max()),
                                         value=(int(data['iyear'].min()), int(data['iyear'].max())))
        future_start_year = None
        future_end_year = None
    else:
        # Slider for future years
        future_start_year, future_end_year = st.slider("Select the year range for future years",
                                                       min_value=int(data['iyear'].max()) + 1,
                                                       max_value=int(data['iyear'].max()) + 10,
                                                       value=(int(data['iyear'].max()) + 1, int(data['iyear'].max()) + 5))
        start_year = None
        end_year = None

    # Define function for future prediction

    def estimate_future_attacks(filtered_data, end_year):
        if len(filtered_data) == 0:
            return pd.DataFrame()  # Return an empty DataFrame if no data is available for future predictions

        # Calculate the average number of attacks per year for the selected country and attack type
        avg_attacks_per_year = len(filtered_data) / (filtered_data['iyear'].max() - filtered_data['iyear'].min() + 1)

        # Generate future years for prediction
        future_years = np.arange(end_year + 1,
                                 end_year + 6)  # Predict for the next 5 years starting from the selected end year

        # Create new data points for future years
        num_samples = len(future_years) * int(avg_attacks_per_year)  # Calculate the required number of samples

        if 'country_txt' in filtered_data.columns:
            country_col = 'country_txt'
        elif 'country' in filtered_data.columns:
            country_col = 'country'
        else:
            st.error("Country column not found in the DataFrame.")
            return

        future_data = pd.DataFrame({
            'iyear': np.repeat(future_years, num_samples),
            'imonth': np.tile(filtered_data['imonth'].sample(num_samples, replace=True).unique(), len(future_years)),
            country_col: np.repeat(filtered_data[country_col].sample(num_samples, replace=True).unique(),
                                   len(future_years)),
            'region': np.repeat(filtered_data['region'].sample(num_samples, replace=True).unique(), len(future_years)),
            'latitude': np.repeat(filtered_data['latitude'].sample(num_samples, replace=True).unique(),
                                  len(future_years)),
            'longitude': np.repeat(filtered_data['longitude'].sample(num_samples, replace=True).unique(),
                                   len(future_years)),
            'attacktype1': np.repeat(filtered_data['attacktype1'].sample(num_samples, replace=True).unique(),
                                     len(future_years)),
            'targtype1': np.repeat(filtered_data['targtype1'].sample(num_samples, replace=True).unique(),
                                   len(future_years)),
            'weaptype1': np.repeat(filtered_data['weaptype1'].sample(num_samples, replace=True).unique(),
                                   len(future_years)),
            'nkill': np.repeat(filtered_data['nkill'].mean(), num_samples)
        })

        future_data = pd.get_dummies(future_data,
                                     columns=[country_col, 'attacktype1'])  # One-hot encode categorical features

        return future_data


    if st.button('Predict'):
        # Validate year range selection
        if (start_year is not None and end_year is not None) and (
                future_start_year is not None and future_end_year is not None):
            st.error("Invalid year range. Please select a valid range.")
        else:
            # Filter data based on country name and attack type
            filtered_data = data[(data['country_txt'] == country_name) & (data['attacktype1_txt'] == attack_type)]

            # Preprocess the filtered data
            filtered_data = filtered_data[features].copy()
            # Remove rows with missing values
            filtered_data.dropna(inplace=True)
            # One-hot encode categorical features
            filtered_data = pd.get_dummies(filtered_data, columns=['country_txt',
                                                                   'attacktype1'])

            if filtered_data.empty:
                st.write("No data available for the given country and attack type.")
            else:
                if start_year and end_year:
                    if end_year <= data['iyear'].max():
                        # Filter data based on the selected year range
                        filtered_data = filtered_data[
                            (filtered_data['iyear'] >= start_year) & (filtered_data['iyear'] <= end_year)]
                        if filtered_data.empty:
                            st.write("No data available for the given timeframe.")
                        else:
                            # Align feature names with training data
                            missing_features = set(X_train.columns) - set(filtered_data.columns)
                            for feature in missing_features:
                                filtered_data[feature] = 0
                            filtered_data = filtered_data[X_train.columns]

                            prediction = model.predict_proba(filtered_data)[:, 1]  # Get the probability of success

                            # Display prediction result
                            st.subheader('Prediction Result')
                            st.write('Country:', country_name)
                            st.write('Attack Type:', attack_type)
                            st.write('The chances of', country_name, 'getting attacked by', attack_type,
                                     'in the years', start_year, 'to', end_year, 'were',
                                     round(prediction.mean() * 100, 2), '%')
                else:
                    # Prepare data for time series forecasting
                    time_series_data = filtered_data.groupby('iyear').size().reset_index(name='count')
                    time_series_data = time_series_data.set_index('iyear')

                    # Fit ARIMA model
                    model = ARIMA(time_series_data, order=(1, 1, 1))
                    model_fit = model.fit()

                    # Generate future years for prediction
                    future_years = np.arange(future_start_year, future_end_year + 1)

                    # Make predictions for future years
                    forecast = model_fit.forecast(steps=len(future_years))

                    # Display prediction result
                    st.subheader('Prediction Result')
                    st.write('Country:', country_name)
                    st.write('Attack Type:', attack_type)
                    st.write('The predicted number of attacks in', country_name, 'in the years', future_start_year, 'to'
                             , future_end_year, 'by', attack_type, 'is',
                             round(forecast.mean(), 2))

elif analysis_choice == 'Geospatial Analysis':
    st.title('Geospatial Analysis')

    # Filter and add markers for terrorist incidents
    filtered_data = data.dropna(subset=['latitude', 'longitude'])

    # Create a folium map centered on the average location of incidents
    map_center = [filtered_data['latitude'].mean(), filtered_data['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=2)
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers for each incident
    for idx, row in filtered_data.iterrows():
        folium.Marker([row['latitude'], row['longitude']]).add_to(marker_cluster)

    # Display the map
    st.subheader('Global Terrorism Incidents Map')
    st.write(m)

    # Add geospatial filters
    st.sidebar.markdown('### Map Filters')
    year_range = st.sidebar.slider('Select Year Range', int(data['iyear'].min()), int(data['iyear'].max()), (2000, 2010))
    filtered_data = filtered_data[(filtered_data['iyear'] >= year_range[0]) & (filtered_data['iyear'] <= year_range[1])]
    st.write(filtered_data)

elif analysis_choice == 'Temporal Analysis':
    st.title('Temporal Analysis')

    # Select relevant temporal features
    temporal_features = ['iyear', 'imonth']  # Update with actual temporal features

    # Display temporal filter options
    st.sidebar.markdown('### Temporal Filters')
    year_range = st.sidebar.slider('Select Year Range', int(data['iyear'].min()), int(data['iyear'].max()), (2000, 2010))
    month_range = st.sidebar.slider('Select Month Range', 1, 12, (1, 12))

    # Filter data based on temporal filters
    filtered_data = data[
        (data['iyear'] >= year_range[0]) & (data['iyear'] <= year_range[1]) &
        (data['imonth'] >= month_range[0]) & (data['imonth'] <= month_range[1])
    ]

    # Display filtered data
    st.subheader('Filtered Data')
    st.write(filtered_data)

    # Temporal analysis and visualization
    st.subheader('Temporal Visualization')

    # Group data by year and count incidents
    incidents_by_year = filtered_data.groupby('iyear').size().reset_index(name='incident_count')

    # Line plot of incidents by year
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=incidents_by_year, x='iyear', y='incident_count')
    plt.xlabel('Year')
    plt.ylabel('Incident Count')
    plt.title('Terrorism Incidents Over Time')
    st.pyplot(plt)

elif analysis_choice == 'Attack Types and Targets':
    st.title('Attack Types and Targets')

    # Select relevant features for attack types and targets
    attack_type_feature = 'attacktype1_txt'  # Update with actual attack type feature
    target_feature = 'targtype1_txt'  # Update with actual target feature

    # Display attack type and target filter options
    st.sidebar.markdown('### Attack Type and Target Filters')
    selected_attack_types = st.sidebar.multiselect('Select Attack Types', data[attack_type_feature].unique())
    selected_targets = st.sidebar.multiselect('Select Targets', data[target_feature].unique())

    # Filter data based on attack type and target filters
    filtered_data = data[
        (data[attack_type_feature].isin(selected_attack_types)) &
        (data[target_feature].isin(selected_targets))
    ]

    # Display filtered data
    st.subheader('Filtered Data')
    st.write(filtered_data)

    # Attack types and targets visualization
    st.subheader('Attack Types and Targets Visualization')

    if not filtered_data.empty:
        # Countplot of attack types
        plt.figure(figsize=(10, 6))
        sns.countplot(data=filtered_data, x=attack_type_feature)
        plt.xlabel('Attack Type')
        plt.ylabel('Count')
        plt.title('Distribution of Attack Types')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Countplot of targets
        plt.figure(figsize=(10, 6))
        sns.countplot(data=filtered_data, x=target_feature)
        plt.xlabel('Target')
        plt.ylabel('Count')
        plt.title('Distribution of Targets')
        plt.xticks(rotation=45)
        st.pyplot(plt)
    else:
        st.write("No data available for the selected attack types and targets.")

elif analysis_choice == 'Investigation and Analysis':
    st.title('Investigation and Analysis')

    # Select relevant features for Investigation and Analysis
    demographic_features = ['country_txt', 'region_txt', 'provstate', 'city', 'gname', 'nkill', 'nwound', 'weapdetail']
    selected_features = st.multiselect('Select Features', demographic_features)

    # Filter data based on selected features
    filtered_data = data[selected_features]

    # Display filtered data
    st.subheader('Filtered Data')
    st.write(filtered_data)

    # Investigation and Analysis
    if 'nkill' in selected_features:
        st.subheader('Number of Kills')
        st.write('Average number of kills:', filtered_data['nkill'].mean())
        st.write('Maximum number of kills:', filtered_data['nkill'].max())

        # Grouping and aggregation by country
        if 'country_txt' in filtered_data.columns:
            kills_by_country = filtered_data.groupby('country_txt')['nkill'].sum()
            st.subheader('Number of Kills by Country')
            st.write(kills_by_country)

            # Plotting kills by country
            st.subheader('Kills by Country - Bar Chart')
            st.bar_chart(kills_by_country)
        else:
            st.write('Selected dataset does not contain "country_txt" column.')

    if 'nwound' in selected_features:
        st.subheader('Number of Wounded')
        st.write('Average number of wounded:', filtered_data['nwound'].mean())
        st.write('Maximum number of wounded:', filtered_data['nwound'].max())

        # Grouping and aggregation by region
        st.subheader('Number of Wounded by Region')
        wounded_by_region = filtered_data.groupby('region_txt')['nwound'].sum()
        st.write(wounded_by_region)

        # Plotting wounded by region
        st.subheader('Wounded by Region - Pie Chart')
        st.pie(wounded_by_region, labels=wounded_by_region.index)

    if 'gname' in selected_features:
        st.subheader('Terrorist Groups')
        terrorist_groups = filtered_data['gname'].unique()
        st.write('Total number of unique terrorist groups:', len(terrorist_groups))
        st.write('List of terrorist groups:', terrorist_groups)

        # Grouping and aggregation by terrorist group
        st.subheader('Number of Incidents by Terrorist Group')
        incidents_by_group = filtered_data['gname'].value_counts()
        st.write(incidents_by_group)

        # Plotting incidents by terrorist group
        st.subheader('Incidents by Terrorist Group - Horizontal Bar Chart')
        st.bar_chart(incidents_by_group)

    if 'weapdetail' in selected_features:
        st.subheader('Weapon Details')
        weapon_details = filtered_data['weapdetail'].unique()
        st.write('Total number of unique weapon details:', len(weapon_details))
        st.write('List of weapon details:', weapon_details)

        # Grouping and aggregation by weapon detail
        incidents_by_weapon = filtered_data['weapdetail'].value_counts()
        st.subheader('Number of Incidents by Weapon Detail')
        st.write(incidents_by_weapon)

        # Plotting incidents by weapon detail
        st.subheader('Incidents by Weapon Detail - Vertical Bar Chart')
        st.bar_chart(incidents_by_weapon)

# Additional interactive features
st.sidebar.markdown('### Additional Options')
show_data = st.sidebar.checkbox('Show Dataset')
if show_data:
    st.subheader('Global Terrorism Dataset')
    st.write(data)
