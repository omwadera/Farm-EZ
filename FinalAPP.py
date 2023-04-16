from flask import Flask, render_template, request
from geopy.geocoders import Nominatim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

sns.set()

app = Flask(__name__)

raw_data = pd.read_csv('FinalDataset2.csv')
raw_data = raw_data.drop(['Latitude', 'Longitude'], axis=1)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/popup')
def popup():
    return render_template('popup.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/crop')
def home():
    return render_template('cindex.html')


@app.route('/chart', methods=['POST'])
def chart():
    
    raw_data['DISTRICT_NAME'] = raw_data['DISTRICT_NAME'].str.replace(' ', '')
    district = request.form['district']
    df = raw_data[raw_data['DISTRICT_NAME'] == district]
    
    df_sum = df.append(df.sum(numeric_only=True), ignore_index=True)
    sum_row = df_sum.iloc[[-1]]
    n_row = sum_row.drop('DISTRICT_NAME', axis=1)
    p_row = n_row.drop('TALUKA_NAME', axis=1)
    q_row = p_row.astype(int)
    max_row = q_row.loc[q_row.sum(axis=1).idxmax()]
    max_col = max_row.idxmax()
    row_to_analyze = q_row.iloc[0]
    top_5 = row_to_analyze.nlargest(5).index.tolist()
    
    crop1 = request.form['crop1']
    crop2 = request.form['crop2']
    crop3 = request.form['crop3']
    crop4 = request.form['crop4']
    crop5 = request.form['crop5']

    selected_crops = [crop1, crop2, crop3, crop4, crop5]
    lat_df = sum_row[selected_crops]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    plt.figure(figsize=(8, 6))
    sns.set_style('whitegrid')
    palette = 'Paired'
    ax = sns.barplot(data=lat_df, palette=palette)
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Crops', fontsize=14)
    ax.set_ylabel('Yield', fontsize=14)
    ax.set_title('Crop Yield by Crop Type', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('static/chart1.png')


    colors = sns.color_palette('Paired')
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(lat_df.values[0], colors=colors, autopct='%1.1f%%', shadow=False, startangle=90, 
                                    wedgeprops=dict(width=0.6, edgecolor='w'))
    ax.set_title('Pie Chart', fontsize=15)
    ax.legend(wedges, lat_df.columns, title='Crops', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.savefig('static/chart2.png', bbox_inches='tight')

    # selected_crops = [crop1, crop2, crop3, crop4, crop5]
    top_districts = []
    for i in selected_crops:
        crop_data = raw_data[['DISTRICT_NAME'] + [i]]
        crop_data = crop_data.groupby('DISTRICT_NAME').sum().reset_index()
        crop_data['Total'] = crop_data[[i]].sum(axis=1)
        crop_data = crop_data.sort_values('Total', ascending=False).reset_index(drop=True)
        top_3 = crop_data.head(3)['DISTRICT_NAME'].tolist()
        top_districts.append((i, top_3))

        
    crops = []
    for crop in selected_crops:
        if lat_df[crop].iloc[0] == 0:
            crops.append((crop, f'does not grow in {district}.'))
        else:
            crops.append((crop, f'grows in {district}.'))

    return render_template('cindex.html', crops=crops, max_crop=max_col, top_5=top_5, top_districts=top_districts)

farmers_data = pd.read_csv('F_Dataset.csv')
farmers_data['District'] = farmers_data['District'].str.strip()

@app.route('/farmer')
def farmindex():
    return render_template('findex.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    age = request.form['age']
    email = request.form['email']
    district = request.form['district']
    taluka = request.form['taluka']
    address = request.form['address']
    landsize = request.form['landsize']
    phone_no= request.form['phone_no']
    other_info = request.form['other_info']
    # Create a new DataFrame
    data = {'Name': [name],'Phone_no': [phone_no],'Land_size': [landsize],
            'Address': [address],'District': [district],'Taluka': [taluka],'Age':[age],'Email':[email],'Other_info':[other_info]}
   
    df = pd.DataFrame(data)
    
    # Get latitude and longitude from address using Geopy
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    
    # Add latitude and longitude to DataFrame
    df['Latitude'] = latitude
    df['Longitude'] = longitude
    
    # Save the DataFrame to F_dataset.csv without appending age and email columns
    existing_df = pd.read_csv('F_Dataset.csv')
    # existing_df = existing_df.drop(['Age', 'Email'], axis=1)
    updated_df = pd.concat([existing_df, df], ignore_index=True)
    updated_df.to_csv('F_Dataset.csv', index=False)
    
    return render_template('popup.html')

@app.route('/map', methods=['GET', 'POST'])
def mapindex():
    district = None
    map_html = None

    if request.method == 'POST':
        
        district = request.form['district']

        # Filter data for the district
        f_data = farmers_data[farmers_data['District'] == district]

        # Drop rows with missing location data
        f_data = f_data.dropna(subset=['Latitude', 'Longitude'])

        # Calculate center of the district
        district_center = [f_data['Latitude'].mean(), f_data['Longitude'].mean()]

        # Create the map object
        m = folium.Map(location=district_center, zoom_start=10)

        # Add markers for each farmer
        fg = folium.FeatureGroup(name='Farmers')
        for _, row in f_data.iterrows():
            popup_html = f'<table style="width: 300px;"><tr><th>Farmer Name:</th><td>{row["Name"]}</td></tr><tr><th>Phone No:</th><td>{row["Phone_no"]}</td></tr><tr><th>Land size:</th><td>{row["Land_size"]} acre</td></tr></table>'
            fg.add_child(folium.Marker(location=[row["Latitude"], row["Longitude"]], popup=popup_html, icon=folium.Icon(color='darkgreen')))
        
        m.add_child(fg)

        # Convert the map object to HTML
        map_html = m.get_root().render()

    # Render the HTML template with the form and map
    return render_template('mindex.html', district=district, map_html=map_html)

########---------Hindi Routes-------########

@app.route('/hi')
def hindiindex():
    return render_template('index_hi.html')

@app.route('/hisignin')
def hindisignin():
    return render_template('signin_hi.html')

@app.route('/hisignup')
def hindisignup():
    return render_template('signup_hi.html')

@app.route('/hiabout')
def hindiin():
    return render_template('aboutus_hi.html')

@app.route('/hicontact')
def hindicontact():
    return render_template('contact_hi.html')

@app.route('/hipopup')
def hindipopup():
    return render_template('popup_hi.html')


@app.route('/himap', methods=['GET', 'POST'])
def himapindex():
    district = None
    map_html = None

    if request.method == 'POST':
        
        district = request.form['district']

        # Filter data for the district
        f_data = farmers_data[farmers_data['District'] == district]

        # Drop rows with missing location data
        f_data = f_data.dropna(subset=['Latitude', 'Longitude'])

        # Calculate center of the district
        district_center = [f_data['Latitude'].mean(), f_data['Longitude'].mean()]

        # Create the map object
        m = folium.Map(location=district_center, zoom_start=10)

        # Add markers for each farmer
        fg = folium.FeatureGroup(name='Farmers')
        for _, row in f_data.iterrows():
            popup_html = f'<table style="width: 300px;"><tr><th>Farmer Name:</th><td>{row["Name"]}</td></tr><tr><th>Phone No:</th><td>{row["Phone_no"]}</td></tr><tr><th>Land size:</th><td>{row["Land_size"]} acre</td></tr></table>'
            fg.add_child(folium.Marker(location=[row["Latitude"], row["Longitude"]], popup=popup_html, icon=folium.Icon(color='darkgreen')))
        
        m.add_child(fg)

        # Convert the map object to HTML
        map_html = m.get_root().render()

    # Render the HTML template with the form and map
    return render_template('mindex_hi.html', district=district, map_html=map_html)

@app.route('/hifarmer')
def hifarmindex():
    return render_template('findex_hi.html')

@app.route('/hisubmit', methods=['POST'])
def hisubmit():
    name = request.form['name']
    age = request.form['age']
    email = request.form['email']
    district = request.form['district']
    taluka = request.form['taluka']
    address = request.form['address']
    landsize = request.form['landsize']
    phone_no= request.form['phone_no']
    other_info = request.form['other_info']
    # Create a new DataFrame
    data = {'Name': [name],'Phone_no': [phone_no],'Land_size': [landsize],
            'Address': [address],'District': [district],'Taluka': [taluka],'Age':[age],'Email':[email],'Other_info':[other_info]}
   
    df = pd.DataFrame(data)
    
    # Get latitude and longitude from address using Geopy
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    
    # Add latitude and longitude to DataFrame
    df['Latitude'] = latitude
    df['Longitude'] = longitude
    
    # Save the DataFrame to F_dataset.csv without appending age and email columns
    existing_df = pd.read_csv('F_Dataset.csv')
    # existing_df = existing_df.drop(['Age', 'Email'], axis=1)
    updated_df = pd.concat([existing_df, df], ignore_index=True)
    updated_df.to_csv('F_Dataset.csv', index=False)
    
    return render_template('popup_hi.html')

@app.route('/hicrop')
def hicrop():
    return render_template('cindex_hi.html')

@app.route('/hichart', methods=['POST'])
def hichart():
    
    raw_data['DISTRICT_NAME'] = raw_data['DISTRICT_NAME'].str.replace(' ', '')
    district = request.form['district']
    df = raw_data[raw_data['DISTRICT_NAME'] == district]
    
    df_sum = df.append(df.sum(numeric_only=True), ignore_index=True)
    sum_row = df_sum.iloc[[-1]]
    n_row = sum_row.drop('DISTRICT_NAME', axis=1)
    p_row = n_row.drop('TALUKA_NAME', axis=1)
    q_row = p_row.astype(int)
    max_row = q_row.loc[q_row.sum(axis=1).idxmax()]
    max_col = max_row.idxmax()
    row_to_analyze = q_row.iloc[0]
    top_5 = row_to_analyze.nlargest(5).index.tolist()
    
    crop1 = request.form['crop1']
    crop2 = request.form['crop2']
    crop3 = request.form['crop3']
    crop4 = request.form['crop4']
    crop5 = request.form['crop5']

    selected_crops = [crop1, crop2, crop3, crop4, crop5]
    lat_df = sum_row[selected_crops]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    plt.figure(figsize=(8, 6))
    sns.set_style('whitegrid')
    palette = 'Paired'
    ax = sns.barplot(data=lat_df, palette=palette)
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Crops', fontsize=14)
    ax.set_ylabel('Yield', fontsize=14)
    ax.set_title('Crop Yield by Crop Type', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('static/chart1.png')


    colors = sns.color_palette('Paired')
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(lat_df.values[0], colors=colors, autopct='%1.1f%%', shadow=False, startangle=90, 
                                    wedgeprops=dict(width=0.6, edgecolor='w'))
    ax.set_title('Pie Chart', fontsize=15)
    ax.legend(wedges, lat_df.columns, title='Crops', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.savefig('static/chart2.png', bbox_inches='tight')

    # selected_crops = [crop1, crop2, crop3, crop4, crop5]
    top_districts = []
    for i in selected_crops:
        crop_data = raw_data[['DISTRICT_NAME'] + [i]]
        crop_data = crop_data.groupby('DISTRICT_NAME').sum().reset_index()
        crop_data['Total'] = crop_data[[i]].sum(axis=1)
        crop_data = crop_data.sort_values('Total', ascending=False).reset_index(drop=True)
        top_3 = crop_data.head(3)['DISTRICT_NAME'].tolist()
        top_districts.append((i, top_3))

        
    crops = []
    for crop in selected_crops:
        if lat_df[crop].iloc[0] == 0:
            crops.append((crop, f'{district} जिला में उगता नहीं है।'))
        else:
            crops.append((crop, f"{district} जिला में उगता है।"))

    return render_template('cindex_hi.html', crops=crops, max_crop=max_col, top_5=top_5, top_districts=top_districts)

#---------------Marathi Routes-------------

@app.route('/ma')
def marathiindex():
    return render_template('index_ma.html')

@app.route('/masignin')
def marathisignin():
    return render_template('signin_ma.html')

@app.route('/masignup')
def marathisignup():
    return render_template('signup_ma.html')

@app.route('/maabout')
def marathiin():
    return render_template('aboutus_ma.html')

@app.route('/macontact')
def marathicontact():
    return render_template('contact_ma.html')

@app.route('/mapopup')
def marathipopup():
    return render_template('popup_ma.html')

@app.route('/mamap', methods=['GET', 'POST'])
def mamapindex():
    district = None
    map_html = None

    if request.method == 'POST':
        
        district = request.form['district']

        # Filter data for the district
        f_data = farmers_data[farmers_data['District'] == district]

        # Drop rows with missing location data
        f_data = f_data.dropna(subset=['Latitude', 'Longitude'])

        # Calculate center of the district
        district_center = [f_data['Latitude'].mean(), f_data['Longitude'].mean()]

        # Create the map object
        m = folium.Map(location=district_center, zoom_start=10)

        # Add markers for each farmer
        fg = folium.FeatureGroup(name='Farmers')
        for _, row in f_data.iterrows():
            popup_html = f'<table style="width: 300px;"><tr><th>Farmer Name:</th><td>{row["Name"]}</td></tr><tr><th>Phone No:</th><td>{row["Phone_no"]}</td></tr><tr><th>Land size:</th><td>{row["Land_size"]} acre</td></tr></table>'
            fg.add_child(folium.Marker(location=[row["Latitude"], row["Longitude"]], popup=popup_html, icon=folium.Icon(color='darkgreen')))
        
        m.add_child(fg)

        # Convert the map object to HTML
        map_html = m.get_root().render()

    # Render the HTML template with the form and map
    return render_template('mindex_ma.html', district=district, map_html=map_html)

@app.route('/mafarmer')
def mafarmindex():
    return render_template('findex_ma.html')

@app.route('/masubmit', methods=['POST'])
def masubmit():
    name = request.form['name']
    age = request.form['age']
    email = request.form['email']
    district = request.form['district']
    taluka = request.form['taluka']
    address = request.form['address']
    landsize = request.form['landsize']
    phone_no= request.form['phone_no']
    other_info = request.form['other_info']
    # Create a new DataFrame
    data = {'Name': [name],'Phone_no': [phone_no],'Land_size': [landsize],
            'Address': [address],'District': [district],'Taluka': [taluka],'Age':[age],'Email':[email],'Other_info':[other_info]}
   
    df = pd.DataFrame(data)
    
    # Get latitude and longitude from address using Geopy
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    
    # Add latitude and longitude to DataFrame
    df['Latitude'] = latitude
    df['Longitude'] = longitude
    
    # Save the DataFrame to F_dataset.csv without appending age and email columns
    existing_df = pd.read_csv('F_Dataset.csv')
    # existing_df = existing_df.drop(['Age', 'Email'], axis=1)
    updated_df = pd.concat([existing_df, df], ignore_index=True)
    updated_df.to_csv('F_Dataset.csv', index=False)
    
    return render_template('popup_ma.html')

@app.route('/macrop')
def macrop():
    return render_template('cindex_ma.html')

@app.route('/machart', methods=['POST'])
def machart():
    
    raw_data['DISTRICT_NAME'] = raw_data['DISTRICT_NAME'].str.replace(' ', '')
    district = request.form['district']
    df = raw_data[raw_data['DISTRICT_NAME'] == district]
    
    df_sum = df.append(df.sum(numeric_only=True), ignore_index=True)
    sum_row = df_sum.iloc[[-1]]
    n_row = sum_row.drop('DISTRICT_NAME', axis=1)
    p_row = n_row.drop('TALUKA_NAME', axis=1)
    q_row = p_row.astype(int)
    max_row = q_row.loc[q_row.sum(axis=1).idxmax()]
    max_col = max_row.idxmax()
    row_to_analyze = q_row.iloc[0]
    top_5 = row_to_analyze.nlargest(5).index.tolist()
    
    crop1 = request.form['crop1']
    crop2 = request.form['crop2']
    crop3 = request.form['crop3']
    crop4 = request.form['crop4']
    crop5 = request.form['crop5']

    selected_crops = [crop1, crop2, crop3, crop4, crop5]
    lat_df = sum_row[selected_crops]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    plt.figure(figsize=(8, 6))
    sns.set_style('whitegrid')
    palette = 'Paired'
    ax = sns.barplot(data=lat_df, palette=palette)
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Crops', fontsize=14)
    ax.set_ylabel('Yield', fontsize=14)
    ax.set_title('Crop Yield by Crop Type', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('static/chart1.png')


    colors = sns.color_palette('Paired')
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(lat_df.values[0], colors=colors, autopct='%1.1f%%', shadow=False, startangle=90, 
                                    wedgeprops=dict(width=0.6, edgecolor='w'))
    ax.set_title('Pie Chart', fontsize=15)
    ax.legend(wedges, lat_df.columns, title='Crops', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.savefig('static/chart2.png', bbox_inches='tight')

    # selected_crops = [crop1, crop2, crop3, crop4, crop5]
    top_districts = []
    for i in selected_crops:
        crop_data = raw_data[['DISTRICT_NAME'] + [i]]
        crop_data = crop_data.groupby('DISTRICT_NAME').sum().reset_index()
        crop_data['Total'] = crop_data[[i]].sum(axis=1)
        crop_data = crop_data.sort_values('Total', ascending=False).reset_index(drop=True)
        top_3 = crop_data.head(3)['DISTRICT_NAME'].tolist()
        top_districts.append((i, top_3))

        
    crops = []
    for crop in selected_crops:
        if lat_df[crop].iloc[0] == 0:
            crops.append((crop, f'जमिनी,{district} येथे वाढत नाही.'))
        else:
            crops.append((crop, f'{district}मध्ये पिकतात.'))

    return render_template('cindex_ma.html', crops=crops, max_crop=max_col, top_5=top_5, top_districts=top_districts)


if __name__ == '__main__':
    app.run(debug=True)
