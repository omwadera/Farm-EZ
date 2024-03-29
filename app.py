from flask import Flask, render_template, request, redirect, url_for,session
from geopy.geocoders import Nominatim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from flask_cors import CORS, cross_origin
from flask_pymongo import PyMongo
from bson import ObjectId
from werkzeug.utils import secure_filename
import os
import sklearn
import pickle
import jsonify
from pymongo import MongoClient
import plotly.express as px
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
import datetime
from dotenv import load_dotenv
import google.generativeai as genai  # Import the Gemini library
from gtts import gTTS
import asyncio
import string
from folium.plugins import FastMarkerCluster
from datetime import datetime
import crops
import random
sns.set()

app = Flask(__name__)
CORS(app)

cors = CORS(app, resources={r"/ticker": {"origins": "http://localhost:port"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['MONGO_DBNAME'] = 'FarmEz'
app.config['MONGO_URI'] = 'mongodb+srv://nareshvaishnavrko11:nareshrko11@cluster0.hudqzdr.mongodb.net/FarmEz'
client = MongoClient('mongodb+srv://nareshvaishnavrko11:nareshrko11@cluster0.hudqzdr.mongodb.net/')

hugging_face = os.getenv('hugging_face')
gemini_api_key = os.getenv('GOOGLE_API_KEY')  # Use correct key for Gemini
genai.configure(api_key=gemini_api_key)

db = client['FarmEz']
shopping_list_collection = db['cart']
mongo = PyMongo(app)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'webm'}

app.secret_key = 'nareshrko10'
app.config['SESSION_COOKIE_SECURE'] = True  # Ensures session cookie is sent only over HTTPS (secure)
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevents access to the session cookie via JavaScript
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Session cookie sent with same-site requests (Lax or Strict)

# importing model
model = pickle.load(open('crop_recommendation_model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

raw_data = pd.read_csv('FinalDataset2.csv')
raw_data = raw_data.drop(['Latitude', 'Longitude'], axis=1)

@app.route('/')
def index():
    return render_template('index.html')

api_key = '54f5fef66dd24f61a654f4f36667b65f'

@app.route('/news')
def news():
    return render_template('news.html')

# Function to fetch news using the API
def fetch_news(page, q):
    
    current_date = datetime.datetime.now()
    yesterday = current_date - datetime.timedelta(days=1)
    yesterday_date = yesterday.strftime('%Y-%m-%d')
    
    q = 'agriculture'
    url = f'https://newsapi.org/v2/everything?q={q}&from={yesterday_date}&language=en&pageSize=20&page={page}&sortBy=popularity'
    headers = {'x-api-key': api_key}
    response = requests.get(url, headers=headers)
    news_data = response.json()
    articles = news_data.get('articles', [])
    cleaned_articles = [{'title': article['title'], 'description': article['description'], 'urlToImage': article['urlToImage'], 'url': article['url']} for article in articles]
    return cleaned_articles, news_data.get('totalResults', 0)


def get_gemini_response(question):
  model = genai.GenerativeModel('gemini-pro')  # Load Gemini Pro model
  response = model.generate_content(question)  # Generate response
  return response.text


def text_to_audio(text,filrname):
    tts = gTTS(text)
    tts.save(f'static/audio/{filrname}.mp3')


@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    if 'audio' in request.files:
        audio = request.files['audio']
        if audio and allowed_file(audio.filename):
            filename = secure_filename(audio.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio.save(filepath)
            transcription = process_audio(filepath)
            return jsonify({'text': transcription})

    text = request.form.get('text')
    if text:
        response = process_text(text)
        return {'text': response['text'],'voice': url_for('static', filename='audio/' + response['voice'])}

    return jsonify({'text': 'Invalid request'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_audio(filepath):
    # Placeholder function for processing audio (speech-to-text transcription)
    # Replace this with your own implementation using libraries like SpeechRecognition or DeepSpeech
    #return 'hello This is a placeholder transcription for audio'
    API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
    headers = {"Authorization": hugging_face}
    with open(filepath, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    data = response.json()
    return data['text']
    

def process_text(text):
    # Placeholder function for processing user's text input
    # Replace this with your own implementation
    return_text = get_gemini_response(text)
    #asyncio.run(text_to_audio(return_text))
    # generating random strings
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=8))
    text_to_audio(return_text,res)
    return {"text":return_text,"voice": f"{res}.mp3"}



@app.route('/product/<product_id>')
def product_detail(product_id):
    # Fetch the product details from the database using the product_id
    product = mongo.db.trades.find_one({'_id': ObjectId(product_id)})
    return render_template('product_profile.html', product=product)
    
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
    return render_template('sign_up.html')

@app.route('/crop')
def home():
    return render_template('cindex.html')

@app.route('/recommend')
def cropre():
    return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index2.html',result = result)


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
    
    df = df[selected_crops]

    # Melt the DataFrame to long format
    melted_df = pd.melt(df, value_vars=selected_crops, var_name='Crops', value_name='Yield')

    # Create Plotly bar chart
    fig = px.bar(melted_df, x='Crops', y='Yield', color='Crops',
                 labels={'Yield': 'Yield', 'Crops': 'Crop'},
                 title='Crop Yield by Crop Type in {}'.format(district),
                 width=1200, height=700)

    fig.update_layout(
        xaxis=dict(title='Crop', tickfont=dict(size=12)),
        yaxis=dict(title='Yield', tickfont=dict(size=12)),
        showlegend=False,
        margin=dict(t=50, l=10, r=10, b=10),
    )

    # Convert the Plotly figure to JSON
    chart_json =  json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create Plotly pie chart
    figp = px.pie(melted_df, values='Yield', names='Crops',
              title='Pie Chart - Crop Distribution in {}'.format(district),
              labels={'label': 'Crops', 'value': 'Yield'},
              hole=0.6)

    figp.update_layout(
        width=1200,  # Set the desired width
        height=600,  # Set the desired height
        font=dict(size=16)  # Set the desired font size for the text in the chart
    )

    # Convert the Plotly figure to JSON
    pie_json = json.dumps(figp, cls=plotly.utils.PlotlyJSONEncoder)
    
    dfn = raw_data[(raw_data['DISTRICT_NAME'] == district)][['TALUKA_NAME'] + selected_crops]

    mel_df = pd.melt(dfn, id_vars=['TALUKA_NAME'], value_vars=selected_crops,
                        var_name='Crops', value_name='Yield')

    # Create Plotly line chart
    figl = px.line(mel_df, x='TALUKA_NAME', y='Yield', color='Crops',
                  labels={'Yield': 'Yield', 'TALUKA_NAME': 'Taluka'},
                  title='Crop Yield by Taluka in {}'.format(district),
                   )  # Optional: Choose a template

    # Convert the Plotly figure to JSON
    line_json = json.dumps(figl, cls=plotly.utils.PlotlyJSONEncoder)

    
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

    return render_template('cindex.html',line_json=line_json,pie_json=pie_json,chart_json=chart_json, crops=crops, max_crop=max_col, top_5=top_5, top_districts=top_districts)


@app.route('/account', methods=['POST'])
def create_account():
    
    if request.method == 'POST':
        # Get form data
        fullName = request.form['full-name']
        Age = request.form['Age']
        email = request.form['email']
        phone = request.form['phone']
        district = request.form['district']
        taluka = request.form['taluka']
        
        # Check if a file was uploaded
        if 'Photo' in request.files :
            
            Photo = request.files['Photo']           
            # Securely save the uploaded photos to the defined folder
            filename = secure_filename(Photo.filename)
            Photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))    
        else:
            # Set photo and land_photo to None or default image paths
            filename = None  
        
        # Insert data into MongoDB
        mongo.db.users.insert_one({
            'full-name': fullName,
            'Age': Age,
            'email': email,
            'phone': phone,
            'district': district,
            'taluka': taluka,
            'Photo': filename ,
        })
        
        # Redirect to success page...
        return redirect(url_for('index'))
    else:
        return 'Error'



@app.route('/farmer')
def farmindex():
    return render_template('findex.html')


# Handle form submissions
@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
    
        email = request.form['email']
        landsize = request.form['landsize']
        address = request.form['address']
        latitude = request.form['latitude']
        Land_Survey_No = request.form['Land_Survey_No']
        longitude = request.form['longitude']
        otherinfo = request.form['other-info']

        # Check if a file was uploaded
        if 'land_photo' in request.files:
            # photo = request.files['photo']
            land_photo = request.files['land_photo']
            land_photo_filename = secure_filename(land_photo.filename)
            
            # photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            land_photo.save(os.path.join(app.config['UPLOAD_FOLDER'], land_photo_filename))
        else:
            land_photo_filename = None  # Or provide a default image path
        
        # Insert data into MongoDB
        
        mongo.db.users.update_one(
            {'email': email},
            {
                '$set': {
                    'landsize': landsize,
                    'address': address,
                    'latitude': latitude,
                    'longitude': longitude,
                    'Land_Survey_No': Land_Survey_No,
                    'other-info': otherinfo,
                    'land_photo': land_photo_filename
                }
            }
        ) 
        
        return redirect(url_for('popup'))
    else:
        return 'Error'
    

@app.route('/map', methods=['GET', 'POST'])
def display_map():

    if request.method == 'POST':
        district = request.form['district'].strip()

        # Query the MongoDB database for the latitude and longitude of the given district
        # and store the results in a list of dictionaries
        locations = list(mongo.db.users.find({'district': district, 'latitude': {'$exists': True}, 'longitude': {'$exists': True}}, {'_id': 0, 'latitude': 1, 'longitude': 1}))
        
        if not locations:
            return render_template('mindex.html', district=district, error='No records found for this district.')
        
        # Create a Folium map centered on the first location in the list
        map = folium.Map(location=[locations[0]['latitude'], locations[0]['longitude']], zoom_start=10)
        
        tile_layer = folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(map)
        
        # Add markers for all the locations in the list
        for location in locations:
            # Query the MongoDB database for the user information
            user_info = mongo.db.users.find_one({'district': district, 'latitude': location['latitude'], 'longitude': location['longitude']})
            
            # Create the URL for the farmer's profile using the farmer's ID
            profile_url = url_for('farmer_profile', farmer_id = str(user_info['_id']))
            
            # Modify the popup HTML to include the "More Info" link leading to the farmer's profile
            popup_html = f"""
            <div style="width: 300px;">
                <h3 style="margin: 0; padding: 10px; background-color: #00704A; color: #FFF; text-align: center; font-size: 20px;">
                    {user_info['full-name']}
                </h3>
                <div style="padding: 10px;">
                    <p style="margin: 0; margin-bottom: 5px; font-size: 16px;">Phone: {user_info['phone']}</p>
                    <p style="margin: 0; margin-bottom: 5px; font-size: 16px;">Land Size: {user_info['landsize']} acres</p>
                    <p style="margin: 0; margin-bottom: 5px; font-size: 16px;">Land Survey Number: {user_info['Land_Survey_No']}</p>
                    <div style="text-align: center;">
                        <a href='{profile_url}' target='_blank' style="color: #002F6C; text-decoration: none; font-size: 13px; display: inline-block;">More Info</a>
                    </div>
                </div>
            </div>
            """  # Add a marker with the pop-up to the map
            folium.Marker(location=[location['latitude'], location['longitude']], popup=popup_html).add_to(map)
        
        # Convert the map to HTML and pass it to the template
        map_html = map._repr_html_()
        return render_template('mindex.html', district=district, map_html=map_html)

    # If the request method is not 'POST', return the default map page
    return render_template('mindex.html', district='', map_html='', error='')



@app.route('/payment')
def pay():
    return render_template('gateway.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get the email from the form
        email = request.form['email']

        # Check if the email exists in the database
        farmer_info = mongo.db.users.find_one({'email': email})

        if farmer_info:
            # If the email exists, store the farmer_id in the session
            session['farmer_id'] = str(farmer_info['_id'])
            return redirect(url_for('index'))  # Redirect to home page after successful login
        else:
            # If the email is not found, show an error message or redirect to a registration page
            return "Email not found. Please sign up first."

    # If the request method is GET, render the login page (signin.html)
    return render_template('signin.html')

    
@app.route('/farmer/<farmer_id>')
def farmer_profile(farmer_id):
    # Check if the user is logged in by verifying the 'farmer_id' in the session
    logged_in_farmer_id = session.get('farmer_id')

    # Fetch the farmer's details from MongoDB using the given ID
    farmer_info = mongo.db.users.find_one({'_id': ObjectId(farmer_id)})

    if farmer_info:
        # If the user is logged in, allow them to view any farmer profile
        if logged_in_farmer_id:
            return render_template('profile.html', farmer_info=farmer_info)
        else:
            return "Access denied ! Log in first"
    else:
        return "Farmer not found"

        
# @app.route('/me/<farmer_id>')
# def my_profile(farmer_id):
#     # Check if the user is logged in by verifying the 'farmer_id' in the session
#     if 'farmer_id' in session and str(session['farmer_id']) == farmer_id:
#         # Fetch the farmer's details from MongoDB using the given ID
#         new_farmer_info = mongo.db.users.find_one({'_id': ObjectId(farmer_id)})
#          # Query trade IDs associated with the farmer's ID
#         trade = new_farmer_info['trade']['sell']
        
#         # Query trade data from the "trades" collection using the trade IDs
#         trade_listings = mongo.db.trades.find({'_id': {'$in': trade}})
#         if new_farmer_info:
#             return render_template('my_profile.html', new_farmer_info=new_farmer_info, trade_listings=trade_listings)
#         else:
#             return "Farmer not found"
#     else:
#         return "Access denied ! Log in first"


@app.route('/me/<farmer_id>')
def my_profile(farmer_id):
    # Check if the user is logged in by verifying the 'farmer_id' in the session
    if 'farmer_id' in session and str(session['farmer_id']) == farmer_id:
        # Fetch the farmer's details from MongoDB using the given ID
        new_farmer_info = mongo.db.users.find_one({'_id': ObjectId(farmer_id)})
        
        trade_listings = []
        if 'trade' in new_farmer_info and 'sell' in new_farmer_info['trade']:
            trade_ids = new_farmer_info['trade']['sell']
            if trade_ids:
                # Query trade data from the "trades" collection using the trade IDs
                trade_listings = mongo.db.trades.find({'_id': {'$in': trade_ids}})
        
        if new_farmer_info:
            return render_template('my_profile.html', new_farmer_info=new_farmer_info, trade_listings=trade_listings)
        else:
            return "Farmer not found"
    else:
        return "Access denied! Log in first"

    
@app.route('/logout')
def logout():
    # Clear the session data (log out the user)
    session.clear()
    # Redirect the user to the home page
    return redirect(url_for('index'))


@app.route('/sell')
def sell():
    return render_template('sell.html')

@app.route('/buy')
def buy():
    crops = mongo.db.trades.find()
    return render_template('buy.html',crops=crops)

@app.route('/sell_crops', methods=['POST'])
def sell_crops():
    if 'farmer_id' in session:
        if request.method == 'POST':
            # Get form data
            name =request.form['name']
            crop_image = request.files['crop_image']
            price_per_10kg = request.form['price_per_10kg']
            description = request.form['description']

            # Securely save the uploaded crop image to the defined folder
            crop_image_filename = secure_filename(crop_image.filename)
            crop_image.save(os.path.join(app.config['UPLOAD_FOLDER'], crop_image_filename))

            # Insert trade data into the "trades" collection
            trade_data = {
                'seller_id': ObjectId(session['farmer_id']),
                'name':name,
                'crop_image': crop_image_filename,
                'price_per_10kg': price_per_10kg,
                'description': description
            }
            trade_id = mongo.db.trades.insert_one(trade_data).inserted_id

            # Update the user's document in the "users" collection
            mongo.db.users.update_one(
                {'_id': ObjectId(session['farmer_id'])},
                {'$push': {'trade.sell': trade_id}}
            )

            # Redirect to the profile page after submission
            return redirect(url_for('index', farmer_id=session['farmer_id']))

    return "Access denied. Please log in."


@app.route('/buy_crops', methods=['GET', 'POST'])
def buy_crops():
    crops = []

    if request.method == 'POST':
        crop_name = request.form.get('crop_name', '').strip()
        if crop_name:
            # Query the "trades" collection to get listings for the searched crop
            crops_list = list(mongo.db.trades.find({'name': crop_name}))

    return render_template('buy.html', crops_list=crops_list)


@app.route('/add_to_list', methods=['POST'])
def add_to_list():
    product_id = request.form.get('product_id')
    product = db.trades.find_one({'_id': ObjectId(product_id)})

    if product:
        product['price_per_10kg'] = float(product['price_per_10kg'])  # Convert to float

        cart = db.cart
        product_without_id = {key: value for key, value in product.items() if key != '_id'}
        cart.insert_one(product_without_id)

    return redirect(url_for('buy'))

@app.route('/delete/<string:item_id>')
def delete_item(item_id):
    shopping_list_collection.delete_one({'_id': ObjectId(item_id)})
    return redirect('/shopping_list')


@app.route('/clear_all', methods=['POST'])
def clear_all():
    shopping_list_collection.delete_many({})
    return redirect('/shopping_list')

@app.route('/shopping_list')
def shopping_list():
    shopping_list = list(shopping_list_collection.find())
    total_price = sum([product['price_per_10kg'] for product in shopping_list])
    return render_template('shopping_list.html', shopping_list=shopping_list, total_price=total_price)


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
    if request.method == 'POST':
        district = request.form['district'].strip()

        # Query the MongoDB database for the latitude and longitude of the given district
        # and store the results in a list of dictionaries
        locations = list(mongo.db.farmers.find({'district': district}, {'_id': 0, 'latitude': 1, 'longitude': 1}))
        if not locations:
            return render_template('mindex_hi.html', district=district, error='No records found for this district.')
        # Create a Folium map centered on the first location in the list
        map = folium.Map(location=[locations[0]['latitude'], locations[0]['longitude']], zoom_start=10)
        # Add markers for all the locations in the list
        for location in locations:
            # Query the MongoDB database for the user information
            row = mongo.db.farmers.find_one({'district': district, 'latitude': location['latitude'], 'longitude': location['longitude']})
            # Create a string with the user information to be displayed in the pop-up
            popup_html = f'<table style="width: 300px;"><tr><th>Farmer Name:</th><td>{row["full-name"]}</td></tr><tr><th>Phone No:</th><td>{row["phone"]}</td></tr><tr><th>Land size:</th><td>{row["landsize"]} acre</td></tr></table>'
            # Add a marker with the pop-up to the map
            folium.Marker(location=[location['latitude'], location['longitude']], popup=popup_html,icon=folium.Icon(color='darkgreen')).add_to(map)
        # Convert the map to HTML and pass it to the template
        map_html = map._repr_html_()
        return render_template('mindex_hi.html', district=district, map_html=map_html)

    # If the request method is not 'POST', return the default map page
    return render_template('mindex_hi.html', district='', map_html='', error='')

@app.route('/hifarmer')
def hifarmindex():
    return render_template('findex_hi.html')

@app.route('/hisubmit', methods=['POST'])
def hisubmit():
    if request.method == 'POST':
        fullName = request.form['full-name']
        Age = request.form['Age']
        email = request.form['email']
        phone = request.form['phone']
        district = request.form['district']
        taluka = request.form['taluka']
        landsize = request.form['landsize']
        address = request.form['address']
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        otherinfo = request.form['other-info']
        mongo.db.farmers.insert_one({
            'full-name': fullName,
            'Age': Age,
            'email': email,
            'phone': phone,
            'district': district,
            'taluka': taluka,
            'landsize': landsize,
            'address': address,
            'latitude': latitude,
            'longitude': longitude,
            'other-info': otherinfo
        })
        # Redirect to success page...
        return redirect(url_for('popup'))
    else:
        return 'Error'
    
    
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
            crops.append((crop, f'does not grow in {district}.'))
        else:
            crops.append((crop, f'grows in {district}.'))

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
    if request.method == 'POST':
        district = request.form['district'].strip()

        # Query the MongoDB database for the latitude and longitude of the given district
        # and store the results in a list of dictionaries
        locations = list(mongo.db.farmers.find({'district': district}, {'_id': 0, 'latitude': 1, 'longitude': 1}))
        if not locations:
            return render_template('mindex_ma.html', district=district, error='No records found for this district.')
        # Create a Folium map centered on the first location in the list
        map = folium.Map(location=[locations[0]['latitude'], locations[0]['longitude']], zoom_start=10)
        # Add markers for all the locations in the list
        for location in locations:
            # Query the MongoDB database for the user information
            row = mongo.db.farmers.find_one({'district': district, 'latitude': location['latitude'], 'longitude': location['longitude']})
            # Create a string with the user information to be displayed in the pop-up
            popup_html = f'<table style="width: 300px;"><tr><th>Farmer Name:</th><td>{row["full-name"]}</td></tr><tr><th>Phone No:</th><td>{row["phone"]}</td></tr><tr><th>Land size:</th><td>{row["landsize"]} acre</td></tr></table>'
            # Add a marker with the pop-up to the map
            folium.Marker(location=[location['latitude'], location['longitude']], popup=popup_html).add_to(map)
        # Convert the map to HTML and pass it to the template
        map_html = map._repr_html_()
        return render_template('mindex_ma.html', district=district, map_html=map_html)

    # If the request method is not 'POST', return the default map page
    return render_template('mindex_ma.html', district='', map_html='', error='')

@app.route('/mafarmer')
def mafarmindex():
    return render_template('findex_ma.html')

@app.route('/masubmit', methods=['POST'])
def masubmit():
    if request.method == 'POST':
        fullName = request.form['full-name']
        Age = request.form['Age']
        email = request.form['email']
        phone = request.form['phone']
        district = request.form['district']
        taluka = request.form['taluka']
        landsize = request.form['landsize']
        address = request.form['address']
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        otherinfo = request.form['other-info']
        mongo.db.farmers.insert_one({
            'full-name': fullName,
            'Age': Age,
            'email': email,
            'phone': phone,
            'district': district,
            'taluka': taluka,
            'landsize': landsize,
            'address': address,
            'latitude': latitude,
            'longitude': longitude,
            'other-info': otherinfo
        })
        # Redirect to success page...
        return redirect(url_for('popup'))
    else:
        return 'Error'

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
            crops.append((crop, f'does not grow in {district}.'))
        else:
            crops.append((crop, f'grows in {district}.'))

    return render_template('cindex_ma.html', crops=crops, max_crop=max_col, top_5=top_5, top_districts=top_districts)


commodity_dict = {
    "arhar": "static/Arhar.csv",
    "bajra": "static/Bajra.csv",
    "barley": "static/Barley.csv",
    "copra": "static/Copra.csv",
    "cotton": "static/Cotton.csv",
    "sesamum": "static/Sesamum.csv",
    "gram": "static/Gram.csv",
    "groundnut": "static/Groundnut.csv",
    "jowar": "static/Jowar.csv",
    "maize": "static/Maize.csv",
    "masoor": "static/Masoor.csv",
    "moong": "static/Moong.csv",
    "niger": "static/Niger.csv",
    "paddy": "static/Paddy.csv",
    "ragi": "static/Ragi.csv",
    "rape": "static/Rape.csv",
    "jute": "static/Jute.csv",
    "safflower": "static/Safflower.csv",
    "soyabean": "static/Soyabean.csv",
    "sugarcane": "static/Sugarcane.csv",
    "sunflower": "static/Sunflower.csv",
    "urad": "static/Urad.csv",
    "wheat": "static/Wheat.csv"
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]
base = {
    "Paddy": 1245.5,
    "Arhar": 3200,
    "Bajra": 1175,
    "Barley": 980,
    "Copra": 5100,
    "Cotton": 3600,
    "Sesamum": 4200,
    "Gram": 2800,
    "Groundnut": 3700,
    "Jowar": 1520,
    "Maize": 1175,
    "Masoor": 2800,
    "Moong": 3500,
    "Niger": 3500,
    "Ragi": 1500,
    "Rape": 2500,
    "Jute": 1675,
    "Safflower": 2500,
    "Soyabean": 2200,
    "Sugarcane": 2250,
    "Sunflower": 3700,
    "Urad": 4300,
    "Wheat": 1350

}
commodity_list = []


class Commodity:

    def __init__(self, csv_name):
        self.name = csv_name
        dataset = pd.read_csv(csv_name)
        self.X = dataset.iloc[:, :-1].values
        self.Y = dataset.iloc[:, 3].values

        #from sklearn.model_selection import train_test_split
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

        # Fitting decision tree regression to dataset
        from sklearn.tree import DecisionTreeRegressor
        depth = random.randrange(7,18)
        self.regressor = DecisionTreeRegressor(max_depth=depth)
        self.regressor.fit(self.X, self.Y)
        #y_pred_tree = self.regressor.predict(X_test)
        # fsa=np.array([float(1),2019,45]).reshape(1,3)
        # fask=regressor_tree.predict(fsa)

    def getPredictedValue(self, value):
        if value[1]>=2019:
            fsa = np.array(value).reshape(1, 3)
            #print(" ",self.regressor.predict(fsa)[0])
            return self.regressor.predict(fsa)[0]
        else:
            c=self.X[:,0:2]
            x=[]
            for i in c:
                x.append(i.tolist())
            fsa = [value[0], value[1]]
            ind = 0
            for i in range(0,len(x)):
                if x[i]==fsa:
                    ind=i
                    break
            #print(index, " ",ind)
            #print(x[ind])
            #print(self.Y[i])
            return self.Y[i]

    def getCropName(self):
        a = self.name.split('.')
        return a[0]


@app.route('/trend')
def trends():
    context = {
        "top5": TopFiveWinners(),
        "bottom5": TopFiveLosers(),
        "sixmonths": SixMonthsForecast()
    }
    return render_template('trends.html', context=context)


@app.route('/commodity/<name>')
def crop_profile(name):
    max_crop, min_crop, forecast_crop_values = TwelveMonthsForecast(name)
    prev_crop_values = TwelveMonthPrevious(name)
    forecast_x = [i[0] for i in forecast_crop_values]
    forecast_y = [i[1] for i in forecast_crop_values]
    previous_x = [i[0] for i in prev_crop_values]
    previous_y = [i[1] for i in prev_crop_values]
    current_price = CurrentMonth(name)
    #print(max_crop)
    #print(min_crop)
    #print(forecast_crop_values)
    #print(prev_crop_values)
    #print(str(forecast_x))
    crop_data = crops.crop(name)
    context = {
        "name":name,
        "max_crop": max_crop,
        "min_crop": min_crop,
        "forecast_values": forecast_crop_values,
        "forecast_x": str(forecast_x),
        "forecast_y":forecast_y,
        "previous_values": prev_crop_values,
        "previous_x":previous_x,
        "previous_y":previous_y,
        "current_price": current_price,
        "image_url":crop_data[0],
        "prime_loc":crop_data[1],
        "type_c":crop_data[2],
        "export":crop_data[3]
    }
    return render_template('commodity.html', context=context)

@app.route('/ticker/<item>/<number>')
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def ticker(item, number):
    n = int(number)
    i = int(item)
    data = SixMonthsForecast()
    context = str(data[n][i])

    if i == 2 or i == 5:
        context = '₹' + context
    elif i == 3 or i == 6:

        context = context + '%'

    #print('context: ', context)
    return context


def TopFiveWinners():
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_month_prediction.append(current_predict)
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))
    sorted_change = change
    sorted_change.sort(reverse=True)
    # print(sorted_change)
    to_send = []
    for j in range(0, 5):
        perc, i = sorted_change[j]
        name = commodity_list[i].getCropName().split('/')[1]
        to_send.append([name, round((current_month_prediction[i] * base[name]) / 100, 2), round(perc, 2)])
    #print(to_send)
    return to_send


def TopFiveLosers():
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_month_prediction.append(current_predict)
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))
    sorted_change = change
    sorted_change.sort()
    to_send = []
    for j in range(0, 5):
        perc, i = sorted_change[j]
        name = commodity_list[i].getCropName().split('/')[1]
        to_send.append([name, round((current_month_prediction[i] * base[name]) / 100, 2), round(perc, 2)])
   # print(to_send)
    return to_send



def SixMonthsForecast():
    month1=[]
    month2=[]
    month3=[]
    month4=[]
    month5=[]
    month6=[]
    for i in commodity_list:
        crop=SixMonthsForecastHelper(i.getCropName())
        k=0
        for j in crop:
            time = j[0]
            price = j[1]
            change = j[2]
            if k==0:
                month1.append((price,change,i.getCropName().split("/")[1],time))
            elif k==1:
                month2.append((price,change,i.getCropName().split("/")[1],time))
            elif k==2:
                month3.append((price,change,i.getCropName().split("/")[1],time))
            elif k==3:
                month4.append((price,change,i.getCropName().split("/")[1],time))
            elif k==4:
                month5.append((price,change,i.getCropName().split("/")[1],time))
            elif k==5:
                month6.append((price,change,i.getCropName().split("/")[1],time))
            k+=1
    month1.sort()
    month2.sort()
    month3.sort()
    month4.sort()
    month5.sort()
    month6.sort()
    crop_month_wise=[]
    crop_month_wise.append([month1[0][3],month1[len(month1)-1][2],month1[len(month1)-1][0],month1[len(month1)-1][1],month1[0][2],month1[0][0],month1[0][1]])
    crop_month_wise.append([month2[0][3],month2[len(month2)-1][2],month2[len(month2)-1][0],month2[len(month2)-1][1],month2[0][2],month2[0][0],month2[0][1]])
    crop_month_wise.append([month3[0][3],month3[len(month3)-1][2],month3[len(month3)-1][0],month3[len(month3)-1][1],month3[0][2],month3[0][0],month3[0][1]])
    crop_month_wise.append([month4[0][3],month4[len(month4)-1][2],month4[len(month4)-1][0],month4[len(month4)-1][1],month4[0][2],month4[0][0],month4[0][1]])
    crop_month_wise.append([month5[0][3],month5[len(month5)-1][2],month5[len(month5)-1][0],month5[len(month5)-1][1],month5[0][2],month5[0][0],month5[0][1]])
    crop_month_wise.append([month6[0][3],month6[len(month6)-1][2],month6[len(month6)-1][0],month6[len(month6)-1][1],month6[0][2],month6[0][0],month6[0][1]])

   # print(crop_month_wise)
    return crop_month_wise

def SixMonthsForecastHelper(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name = name.split("/")[1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 7):
        if current_month + i <= 12:
            month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
        else:
            month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[current_month + i - 13]))
    wpis = []
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    change = []

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), y, r])
        wpis.append(current_predict)
        change.append(((current_predict - current_wpi) * 100) / current_wpi)

    crop_price = []
    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2) , round(change[i], 2)])

   # print("Crop_Price: ", crop_price)
    return crop_price

def CurrentMonth(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    current_price = (base[name.capitalize()]*current_wpi)/100
    return current_price

def TwelveMonthsForecast(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if current_month + i <= 12:
            month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
        else:
            month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[current_month + i - 13]))
    max_index = 0
    min_index = 0
    max_value = 0
    min_value = 9999
    wpis = []
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    change = []

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), y, r])
        if current_predict > max_value:
            max_value = current_predict
            max_index = month_with_year.index((m, y, r))
        if current_predict < min_value:
            min_value = current_predict
            min_index = month_with_year.index((m, y, r))
        wpis.append(current_predict)
        change.append(((current_predict - current_wpi) * 100) / current_wpi)

    max_month, max_year, r1 = month_with_year[max_index]
    min_month, min_year, r2 = month_with_year[min_index]
    min_value = min_value * base[name.capitalize()] / 100
    max_value = max_value * base[name.capitalize()] / 100
    crop_price = []
    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2) , round(change[i], 2)])
   # print("forecasr", wpis)
    x = datetime(max_year,max_month,1)
    x = x.strftime("%b %y")
    max_crop = [x, round(max_value,2)]
    x = datetime(min_year, min_month, 1)
    x = x.strftime("%b %y")
    min_crop = [x, round(min_value,2)]

    return max_crop, min_crop, crop_price


def TwelveMonthPrevious(name):
    name = name.lower()
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    commodity = commodity_list[0]
    wpis = []
    crop_price = []
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if current_month - i >= 1:
            month_with_year.append((current_month - i, current_year, annual_rainfall[current_month - i - 1]))
        else:
            month_with_year.append((current_month - i + 12, current_year - 1, annual_rainfall[current_month - i + 11]))

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), 2013, r])
        wpis.append(current_predict)

    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y,m,1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2)])
   # print("previous ", wpis)
    new_crop_price =[]
    for i in range(len(crop_price)-1,-1,-1):
        new_crop_price.append(crop_price[i])
    return new_crop_price


if __name__ == '__main__': 
    arhar = Commodity(commodity_dict["arhar"])
    commodity_list.append(arhar)
    bajra = Commodity(commodity_dict["bajra"])
    commodity_list.append(bajra)
    barley = Commodity(commodity_dict["barley"])
    commodity_list.append(barley)
    copra = Commodity(commodity_dict["copra"])
    commodity_list.append(copra)
    cotton = Commodity(commodity_dict["cotton"])
    commodity_list.append(cotton)
    sesamum = Commodity(commodity_dict["sesamum"])
    commodity_list.append(sesamum)
    gram = Commodity(commodity_dict["gram"])
    commodity_list.append(gram)
    groundnut = Commodity(commodity_dict["groundnut"])
    commodity_list.append(groundnut)
    jowar = Commodity(commodity_dict["jowar"])
    commodity_list.append(jowar)
    maize = Commodity(commodity_dict["maize"])
    commodity_list.append(maize)
    masoor = Commodity(commodity_dict["masoor"])
    commodity_list.append(masoor)
    moong = Commodity(commodity_dict["moong"])
    commodity_list.append(moong)
    niger = Commodity(commodity_dict["niger"])
    commodity_list.append(niger)
    paddy = Commodity(commodity_dict["paddy"])
    commodity_list.append(paddy)
    ragi = Commodity(commodity_dict["ragi"])
    commodity_list.append(ragi)
    rape = Commodity(commodity_dict["rape"])
    commodity_list.append(rape)
    jute = Commodity(commodity_dict["jute"])
    commodity_list.append(jute)
    safflower = Commodity(commodity_dict["safflower"])
    commodity_list.append(safflower)
    soyabean = Commodity(commodity_dict["soyabean"])
    commodity_list.append(soyabean)
    sugarcane = Commodity(commodity_dict["sugarcane"])
    commodity_list.append(sugarcane)
    sunflower = Commodity(commodity_dict["sunflower"])
    commodity_list.append(sunflower)
    urad = Commodity(commodity_dict["urad"])
    commodity_list.append(urad)
    wheat = Commodity(commodity_dict["wheat"])
    commodity_list.append(wheat)
    
    app.run(port=5000, debug=True)