# Django-related imports
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from .logger import logging


# Data manipulation and analysis
import pandas as pd
import numpy as np

# Image processing
import base64
from io import BytesIO
from PIL import Image

# Web scraping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# Natural Language Processing
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter

# Data visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# CSV handling
import csv
from io import StringIO

# Selenium for web automation
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchWindowException ,StaleElementReferenceException, ElementClickInterceptedException, NoSuchElementException,ElementNotInteractableException
from selenium.webdriver.common.keys import Keys

# Miscellaneous
import string
import pickle
import emoji
import time


from .models import keyword_count_data,youtube_comments
from django.http import HttpResponse
import base64

import pandas as pd
from io import BytesIO
from datetime import datetime

   
from django.shortcuts import render
from .models import keyword_count_data, youtube_comments,sentiments_comments


from googleapiclient.discovery import build
import pandas as pd
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
import pandas as pd
from googleapiclient.errors import HttpError

def currentdtt (request):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #return render_template('index.html', current_datetime=current_datetime)
    return render(request, "overview.html",{current_datetime:current_datetime})


def website_keyword(request):
    return render(request,"website_keyword.html",{})
 
extracted_links = []
    
    
def generate_wordcloud_image(keyword_df):
    # Create a dictionary from the DataFrame for WordCloud input
    word_dict = dict(zip(keyword_df['Keyword'], keyword_df['Count']))
    # Check if the word_dict is empty
    if not word_dict:
        # Handle the case where there are no words
        print("No words to plot in the word cloud.")
        return None
    
    meta_mask = np.array(Image.open('assets/images/globe.png'))
    # meta_mask = np.array(Image.open('E:/enfection/internal_product/M_suite/M_Suite_p/assets/images/meta.png'))

    # Generate the WordCloudv 
    #wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_dict)
    wordcloud = WordCloud(background_color = 'white',margin=10 , mask=meta_mask, contour_width = 2, colormap = 'BuPu_r',contour_color = 'white').generate_from_frequencies(word_dict)

    # Save the WordCloud image to a BytesIO object
    image_stream = BytesIO()
    wordcloud.to_image().save(image_stream, format='PNG')
    image_stream.seek(0)

    # Encode the image in base64
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
    return image_base64    
    
def extract_links(request):
    output_url = "" 
    
    if request.method == "POST":
        output_url = request.POST.get('title[0]')
        additional_urls = [request.POST.get(f'title[{i}]') for i in range(1, 10)]
        all_inputs = [output_url] + additional_urls

    def extractlinks(url):
        try:
            # Send an HTTP GET request to the URL
            response = requests.get(url)
            response.raise_for_status()
            


            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all anchor tags (<a>) that contain 'href' attribute
            links = soup.find_all('a', href=True)

            # Extract and normalize the URLs related to the input URL domain
            extracted_links = set()
            base_url = urlparse(url).scheme + '://' + urlparse(url).netloc  # Get base URL
            for link in links:
                href = link.get('href')
                normalized_url = urljoin(base_url, href)  # Normalize the URL
                # Check if the normalized URL belongs to the same domain as the input URL
                if urlparse(normalized_url).netloc == urlparse(url).netloc:
                    extracted_links.add(normalized_url)

            return list(extracted_links)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return []
    
    all_urls = []
    for i in all_inputs:
        all_urls.extend(extractlinks(i))
    print(all_urls)

    
    
    
    def extract_keywords(url):
        try:
            # Send an HTTP GET request to the URL
            response = requests.get(url)
            
            response.raise_for_status()

            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text content from the page
            text = soup.get_text()

            # Tokenize the text by splitting it into words
            words = re.findall(r'\w+', text.lower())

            # Remove stopwords, one-letter, one-digit words, prepositions, and all numbers
            filtered_words = [word for word in words if word not in stopwords.words("english") and len(word) > 1 and not word.isdigit() and word not in 
                            ["a", "an", "the", "in", "on", "at", "to", "us", "day", "back", "contact", "cookies","cookie","help","menu"]]

            # Create a Counter to count word frequencies
            word_counter = Counter(filtered_words)

            return word_counter

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return Counter()
    keyword_data = []
    for url in all_urls:
        print(f"Loading URL: {url}")
        word_counter = extract_keywords(url)

        # Sort keywords by count in descending order
        sorted_keywords = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

        # Get the top 20 keywords with counts
        top_keywords = sorted_keywords[:30]

        # Append data to the keyword_data list
        for keyword, count in top_keywords:
            keyword_data.append([url, keyword, count])

    # Create a DataFrame for keywords
    keyword_df = pd.DataFrame(keyword_data, columns=["URL", "Keyword", "Count"])
    
    keyword_df = keyword_df.groupby('Keyword').agg({'Count': 'sum','URL': lambda x: x.mode().iloc[0] if not x.mode().empty else None
})
    keyword_df['Count'] = keyword_df['Count'].astype('int')
    # Reset the index and sort by 'Count'
    keyword_df = keyword_df.reset_index().sort_values(by='Count',ascending =False)
    request.session['keyword_df'] = keyword_df.to_dict(orient='records')

    
    
    print(keyword_df)
    
    keyword_list_bar = keyword_df.iloc[0:9,:].to_dict(orient='records')
    #keyword_list_bar = keyword_df.to_dict(orient='records')

    
  
    
    

    # Generate the WordCloud image and get the base64 encoding
    wordcloud_image = generate_wordcloud_image(keyword_df)

    

    # Store keyword_df in the session
    for index, row in keyword_df.iterrows():
        keyword_data_instance, created = keyword_count_data.objects.get_or_create(Keyword=row['Keyword'], defaults={'Count': row['Count'], 'Url': row['URL']})
        keyword_data_instance.Count = row['Count']
        keyword_data_instance.Url = row['URL']
        keyword_data_instance.save()
    # Optionally, you might want to save the instance to the database
    # keyword_data_instance.save()

    #KeywordCountData.objects.bulk_create([KeywordCountData(**data) for data in keyword_df])
    #keyword_data_instance = KeywordCountData.objects.create(Keyword=keyword_df['Keyword'], Count=keyword_df['Count'])

    return render(request,"website_keyword.html",{'given_url': output_url, 'keyword_list': keyword_list_bar, 'keyword_list_bar':keyword_list_bar,
                                        'wordcloud_image': wordcloud_image, 'keyword_df': keyword_df}
    )    
    #return render(request,"index.html",{'given_url':output_url,'keyword_list': keyword_list})
    #return None


def download_dataset(request):
    # Retrieve keyword_df from the session
    keyword_df = request.session.get('keyword_df', [])

    # Check if keyword_df is empty
    if not keyword_df:
        messages.error(request, 'No data to download. Please perform the extraction first.')
        return redirect('website_keyword')  # Redirect to the index view

    # Convert the data back to a DataFrame
    keyword_df = pd.DataFrame(keyword_df)

    # Create an in-memory CSV file
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)

    # Write the header
    writer.writerow(['Keyword', 'Count'])

    # Write the data
    for index, row in keyword_df.iterrows():
        writer.writerow([row['Keyword'], row['Count']])

    # Create an HttpResponse and set the headers
    response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="dataset.csv"'

    return response



def overview(request):
    current_datetime = datetime.now()

    # Extracting month, date, and day
    month = current_datetime.strftime('%B')  # Full month name
    date = current_datetime.day
    day = current_datetime.strftime('%A')    # Full day name

    return render(request, "overview.html", {'month': month, 'date': date, 'day': day})
    


def sentiment_analysis(request):
    return render(request,"sentiment_analysis.html",{})

def brand_authority(request):
    return render(request,"brand_authority.html",{})


def brand_personality(request):
    return render(request,"brand_personality.html",{})


def profile_analyzer(request):
    return render(request,"profile_analyzer.html",{})


def website_keyword(request):
    return render(request,"website_keyword.html",{})


def sentiments(request):
    return render(request,"sentiments.html",{})



def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def convert_emojis_to_words(text): 
    # Convert emojis to words using emoji.demojize
    text_with_emojis_as_words = emoji.demojize(text, delimiters=(' ', ' '))

    return text_with_emojis_as_words


with open('./savedModels/model/model.pickle', 'rb') as f:
    model = pickle.load(f)
with open('./savedModels/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()
vocab = pd.read_csv('./savedModels/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()



from nltk.stem import PorterStemmer
ps = PorterStemmer()

def preprocessing(sentences):
    preprocessed_sentences = []

    for text in sentences:
        data = pd.DataFrame([text], columns=['Full_text'])
        data["Full_text"] = data["Full_text"].apply(lambda x: " ".join(x.lower() for x in x.split()))
        data["Full_text"] = data['Full_text'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
        data['Full_text'] = data['Full_text'].apply(convert_emojis_to_words)
        data["Full_text"] = data["Full_text"].apply(remove_punctuations)
        data["Full_text"] = data['Full_text'].str.replace('\d+', '', regex=True)
        data["Full_text"] = data["Full_text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
        data["Full_text"] = data["Full_text"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
        preprocessed_sentences.append(data["Full_text"].iloc[0])

    return preprocessed_sentences
def vectorizer(ds, vocabulary):
    vectorized_lst = []
    
    for sentence in ds:
        sentence_lst = np.zeros(len(vocabulary))
        
        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split():
                sentence_lst[i] = 1
                
        vectorized_lst.append(sentence_lst)
        
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    
    return vectorized_lst_new

negative_threshold = 0.2
positive_threshold = 0.4

# Categorize the results
def categorize(probability):
    if probability < negative_threshold:
        return 'negative'
    elif negative_threshold <= probability < positive_threshold:
        return 'neutral'
    else:
        return 'positive'
    
def get_prediction(vectorized_text):
    vectorized_text = vectorized_text.reshape(1, -1)
    prediction_score = model.predict_proba(vectorized_text)
    return prediction_score


def sentiment_model(df_comments):
    # Retrieve df_comments from the session
    rows = []
    rows_2 = []
    
    sentences = df_comments['Comments']
    
    for sentence in sentences:
        if sentence.strip():  # Check if the sentence is not empty or contains only whitespace
            preprocessed_sentence = preprocessing([sentence])
            #logging.info(f'Preprocessed Text : {preprocessed_sentence}')

            vectorized_sentence = vectorizer(preprocessed_sentence, tokens)
            #logging.info(f'Vectorized Text : {vectorized_sentence}')

            prediction_scores = get_prediction(vectorized_sentence)

            prediction = categorize(prediction_scores[0, 1])
            # Get the top 5 maximum numbers

            
            #logging.info(f'Prediction : {prediction}')

            rows.append({'Sentence': sentence, 'Sentiment': prediction})
            rows_2.append({'Sentence': sentence, 'prediction_scores': prediction_scores})


    output_df = pd.DataFrame(rows)
    output_df_2 = pd.DataFrame(rows_2)
    
    print(output_df_2)
    output_df_2['score'] = output_df_2['prediction_scores'].apply(lambda x: x[0][1])
    output_df_2 = output_df_2.drop('prediction_scores', axis=1)


    print(output_df_2)
    df_sorted_p = output_df_2.sort_values(by='score', ascending=False)
    filtered_df = df_sorted_p[df_sorted_p['score'] > positive_threshold]
    
    top_positive_comments=[]
    top_negative_comments=[]
    
    if not filtered_df.empty:
        top_positive_comments = filtered_df['Sentence'].head(5)
        print(top_positive_comments)
    else:
        print("No positive comments with score greater than 0.7 found.")
    
    df_sorted_n= output_df_2.sort_values(by='score', ascending=True)
    
    filtered_df = df_sorted_n[df_sorted_n['score'] < negative_threshold]
    
    if not filtered_df.empty:
        top_negative_comments = filtered_df['Sentence'].head(5)
        print(top_negative_comments)
    else:
        print("No negative comments with score less than 0.4 found.")
    
    
    return output_df,top_positive_comments,top_negative_comments



def get_channel_info(youtube, channel_id):
    try:
        request = youtube.channels().list(
            part='snippet,contentDetails,statistics',
            id=channel_id
        )
        response = request.execute()
        channel_info = response['items'][0]

        channel_name = channel_info['snippet']['title']
        videos_count = int(channel_info['statistics']['videoCount'])
        subscribers_count = int(channel_info['statistics']['subscriberCount'])
        joined_date = pd.to_datetime(channel_info['snippet']['publishedAt']).strftime('%B %Y')
        country = channel_info['snippet'].get('country', 'N/A')
        total_views = int(channel_info['statistics']['viewCount'])
        profile_pic_url = channel_info['snippet']['thumbnails']['default']['url']

        avg_views = int(total_views / videos_count) if videos_count != 0 else 0

        return {
            'Channel_Name': channel_name,
            'Profile_Pic_URL': profile_pic_url,
            'Subscribers': subscribers_count,
            'Total_Views': total_views,
            'Country': country,
            'Joined_Date': joined_date,
            'Total_No_Videos': videos_count,
            'Avg_Views': avg_views
        }
    except HttpError as e:
        print(f"Error fetching channel info, Error message: {str(e)}")
        return None

def get_uploaded_playlist_id(youtube, channel_id):
    try:
        request = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        )
        response = request.execute()
        playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        return playlist_id
    except HttpError as e:
        print(f"Error fetching playlist ID, Error message: {str(e)}")
        return None

def get_all_video_comments(youtube, video_ids):
    all_comments = []

    try:
        for video_id in video_ids:
            page_token = None

            while True:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat='plainText',
                    pageToken=page_token
                )
                response = request.execute()

                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    all_comments.append({
                        'Video_ID': video_id,
                        'Comment': comment
                    })

                    if 'replies' in item:
                        for reply_item in item['replies']['comments']:
                            reply = reply_item['snippet']['textDisplay']
                            all_comments.append({
                                'Video_ID': video_id,
                                'Comment': reply
                            })

                page_token = response.get('nextPageToken')
                if not page_token:
                    break

    except HttpError as e:
        print(f"Error fetching comments, Error message: {str(e)}")

    return all_comments

def get_uploaded_videos(youtube, playlist_id):
    all_video_ids = []
    page_token = None

    try:
        while True:
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=page_token
            )
            response = request.execute()

            for item in response['items']:
                video_id = item['contentDetails']['videoId']
                all_video_ids.append(video_id)

            page_token = response.get('nextPageToken')
            if not page_token:
                break

    except HttpError as e:
        print(f"Error fetching uploaded videos, Error message: {str(e)}")

    # Get comments for all video IDs
    all_comments = get_all_video_comments(youtube, all_video_ids)

    # Combine video IDs and comments
    uploaded_videos = [{'Video_ID': video_id, 'Video_Comments': []} for video_id in all_video_ids]
    for comment in all_comments:
        for video in uploaded_videos:
            if comment['Video_ID'] == video['Video_ID']:
                video['Video_Comments'].append(comment['Comment'])

    return uploaded_videos




api_key = 'AIzaSyDVx3HjgrMGMdlIuai5W8aTmBH9JnU4zrE' 
youtube = build('youtube', 'v3', developerKey=api_key)

# Function to get video titles
def get_video_titles(video_ids, youtube):
    video_titles = {}
    for video_id in video_ids:
        request = youtube.videos().list(
            part='snippet',
            id=video_id
        )
        response = request.execute()
        title = response['items'][0]['snippet']['title']
        video_titles[video_id] = title
    return video_titles



def proceed_yt_url(request):
   
    if request.method == "POST":
        yt_url = request.POST.get('yt_url')
        
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # Specify the YouTube channel ID
    channel_id = yt_url
    video_ids = [] 
    comments_df = None 
    try:
        # Retrieve channel information
        channel_info = get_channel_info(youtube, channel_id)

        if channel_info:
            # Create channel_details dataframe
            channel_details = pd.DataFrame([channel_info])

            # Display channel_details dataframe
            print("\nChannel Details DataFrame:")
            print(channel_details)

            # Retrieve the playlist ID for the channel's uploaded videos
            playlist_id = get_uploaded_playlist_id(youtube, channel_id)

            if playlist_id:
                # Retrieve the list of uploaded videos with comments
                uploaded_videos = get_uploaded_videos(youtube, playlist_id)

                # Create comments_df dataframe
                comments_data = []
                for video in uploaded_videos:
                    comments_data.extend({'Video_ID': video['Video_ID'], 'Comment': comment} for comment in video['Video_Comments'])
                comments_df = pd.DataFrame(comments_data)

                # Display comments_df dataframe
                print("\nComments DataFrame:")
                print(comments_df)
                
            video_ids = comments_df['Video_ID'].unique()

    except Exception as e:
        print(f"Error: {str(e)}")


    

    

    # Check if comments_df is not None before accessing attributes
    if comments_df is not None:
        # Get video titles for the provided Video IDs
        video_titles = get_video_titles(video_ids,youtube)

        # Map Video ID to Video Title in the DataFrame
        comments_df['Video_Title'] = comments_df['Video_ID'].map(video_titles)

        del comments_df['Video_ID']
        comments_df = comments_df[['Video_Title'] + [col for col in comments_df.columns if col != 'Video_Title']]

        # Display the modified DataFrame
        print(comments_df)

        df_comments = comments_df
    else:
        # Handle the case where comments_df is None
        df_comments = pd.DataFrame()

    #df_comments = pd.DataFrame(data, columns=['Channel_Name', 'Comments'])
   
    request.session['df_comments'] = df_comments.to_dict(orient='records')
        # Create a DataFrame from the collected data
    
    df_yt_comments = df_comments.iloc[0:5,:].to_dict(orient='records')

    # Store keyword_df in the session
    for index, row in df_comments.iterrows():
        youtube_comments_instance, created = youtube_comments.objects.get_or_create(Video_Title=row['Video_Title'], Comment=row['Comment'])

        youtube_comments_instance.save()


    # # Create instances for each comment and save them
    # for index, row in df_comments.iterrows():
    #     youtube_comments_instance = youtube_comments(Channel_Name=row['Channel_Name'], Comments=row['Comments'])
    #     youtube_comments_instance.save()
   
    df_yt_comments = df_comments.iloc[0:5,:].to_dict(orient='records')
    # Print the DataFrame
    print(df_comments)
    
    df_comments.rename(columns={'Comment': 'Comments'}, inplace=True)
    df_new = df_comments.copy()
    







    # keyword_df = request.session.get('keyword_df', [])
    
    df_comments_yt,top_positive_comments,top_negative_comments = sentiment_model(df_comments)
    
    df_comments_yt = df_comments_yt['Sentiment'].value_counts().reset_index()
    
    print(df_comments_yt)
    # Assuming your dataframe is named df_comments_yt_counts
    # Initialize counts
    p_count, neg_count, neu_count = 0, 0, 0
    
    sentiments = ['positive', 'negative', 'neutral']
    # Loop through sentiments and update counts
    for sentiment in sentiments:
        if sentiment in df_comments_yt['Sentiment'].values:
            count = df_comments_yt.loc[df_comments_yt['Sentiment'] == sentiment, 'count'].iloc[0]
            if sentiment == 'positive':
                p_count = count
            elif sentiment == 'negative':
                neg_count = count
            elif sentiment == 'neutral':
                neu_count = count


    # If you want to convert the counts to integers, you can do that:
    p_count = int(p_count)
    neg_count = int(neg_count)
    neu_count = int(neu_count)

    

    print(p_count)
    print(neg_count)
    print(neu_count)

    top_positive_comments_df = pd.DataFrame(top_positive_comments)
    top_negative_comments_df = pd.DataFrame(top_negative_comments) 
    
    # Assuming you have a variable named top_negative_comments
    # Assuming you have a variable named top_negative_comments
    print(top_positive_comments_df.reset_index(drop=True, inplace=True))

    # Initialize variables
    tp1 = ''
    tp2 = ''
    tp3 = ''
    tp4 = ''
    tp5 = ''

    # Checking if the DataFrame is not empty before proceeding
    if not top_positive_comments_df.empty:
        # Assigning values to variables if not empty
        if len(top_positive_comments_df['Sentence']) > 0:
            tp1 = top_positive_comments_df['Sentence'][0]
        if len(top_positive_comments_df['Sentence']) > 1:
            tp2 = top_positive_comments_df['Sentence'][1]
        if len(top_positive_comments_df['Sentence']) > 2:
            tp3 = top_positive_comments_df['Sentence'][2]
        if len(top_positive_comments_df['Sentence']) > 3:
            tp4 = top_positive_comments_df['Sentence'][3]
        if len(top_positive_comments_df['Sentence']) > 4:
            tp5 = top_positive_comments_df['Sentence'][4]

        # Printing the assigned variables
        print(tp1)
        print(tp2)
        print(tp3)
        print(tp4)
        print(tp5)
    else:
        print("top_positive_comments_df is empty. Doing nothing.")

    # Resetting the index and modifying the DataFrame in-place
    top_negative_comments_df.reset_index(drop=True, inplace=True)

    # Initialize variables
    tn1 = ''
    tn2 = ''
    tn3 = ''
    tn4 = ''
    tn5 = ''

    # Checking if the DataFrame is not empty before proceeding
    if not top_negative_comments_df.empty:
        # Assigning values to variables if not empty
        if len(top_negative_comments_df['Sentence']) > 0:
            tn1 = top_negative_comments_df['Sentence'][0]
        if len(top_negative_comments_df['Sentence']) > 1:
            tn2 = top_negative_comments_df['Sentence'][1]
        if len(top_negative_comments_df['Sentence']) > 2:
            tn3 = top_negative_comments_df['Sentence'][2]
        if len(top_negative_comments_df['Sentence']) > 3:
            tn4 = top_negative_comments_df['Sentence'][3]
        if len(top_negative_comments_df['Sentence']) > 4:
            tn5 = top_negative_comments_df['Sentence'][4]

        # Printing the assigned variables
        print(tn1)
        print(tn2)
        print(tn3)
        print(tn4)
        print(tn5)
    else:
        print("top_negative_comments_df is empty. Doing nothing.")

    df_yt_titles = df_comments['Video_Title'].unique()
    df_yt_titles = {'Video_Title':df_yt_titles}
    df_yt_titles = pd.DataFrame(df_yt_titles)
    df_yt_titles = df_yt_titles.iloc[0:5,:].to_dict(orient='records')
 

    yt_data_sen_wc =[]
    comments_concatenated = df_new['Comments'].str.cat(sep=' ')

    # Extract words from the concatenated comments
    words = re.findall(r'\w+', comments_concatenated.lower())

    # Remove stopwords, one-letter, one-digit words, prepositions, and all numbers
    filtered_words = [word for word in words if word not in stopwords.words("english") and len(word) > 1 and not word.isdigit() and word not in 
                ["a", "an", "the", "in", "on", "at", "to", "us", "day", "back", "contact", "cookies","cookie","help","menu"]]

    # Create a Counter to count word frequencies
    word_counter = Counter(filtered_words) 
    
    sorted_keywords = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

    # Get the top 20 keywords with counts
    top_keywords = sorted_keywords[:]

    # Append data to the keyword_data list
    for keyword, count in top_keywords:
        yt_data_sen_wc.append([keyword, count])
        
    yt_df_sen_wc = pd.DataFrame(yt_data_sen_wc, columns=["Keyword", "Count"])
    
    yt_df_sen_wc = yt_df_sen_wc.groupby('Keyword').agg({'Count': 'sum'})
    yt_df_sen_wc['Count'] = yt_df_sen_wc['Count'].astype('int')
    # Reset the index and sort by 'Count'
    yt_df_sen_wc = yt_df_sen_wc.reset_index().sort_values(by='Count',ascending =False)
    request.session['keyword_df_sen'] = yt_df_sen_wc.to_dict(orient='records')

    
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print(yt_df_sen_wc)
    
   
    wc_word_list = yt_df_sen_wc.iloc[0:9,:].to_dict(orient='records')


    # Generate the WordCloud image and get the base64 encoding
    wordcloud_image_yt_sen_wc = generate_wordcloud_image(yt_df_sen_wc)
    
    
    
    print(df_yt_titles)
    
    video_titles_text = ' '.join(item['Video_Title'] for item in df_yt_titles)
    yt_data_titles_wc =[]
    words = re.findall(r'\w+', video_titles_text.lower())

    # Remove stopwords, one-letter, one-digit words, prepositions, and all numbers
    filtered_words = [word for word in words if word not in stopwords.words("english") and len(word) > 1 and not word.isdigit() and word not in 
                ["a", "an", "the", "in", "on", "at", "to", "us", "day", "back", "contact", "cookies","cookie","help","menu"]]

    # Create a Counter to count word frequencies
    word_counter = Counter(filtered_words) 
    
    sorted_keywords = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

    # Get the top 20 keywords with counts
    top_keywords = sorted_keywords[:]

    # Append data to the keyword_data list
    for keyword, count in top_keywords:
        yt_data_titles_wc.append([keyword, count])
        
    yt_df_titles_wc = pd.DataFrame(yt_data_titles_wc, columns=["Keyword", "Count"])
    
    yt_df_titles_wc = yt_df_titles_wc.groupby('Keyword').agg({'Count': 'sum'})
    yt_df_titles_wc['Count'] = yt_df_titles_wc['Count'].astype('int')
    # Reset the index and sort by 'Count'
    yt_df_titles_wc = yt_df_titles_wc.reset_index().sort_values(by='Count',ascending =False)
    request.session['yt_df_titles_wc'] = yt_df_titles_wc.to_dict(orient='records')

    
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print(yt_df_titles_wc)
    
   
    wc_word_list = yt_df_titles_wc.iloc[0:9,:].to_dict(orient='records')


    # Generate the WordCloud image and get the base64 encoding
    wordcloud_image_yt_titles_wc = generate_wordcloud_image(yt_df_titles_wc)
    
    Avg_Views = channel_details['Avg_Views'][0]
    return render(request,"sentiment_analysis.html",
                  {'wordcloud_image_yt_sen_wc':wordcloud_image_yt_sen_wc,
                   'wordcloud_image_yt_titles_wc':wordcloud_image_yt_titles_wc,
                   'df_yt_titles':df_yt_titles,
                   'yt_url':channel_id,
                   'df_yt_comments':df_yt_comments,
                   'channel_name':channel_details['Channel_Name'][0],
                   'join_date':channel_details['Joined_Date'][0],
                   'subscribers':channel_details['Subscribers'][0],
                   'num_videos':channel_details['Total_No_Videos'][0],
                   'num_views':channel_details['Total_Views'][0],
                   'avg_n_v':Avg_Views,
                   'country':channel_details['Country'][0],
                   'p_count':p_count,
                   'neg_count':neg_count,
                   'neu_count':neu_count,
                   'tp1':tp1,'tp2':tp2,'tp3':tp3,'tp4':tp4,'tp5':tp5,
                   'tn1':tn1,'tn2':tn2,'tn3':tn3,'tn4':tn4,'tn5':tn5,                
                   })


def download_yt_comments(request):
    # Retrieve df_comments from the session
    df_comments = request.session.get('df_comments', [])

    # Check if df_comments is empty
    if not df_comments:
        messages.error(request, 'No data to download. Please perform the extraction first.')
        return redirect('website_keyword')  # Redirect to the index view

    # Convert the data back to a DataFrame
    df_comments = pd.DataFrame(df_comments)

    # Create an in-memory Excel file
    excel_buffer = BytesIO()

    # Write the DataFrame to an Excel file
    df_comments.to_excel(excel_buffer, index=False, sheet_name='YT_Comments')

    # Create an HttpResponse and set the headers
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=yt_comments.xlsx'

    # Write the Excel file content to the response
    excel_buffer.seek(0)
    response.write(excel_buffer.read())

    return response





def separate_into_sentences(paragraph):
    # Download the punkt tokenizer if not already downloaded
    nltk.download('punkt')

    # Use the nltk.sent_tokenize function to split the paragraph into sentences
    sentences = nltk.sent_tokenize(paragraph)

    return sentences


global text_passed

def proceed_sentiments(request):
    
    if request.method == "POST":
        
        text_passed = request.POST.get('text_or_par')
        print(text_passed)
        
        if text_passed:           
            seperated_sentences  = separate_into_sentences(text_passed)
            
            for i, sentence in enumerate(seperated_sentences, start=1):
                print(f"Sentence {i}: {sentence}")
                
            data_sentences = {'Comments': seperated_sentences, 'Sentiment': ''}
            df_sentences = pd.DataFrame(data_sentences)
            
            df_comments_txt,top_positive_sent,top_negative_sent = sentiment_model(df_sentences)
            
        else:
            return render(request, 'sentiment_analysis.html', {'error_message': 'Field should not be empty!'})

        
    df_comments_txt_copy = df_comments_txt.iloc[:5,:].to_dict(orient='records')
    full_data_sentiments = df_comments_txt
    print('======================')
    print(df_comments_txt)
    
    df_comments_txt_db = df_comments_txt.copy()
    
    
    for index, row in df_comments_txt_db.iterrows():
        sentimens_comments_instance, created = sentiments_comments.objects.get_or_create(Sentence=row['Sentence'], Sentiment=row['Sentiment'])

        sentimens_comments_instance.save()

    df_comments_txt = df_comments_txt['Sentiment'].value_counts().reset_index()

    print(df_comments_txt)
    # Assuming your dataframe is named df_comments_yt_counts
    # Initialize counts
    p_count, neg_count, neu_count = 0, 0, 0
    
    sentiments = ['positive', 'negative', 'neutral']
    # Loop through sentiments and update counts
    for sentiment in sentiments:
        if sentiment in df_comments_txt['Sentiment'].values:
            count = df_comments_txt.loc[df_comments_txt['Sentiment'] == sentiment, 'count'].iloc[0]
            if sentiment == 'positive':
                p_count = count
            elif sentiment == 'negative':
                neg_count = count
            elif sentiment == 'neutral':
                neu_count = count


    # If you want to convert the counts to integers, you can do that:
    p_count_txt = int(p_count)
    neg_count_txt = int(neg_count)
    neu_count_txt = int(neu_count)
    
    request.session['full_data_sentiments'] = full_data_sentiments.to_dict(orient='records')

    
    
    keyword_data_sen = []
    words = re.findall(r'\w+', text_passed.lower())

    # Remove stopwords, one-letter, one-digit words, prepositions, and all numbers
    filtered_words = [word for word in words if word not in stopwords.words("english") and len(word) > 1 and not word.isdigit() and word not in 
                ["a", "an", "the", "in", "on", "at", "to", "us", "day", "back", "contact", "cookies","cookie","help","menu"]]

    # Create a Counter to count word frequencies
    word_counter = Counter(filtered_words) 
    
    sorted_keywords = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

    # Get the top 20 keywords with counts
    top_keywords = sorted_keywords[:]

    # Append data to the keyword_data list
    for keyword, count in top_keywords:
        keyword_data_sen.append([keyword, count])
        
    keyword_df_sen = pd.DataFrame(keyword_data_sen, columns=["Keyword", "Count"])
    
    keyword_df_sen = keyword_df_sen.groupby('Keyword').agg({'Count': 'sum'})
    keyword_df_sen['Count'] = keyword_df_sen['Count'].astype('int')
    # Reset the index and sort by 'Count'
    keyword_df_sen = keyword_df_sen.reset_index().sort_values(by='Count',ascending =False)
    request.session['keyword_df_sen'] = keyword_df_sen.to_dict(orient='records')

    
    
    print(keyword_df_sen)
    
   
    wc_word_list = keyword_df_sen.iloc[0:9,:].to_dict(orient='records')


    # Generate the WordCloud image and get the base64 encoding
    wordcloud_image_sen = generate_wordcloud_image(keyword_df_sen)
    
    
    
    return render(request,"sentiments.html",
         {'df_comments_txt':df_comments_txt_copy,
            'p_count_txt':p_count_txt,
            'neg_count_txt':neg_count_txt,
            'neu_count_txt':neu_count_txt,
            'text_or_par':text_passed,
            'wordcloud_image_sen':wordcloud_image_sen,
            'wc_word_list':wc_word_list
               
                   }
    )
 

def download_txt_sentiments(request):
    # Retrieve df_comments from the session
    df_comments_txt = request.session.get('full_data_sentiments', [])

    # Check if df_comments is empty
    if not df_comments_txt:
        messages.error(request, 'No data to download. Please perform the extraction first.')
        return redirect('sentiments')  # Redirect to the index view

    # Convert the data back to a DataFrame
    df_comments_txt_df = pd.DataFrame(df_comments_txt)

    # Create an in-memory Excel file
    excel_buffer = BytesIO()

    # Write the DataFrame to an Excel file
    df_comments_txt_df.to_excel(excel_buffer, index=False, sheet_name='TXT_Sentiments')

    # Create an HttpResponse and set the headers
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=txt_sentiments.xlsx'

    # Write the Excel file content to the response
    excel_buffer.seek(0)
    response.write(excel_buffer.read())

    return response

def download_dataset_sen_wc(request):
        # Retrieve df_comments from the session
    dataset_sen_wc = request.session.get('keyword_df_sen', [])

    # Check if df_comments is empty
    if not dataset_sen_wc:
        messages.error(request, 'No data to download. Please perform the extraction first.')
        return redirect('sentiments')  # Redirect to the index view

    # Convert the data back to a DataFrame
    dataset_sen_wc_df = pd.DataFrame(dataset_sen_wc)

    # Create an in-memory Excel file
    excel_buffer = BytesIO()

    # Write the DataFrame to an Excel file
    dataset_sen_wc_df.to_excel(excel_buffer, index=False, sheet_name='SEN_WC_Sentiments')

    # Create an HttpResponse and set the headers
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=dataset_sen_wc_df.xlsx'

    # Write the Excel file content to the response
    excel_buffer.seek(0)
    response.write(excel_buffer.read())

    return response
        
def history(request):
    
    # Fetch data from the keyword_count_data and youtube_comments tables
    keyword_data = keyword_count_data.objects.all()
    yt_comments_data = youtube_comments.objects.all()
    sentiments_comments_data = sentiments_comments.objects.all()

    # Convert querysets to lists or dictionaries based on your requirement
    keyword_data_list = list(keyword_data.values())
    yt_comments_data_list = list(yt_comments_data.values())
    sentiments_comments_list = list(sentiments_comments_data.values())
    # Pass the data to the template
    return render(request, "history.html", {'keyword_data': keyword_data_list, 'yt_comments_data': yt_comments_data_list, 'sentiments_comments_data':sentiments_comments_list})

from django.http import JsonResponse
from .models import keyword_count_data, youtube_comments,sentiments_comments

def clear_history_view(request):
    # Clear records from the keyword_count_data table
    keyword_count_data.objects.all().delete()

    # Clear records from the youtube_comments table
    youtube_comments.objects.all().delete()
    
    
    sentiments_comments.objects.all().delete()

    return JsonResponse({'status': 'History cleared successfully'})

