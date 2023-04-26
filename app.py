from flask import Flask, render_template, request,send_file
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
import requests
from bs4 import BeautifulSoup

HEADERS = ({'User-Agent':
			'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
			AppleWebKit/537.36 (KHTML, like Gecko) \
			Chrome/90.0.4430.212 Safari/537.36',
			'Accept-Language': 'en-US, en;q=0.5'})

app = Flask(__name__)
# Load the model


model = pickle.load(open('D:\Study\sem 6\ML\lab_9\model.pkl', 'rb'))
vectorizer = pickle.load(open('D:\Study\sem 6\ML\lab_9\ectoriser.pkl', 'rb'))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    def getdata(url):
       r = requests.get(url, headers=HEADERS)
       return r.text    
    def html_code(url): 
        # pass the url
        # into getdata function
        htmldata = getdata(url)
        soup = BeautifulSoup(htmldata, 'html.parser')   
        # display html code
        return (soup)
    
    url = request.form['text']
    # url ="https://www.amazon.in/OnePlus-Eternal-Green-128GB-Storage/product-reviews/B0BQJLCQD3/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

    soup = html_code(url)
    # print(soup)


    def cus_data(soup):
        # find the Html tag
        # with find()
        # and convert into string
        data_str = ""
        cus_list = []
    
        for item in soup.find_all("span", class_="a-profile-name"):
            data_str = data_str + item.get_text()
            cus_list.append(data_str)
            data_str = ""
        return cus_list
    
    
    # cus_res = cus_data(soup)
    # cus_res.remove(cus_res[0])
    # cus_res.remove(cus_res[0])
    # print(cus_res)

    def cus_rev(soup):
        # find the Html tag
        # with find()
        # and convert into string
        data_str = ""   
        for item in soup.find_all("div", class_="a-row a-spacing-small review-data"):
        	data_str = data_str + item.get_text()   
        result = data_str.split("\n")
        return (result)



    # print(rev_result)
    # print(len(cus_res))
    # print(len(rev_result))

    name = []
    rev = []
    for i in range (10):
        tempurl = url + "&pageNumber="+ str(i+1)
        soup = html_code(tempurl)
        cus_res = cus_data(soup)
        name.extend(cus_res)

        rev_data = cus_rev(soup)
        rev_result = []
        for i in rev_data:
            if i == "":
                pass
            else:
                rev_result.append(i)
            rev_result

        rev.extend(rev_result)

        # Preprocess the input
        # input_data = vectorizer.transform([text])
    # print(input_data.shape)

    # Make prediction using th.e loaded model
    import nltk
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    def remove_stop_words(doc):
        words = word_tokenize(doc)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    rev = [remove_stop_words(r) for r in rev]

    # Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define a function to lemmatize a sentence
    def lemmatize_sentence(sentence):
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        # Lemmatize each word in the sentence
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        # Join the lemmatized words back into a sentence
        lemmatized_sentence = ' '.join(lemmatized_words)
        return lemmatized_sentence

    rev = [lemmatize_sentence(sentence) for sentence in rev]
    inp = vectorizer.transform(rev)
    prediction = model.predict(inp)
    data = {'message': rev,
		'lebel':prediction }

    # Create DataFrame
    cf = pd.DataFrame(data)

    # Save the output.
    cf.to_csv('D:\Study\sem 6\ML\lab_9\pre.csv')

    # Render the result template with the prediction
    return send_file('pre.csv')
    # return render_template('index.html', prediction=prediction)
    # return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
