import re

import bs4
import pandas as pd
from langdetect import detect
from googletrans import Translator

reviews = pd.read_csv('reviews.csv')

translator = Translator(service_urls=['translate.googleapis.com'])


# remove html elements
def remove_html(word):
    try:
        word = bs4.BeautifulSoup(word).get_text(separator=" ")
    except:
        word = ''
    return word


def translate_comments(index, comment):
    try:
        lang = detect(comment)
    except:
        lang = 'en'
    print(str(index) + lang)
    if lang == 'en':
        return comment
    else:
        return translator.translate(comment, src=lang, dest='en').text


# remove \r and ,
def remove_comma(dest, restr=''):
    res = re.compile(u'[\r|,]')
    return res.sub(restr, dest)


print('Start removing html elements')
reviews['comments'] = reviews['comments'].apply(lambda x: remove_html(x), 1)
print('Finished removing html elements')

print('Start translating comments')
new_review = reviews.apply(lambda x: translate_comments(x[0], x[5]), 1)
print('Finished translating comments')

print('Start removing \\r and ,')
new_review = new_review.apply(lambda x: remove_comma(x))
print('Finished removing \\r and ,')

print('Start writing to file')
final = pd.concat([reviews['listing_id'], new_review.to_frame()], axis=1, join='outer')
final.to_csv(path_or_buf='new_reviews.csv')
print('Finished writing to file')
