import pandas as pd
import numpy as np
import random
import re
import pickle


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split




def makeToken(web):                      
	web = web.lower()
	token = []
	lex=[]
	dot_token_slash = []
	raw = str(web).split('/')
    
    
	if 'http:' in raw:   
		raw.remove('http:')
	if 'https:' in raw:    
		raw.remove('https:')  
	if '' in raw:
		raw.remove('')
        
	var=raw.pop(0)
	substring = "."
	count = var.count(substring)
	if count>2:
		lex.append('dotexceed')
        
	spcharacter = re.compile('[@_!#$%^&*()<>?/\|}{~:]') 
	if(spcharacter.search(var) != None):
		lex.append('spchar')
        
	digit=0    
	for i in var:    
		if i.isnumeric():
			digit += 1
	if digit>4:
		lex.append('digitcount')
                
	if len(var)>24:
		lex.append('lenexceed')
        
	var1 = str(var).split('.')
	c=0
	if 'com' in var1:   
		c=1 
	if 'en' in var1:   
		c=1
	if 'net' in var1:   
		c=1
	if 'org' in var1:   
		c=1
	if 'cc' in var1:   
		c=1 
	if c==1:
		lex.append('topdomain')
    
        
        
	for i in raw:
		raw1 = str(i).split('-')          
		slash_token = []
		for j in range(0,len(raw1)):
			raw2 = str(raw1[j]).split('.') 
			slash_token = slash_token + raw2
		dot_token_slash = dot_token_slash + raw1 + slash_token + lex
	token = list(set(dot_token_slash))        
   
	if 'com' in token:
		token.remove('com')   
	if 'www' in token:    
		token.remove('www')
	return token



def logit_predict(url):
    urls_data = pd.read_csv("url_data.csv")
    y = urls_data["label"]
    url_list = urls_data["url"]
    vectorizer = TfidfVectorizer(tokenizer=makeToken)
    X = vectorizer.fit_transform(url_list)
    model_logit = pickle.load(open("model_logit","rb"))

    predict = vectorizer.transform(url)
    new_predict = model_logit.predict(predict)
    return new_predict
    
