from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import random, json
from nltk_utils import *
from model import NeuralNet
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('english_intent.json', 'r') as json_data:
    intents = json.load(json_data, strict = "False")

with open('tamil_intent.json', 'r', encoding="utf-8") as json_data:
    tamil_intents = json.load(json_data , strict = "False")    

FILE = "data.pth"
data = torch.load(FILE)

TamilFILE = "tamil_data.pth"
tdata = torch.load(TamilFILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

tamilData = {
    "input_size" : tdata["input_size"],
    "hidden_size" : tdata["hidden_size"],
    "output_size" : tdata["output_size"],
    "all_words" : tdata['all_words'],
    "tags" : tdata['tags'],
    "model_state" : tdata["model_state"]
}
# print(tamilData)

tamilModel = NeuralNet(tamilData['input_size'], tamilData['hidden_size'], tamilData['output_size']).to(device)
tamilModel.load_state_dict(tamilData["model_state"])
tamilModel.eval()

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def testTamil(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, tamilData['all_words'])
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = tamilModel(X)

    _, predicted = torch.max(output, dim=1)
    # print(predicted)

    tag = tamilData['tags'][predicted.item()]
    # return tag
    # for i in tag:
    #     print(i, end = " ")
    # print('')

    for i in tamil_intents["intents"]:
        if(i["tag"] == tag):
            return random.choice(i["responses"])

    probs = torch.softmax(output, dim=1)
    # for i in probs:
    #     print(i, end = " ")
    # print('')
    prob = probs[0][predicted.item()]
    # print(tag)
    # print("hereee ")
    # print(intent['responses'])

    if prob.item() > 0.75:
        for intent in tamil_intents['intents']:
            # if tag == intent["tag"]:
                print(random.choice(intent['responses']))
                return (random.choice(intent['responses']))
            # else:
            #     return "I do not understand... Please elaborate"


def test(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    # print(predicted)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return (random.choice(intent['responses']))
    else:
        return ("I do not understand...")


@app.route('/', methods = ['GET', 'POST'])
def home():
    if(request.method == 'GET'):
        # print(request.args)
        data = "request"
    if(request.method == 'POST'):
        try:
            lang = request.get_json()['lang']
            if lang == "en-IN":
                data = test(request.get_json()['query'])
            else:
                print(request.get_json()['query'])
                data = testTamil(request.get_json()['query'])
            # data = request.get_json()['query'] + lang 
        except:
            data = ""
    # print(data)
    return jsonify({'data': data})

if __name__ == '__main__':
  
    app.run(debug = True)