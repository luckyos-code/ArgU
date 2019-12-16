from preprocessing.tokenizing import tokenizing_q

query = "fuck racism, it is not good Donald"
for i, token in enumerate(tokenizing_q(query)):
    print(i, "   ", token['token'], "    ", token['synonyms'], "    ", token['pos_tag'])