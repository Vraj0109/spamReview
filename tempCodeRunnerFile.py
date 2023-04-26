with open('D:\Study\sem 6\ML\lab_8\model.pkl', 'wb') as file:
    pickle.dump(classifier, file)

with open('D:\Study\sem 6\ML\lab_8\ectoriser.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)