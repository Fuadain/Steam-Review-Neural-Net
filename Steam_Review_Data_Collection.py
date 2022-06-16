import requests, json, string, nltk
from nltk.corpus import stopwords
#stop words only need to be downloaded once
#nltk.download("stopwords")     
stop_words = set(stopwords.words("english"))

print("Input game name:")
game = input()
file_name = (game + ".txt")
print("Input game store ID:")
store_ID = input()

#clean data
def cleanUp(data):
    text = data["text"]

    #detect ascii art
    if '⣿' in text:
        data["art_present"] = True

    # remove UNICODE characters
    text_encode = text.encode(encoding="ascii", errors="ignore")
    text = text_encode.decode()

    # cleaning the text to remove extra whitespace 
    text = " ".join([word for word in text.split()])

    # remove punctuation
    punct = set(string.punctuation) 
    text = "".join([ch for ch in text if ch not in punct])

    # make text lower case
    text = text.lower()
    
    # remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    data["text"] = text
    return data

#open file to write to
output_file = open(file_name, 'w')


def get_reviews(rev_type):
    cursor = "*"
    clean_reviews = []
    for i in range(100):
        if '+' in cursor:
            cursor = cursor.replace("+", "%2B")
        print(cursor)
        #get json review data, cursor is for next batch of data
        url = requests.get("https://store.steampowered.com/appreviews/" + str(store_ID) + "?json=1&language=english&num_per_page=10&review_type=" + rev_type +"&cursor=" + cursor)
        json_data = json.loads(url.text)    #not json.load
        #print(json_data["reviews"])
        reviews = json_data["reviews"]
        cursor = json_data["cursor"]

        #iterate through reviews to pull and clean data
        for r in reviews:
            #print(r["review"])
            review_data = {"text":r["review"], "voted_up":r["voted_up"], "art_present":False}
            clean_data = cleanUp(review_data)
            clean_reviews.append(clean_data)
            #print(clean_data)

            #write current stack of reviews to file
        for item in clean_reviews:
            output_file.write("%s\n" % item)
        #empty review list for stack
        clean_reviews = []
        reviews = []

get_reviews("positive")
get_reviews("negative")
output_file.close()


    



#r["voted_up"] gives positive or negative label
#if '⣿' in test: print("yes")   all steam ascii art uses braille ascii, and tends to start with \n
#else: print("no")