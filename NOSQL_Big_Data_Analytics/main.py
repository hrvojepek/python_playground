import os
import sys
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import nltk
from bs4 import BeautifulSoup as Soup
from nltk.collocations import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.text import Text
from pymongo import MongoClient


def parse_and_save_data_from_file():
    handler = open(directory + "/" + file, 'r').read()
    soup = Soup(handler, "lxml")
    car = {}
    car_name = ""
    for docno in soup.find_all("docno"):
        car_name = docno

    car["name"] = car_name.text
    car["data_array"] = []
    for doc in soup.find_all("doc"):
        car_data = {}
        for date in doc.find_all("date"):
            car_data["date"] = date.text
        for text in doc.find_all("text"):
            car_data["text"] = text.text
        for favorite in doc.find_all("favorite"):
            car_data["favorite"] = favorite.text
        car["data_array"].append(car_data)
    coll_posts.insert_one(car)
    return


def calculate_word_freq(input_coll, output_coll):
    all_data_count = []

    for doc in input_coll.find():
        print(doc["name"] + " - " + str(doc["data_array"]))
        data_array_count = []
        for car_data in doc["data_array"]:
            text_words = {}
            favourite_words = {}
            data_count = {'date': car_data["date"]}

            for text_word in car_data["text"].lower().split():
                if text_word.isalpha():
                    text_words[text_word] = car_data['text'].count(text_word)
            data_count['text_count'] = text_words

            for favourite_word in car_data["favorite"].lower().split():
                if favourite_word.isalpha():
                    favourite_words[favourite_word] = car_data['favorite'].count(favourite_word)
            data_count['fav_count'] = favourite_words

            data_array_count.append(data_count)
        data = {'name': doc['name'], 'data_array_count': data_array_count}
        output_coll.insert_one(data)
        all_data_count.append(data)


def all_word_freq():
    words = {}
    all_text = ''

    for obj in coll_posts.find():
        for post in obj['data_array']:
            all_text += post['text']
        for post in obj['data_array']:
            all_text += post['favorite']

    for word in all_text.lower().replace('.', ' ').split():
        if words.get(word):
            words[word] += 1
        else:
            words[word] = 1
    return words


def get_sorted_word_list(words):
    return sorted(words, key=words.get, reverse=True)


def nltk_freq():
    all_text = ''

    for obj in coll_posts.find():
        for post in obj['data_array']:
            all_text += post['text']
        for post in obj['data_array']:
            all_text += post['favorite']

    filtered_text_list = []
    for word in all_text.lower().replace('.', ' ').split():
        if word.isalpha():
            filtered_text_list.append(word)
    filtered_text = ' '.join(filtered_text_list)

    # regex = re.compile('[{}.,()\';!-:]')
    # filtered_text = regex.sub(' ', all_text.lower())

    filtered_tokens = nltk.word_tokenize(filtered_text)

    fd = nltk.FreqDist(filtered_tokens)
    return {"tokens": filtered_tokens, "filtered_text": filtered_text, "fd": fd}


def remove_stop_words(tokens, coll_filtered_tokens_nltk):
    stop_words = set(stopwords.words('english'))

    tokens_without_stop_words = [token.lower() for token in tokens if
                                 token.lower() not in stop_words and len(token) > 1]
    coll_filtered_tokens_nltk.insert_one({'tokens_without_stop_words': tokens_without_stop_words})

    return tokens_without_stop_words


def get_bigrams(filtered_tokens, freq):
    bigram_finder = BigramCollocationFinder.from_words(filtered_tokens)
    bigram_finder.apply_freq_filter(freq)

    bigrams = list(bigram_finder.ngram_fd.items())
    bigrams.sort(key=lambda item: item[-1], reverse=True)
    return bigrams


def draw_sna_network(ngram):
    G = nx.Graph()

    node_sizes = {}
    for bg in ngram:
        G.add_edge(bg[0][0], bg[0][1])

        if not node_sizes.get(bg[0][0]):
            node_sizes[bg[0][0]] = bg[1] / 10
        else:
            node_sizes[bg[0][0]] += bg[1] / 10

        if not node_sizes.get(bg[0][1]):
            node_sizes[bg[0][1]] = bg[1] / 10
        else:
            node_sizes[bg[0][1]] += bg[1] / 10

    pos = nx.fruchterman_reingold_layout(G)
    sizes = []
    for item in node_sizes:
        sizes.append(node_sizes[item])

        sys.stdout = open('outputs/sna_info.txt', "w")
        print("Info:")
        print(nx.info(G))
        print("Degree histogram:")
        print(nx.degree_histogram(G))
        print("Density:")
        print(nx.density(G))
        print("Number of nodes :")
        print(G.number_of_nodes())
        print("Number of edges :")
        print(G.number_of_edges())
        dc = nx.degree_centrality(G)

        Sorted_degree = sorted(dc.items(), key=itemgetter(1), reverse=True)
        print("Sorted degree :")
        print(Sorted_degree[0:5])

        bc = nx.betweenness_centrality(G)
        Sorted_betweenness = sorted(bc.items(), key=itemgetter(1), reverse=True)
        print("Sorted betweenness :")
        print(Sorted_betweenness[0:5])

        cc = nx.closeness_centrality(G)
        Sorted_closeness = sorted(cc.items(), key=itemgetter(1), reverse=True)
        print("Sorted closeness :")
        print(Sorted_closeness[0:5])
    nx.draw_networkx(G, pos, node_size=sizes)
    plt.show()


def most_common_concordance(size):
    fd = nltk.FreqDist(tokens_without_stop_words)
    for word, frequency in fd.most_common(size):
        print(Text(tokens).concordance(word))


def concordance_by_word(word):
    sys.stdout = open('concordance/' + word + '.txt', "w")
    print(Text(tokens).concordance(word, width=200, lines=100))


def get_word_synonyms_from_words(word, words):
    word_synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemma_names():
            if lemma in words:
                word_synonyms.append(lemma)
    return word_synonyms


def get_list_freq_of_words(text):
    return words_with_freq.get(w)


def create_data_for_genie(filename, car_name, positive_words, negative_words, use_context_method):
    stop = set(stopwords.words('english'))

    file_text = "Adjectives Car\n"
    dataset_data_text = ''

    all_text = ''
    pos_words = []
    neg_words = []

    col_find = []
    if car_name != '':
        col_find = [a for a in coll_posts.find({'name': car_name})]
    else:
        col_find = [a for a in coll_posts.find()]

    for doc in col_find:
        for data in doc['data_array']:
            all_text += data['text']
            all_text += data['favorite']

    all_words = [a for a in all_text.lower().replace('.', ' ').split() if a not in stop and a.isalpha()]

    all_tokens = nltk.word_tokenize(' '.join(all_words))
    fd = nltk.FreqDist(all_tokens)

    if not use_context_method:
        good_synonyms = []
        for pos_word in positive_words:
            good_synonyms += get_word_synonyms_from_words(pos_word, all_words)

        bad_synonims = []
        for neg_word in negative_words:
            bad_synonims += get_word_synonyms_from_words(neg_word, all_words)

        num_of_good_words = 0
        for synonym in set(good_synonyms):
            freq = fd.get(synonym)
            num_of_good_words += freq
            for i in range(freq):
                file_text += synonym + " " + "good\n"

        num_of_bad_words = 0
        for synonym in set(bad_synonims):
            freq = fd.get(synonym)
            num_of_bad_words += freq
            for i in range(freq):
                file_text += synonym + " " + "bad\n"

        dataset_data_text += get_data_for_data_set(car_name, num_of_good_words, num_of_bad_words, len(all_text.split()),
                                                   len(all_words))

    # This is another way i didn't implement, searching words by context
    if use_context_method:
        idx = nltk.text.ContextIndex(all_words)

        for word in positive_words:
            pos_words += idx.similar_words(word)

        for word in negative_words:
            neg_words += idx.similar_words(word)

        for word in pos_words:
            file_text += word + " good\n"

        for word in neg_words:
            file_text += word + " bad\n"

    print_to_file('training_sets', filename + ".txt", file_text)
    return dataset_data_text


def drop_database():
    coll_posts.drop()
    coll_posts_freq.drop()
    coll_nltk_posts_freq.drop()
    coll_nltk_posts_freq_filtered.drop()
    coll_linguistic_classification.drop()


def print_to_file(directory, filename, data):
    file = open(directory + '/' + filename, "w")
    file.write(str(data))
    file.close()


def get_data_for_data_set(car_name, num_of_good_words, num_of_bad_words, num_of_words, num_of_filtered_words):
    text = car_name + "\n"
    text += "Number of words - " + str(num_of_words) + "\n"
    text += "Number of filtered (useful) words (without stop words) - " + str(num_of_filtered_words) + "\n"
    text += "Number of good words (praises) - " + str(num_of_good_words) + "\n"
    text += "Number of bad words (words with bad comment) - " + str(num_of_bad_words) + "\n\n\n"
    return text


if __name__ == '__main__':
    directory = "./dataset"
    client = MongoClient()
    db = client.cars

    # COLUMNS
    coll_posts = db.posts  # list of all posts group by a car
    coll_posts_freq = db.posts_freq  # manual words frequency
    coll_nltk_posts_freq = db.nltk_posts_freq  # nltk words frequency
    coll_nltk_posts_freq_filtered = db.nltk_posts_freq_filtered  # without stop words
    coll_linguistic_classification = db.linguistic_classification  # linguistic classification

    drop_database()

    # VARIABLES
    words_with_freq = []
    words_with_freq_sorted = []

    filtered_text = ''
    tokens = []
    tokens_without_stop_words = []

    # nltk.download()

    # 1 Reading datasets and saving in mongoDB group by Audi models
    for file in os.listdir(directory):
        parse_and_save_data_from_file()

    # 2 Frequency of words group by words
    sys.stdout = open('outputs/data.txt', "w")
    calculate_word_freq(coll_posts, coll_posts_freq)

    # 3 Manual calculating frequency of words
    sys.stdout = open('outputs/words_with_frequency.txt', "w")
    words_with_freq = all_word_freq()
    print(words_with_freq)

    # 4	Sorting words by frequency
    sys.stdout = open('outputs/words_sorted_by_freq.txt', "w")
    words_with_freq_sorted = get_sorted_word_list(words_with_freq)
    for w in words_with_freq_sorted:
        print(w)

    # 5 NLTK word frequency
    sys.stdout = open('outputs/nltk_freq_top_30.txt', "w")
    result = nltk_freq()
    tokens = result["tokens"]
    fd = result["fd"]
    filtered_text = result["filtered_text"]
    for word, frequency in fd.most_common(30):
        print(u'{} {}'.format(word, frequency))

    # 6 Removing stop words
    sys.stdout = open('outputs/nltk_tokens_without_stop_words.txt', "w")
    tokens_without_stop_words = tokens_without_stop_words = remove_stop_words(tokens, coll_nltk_posts_freq_filtered)
    for t in tokens_without_stop_words:
        print(t)

    # 7 Graph frequency
    fd = nltk.FreqDist(tokens_without_stop_words)
    fd.plot(40, cumulative=False)

    # 8 Bigrams creation
    sys.stdout = open('outputs/bigrams.txt', "w")
    bigrams = get_bigrams(tokens_without_stop_words, 10)
    for b in bigrams:
        print(b)

    # 9 SNA net
    draw_sna_network(bigrams)

    # 10 Concordance of words
    sys.stdout = open('outputs/concordance.txt', "w")
    most_common_concordance(10)

    # 11 Lingvistička klasifikacija
    sys.stdout = open('outputs/linguistic_classification.txt', "w")
    linguistic_classification = nltk.tag.pos_tag(set(tokens_without_stop_words))
    coll_linguistic_classification.insert_one({'linguistic_classification': linguistic_classification})
    for lc in linguistic_classification:
        print(lc)

    # 12 Bayesova mreža
    positive_words = ['great', 'good', 'nice', 'outstanding', 'lovely', 'professional', 'perfect', 'excellent']
    negative_words = ['bad', 'worst', 'problem', 'negative']

    data_set_metrics = ''
    data_set_metrics += create_data_for_genie('all_audi_cars', '', positive_words, negative_words, False)
    for file in os.listdir(directory):
        data_set_metrics += create_data_for_genie(file, file, positive_words, negative_words, False)

    print_to_file('outputs', 'data_set_metrics.txt', data_set_metrics)

    # Concordance by wanted words
    wanted_words_concordance = ['price', 'engine', 'brakes', 'interior', 'power', 'transmission']
    for word in wanted_words_concordance:
        concordance_by_word(word)

    # drop_database()
