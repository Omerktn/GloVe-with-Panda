# -*- coding: utf-8 -*-
import pandas as pd
from glove import Glove
import sys
import csv
import codecs
import io
import h5py
from nltk import word_tokenize
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")

glove_data_file = open("vectors.txt", "r")
words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None,
                      quoting=csv.QUOTE_NONE, na_values=None, keep_default_na=False, low_memory=False)

def main():
    # This code is special for producing News vector. It can be modified to produce vector of a spesific file.
    newsdir = '/media/omer/Hard Disk/42bin_haber/news/'
    vecdir = '/home/omer/Documents/gv_vectors/TW/'
    glovedir = '/home/omer/GloVe - 1.2/'

    news_count = [3724, 3262, 6673, 5847, 1155, 2792, 1953, 1383, 1819, 8632, 771, 1939, 644]
    news_name = ["dunya", "ekonomi", "genel", "guncel", "kultur-sanat", "magazin", "planet", "saglik", "siyaset",
                 "spor", "teknoloji", "turkiye", "yasam"]
    stop_words = 0

    for i in range(len(news_name)):
        dataset = []
        with h5py.File(vecdir + news_name[i] + ".h5", 'w') as hf:
            hf.create_dataset("tw", data=dataset)  # create first data file


        for j in range(1, news_count[i] + 1):
            count_line = 0
            text = []
            try:
                with open(newsdir + news_name[i] + "/" + str(j) + ".txt", "r") as file:
                    text = file.readlines()
            except:
                print("excepted.")
                continue

            tmp_vec = np.zeros(100)
            for in_line in text:  # Get avereage vector of a file
                try:
                    line = word_tokenize(in_line)
                except:
                    continue
                new_vec = getVector(line)
                tmp_vec = add(tmp_vec, new_vec)
                if int(new_vec[0] + new_vec[1]):
                    count_line = count_line + 1  # count only non-zero vectors

            if count_line:
                tmp_vec = div(tmp_vec, count_line)
            dataset.append(tmp_vec)

            stop_words = stop_words + (len(text) - count_line)
            if (j - 1) % 100 == 0:
                # delete
                with h5py.File(vecdir + news_name[i] + ".h5", "a") as f:
                    del f["tw"]
                # save
                with h5py.File(vecdir + news_name[i] + ".h5", 'w') as hf:
                    hf.create_dataset("tw", data=dataset)
                sys.stdout.write("\r [" + news_name[i] + "] : " + str(j) + "/" + str(news_count[i]))

        sys.stdout.write("\r [" + news_name[i] + "] : " + str(j) + "/" + str(news_count[i]) + "\n")
        # delete
        with h5py.File(vecdir + news_name[i] + ".h5", "a") as f:
            del f["tw"]
            # save
        with h5py.File(vecdir + news_name[i] + ".h5", 'w') as hf:
            hf.create_dataset("tw", data=dataset)


    print("Non-exist words per news: " + str(float(stop_words / 39500)))


def getVector(word_array):
    tmp_vec = np.zeros(100)
    count_word = 0
    for in_word in word_array:
        try:
            tmp_vec = add(tmp_vec, vec(in_word))
            count_word = count_word + 1
        except:
            pass

    if (count_word):
        tmp_vec = div(tmp_vec, count_word)
    return tmp_vec


def add(vec1, vec2):
    for i in range(len(vec1)):
        vec1[i] = vec1[i] + vec2[i]
    return vec1


def div(vec, m):
    for i in range(len(vec)):
        vec[i] = vec[i] / m
    return vec


def vec(w):
    return words.loc[w].as_matrix()


if __name__ == "__main__":
    main()
