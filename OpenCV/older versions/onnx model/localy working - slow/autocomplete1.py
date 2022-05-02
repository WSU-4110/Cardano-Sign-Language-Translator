import numpy
import csv
from fast_autocomplete.misc import read_csv_gen
from fast_autocomplete import AutoComplete


def get_words(path):

    csv_gen = read_csv_gen('dict.xlsx', csv_func=csv.DictReader)

    words = {}

    for line in csv_gen:
        make = line['make']
        model = line['model']
        if make != model:
            local_words = [model, '{} {}'.format(make, model)]
            while local_words:
                word = local_words.pop()
                if word not in words:
                    words[word] = {}
        if make not in words:
            words[make] = {}
    return words

words = ""
synonyms = ""
autocomplete = AutoComplete(words=, synonyms=synonyms)

autocomplete.search(word = 'pap', max_cost = 3, size = 3)

#autocomplete program that uses a levensthainDistance matrix to calculate the similarity between two words, by looking at their 
#distance 

def calcDictDistance(word, numWords):
    file = open('1-1000.txt', 'r') 
    lines = file.readlines() 
    file.close()
    dictWordDist = []
    wordIdx = 0
    
    for line in lines: 
        wordDistance = levenshteinDistanceDP(word, line.strip())
        if wordDistance >= 10:
            wordDistance = 9
        dictWordDist.append(str(int(wordDistance)) + "-" + line.strip())
        wordIdx = wordIdx + 1

    closestWords = []
    wordDetails = []
    currWordDist = 0
    dictWordDist.sort()
    for i in range(numWords):
        currWordDist = dictWordDist[i]
        wordDetails = currWordDist.split("-")
        closestWords.append(wordDetails[1])
    return closestWords

def levenshteinDistanceDP(token1, token2):
    distances = numpy.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    distance = distances[len(token1)][len(token2)]

    return distance;


def printDistances(distances, token1Length, token2Length):
    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(int(distances[t1][t2]), end=" ")
        print()


#distance = levenshteinDistanceDP("kelm", "hello")

print(calcDictDistance("pas", 5))

