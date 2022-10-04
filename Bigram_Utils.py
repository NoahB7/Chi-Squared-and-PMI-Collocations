import sys,io,os,glob,nltk
from nltk.collocations import *
from nltk.metrics import BigramAssocMeasures
from Chi_Square import Chi_Square
from nltk.corpus import stopwords


class Bigram():

    def __init__(self, do_tokenize : bool = True, top_bigrams : int = 100, max_lines : int = 10000):
        
        self.do_tokenize_ : bool = do_tokenize
        self.top_bigrams_ : int = top_bigrams
        self.max_lines_ : int = max_lines
        self.current_offset_ : int = 0
        self.chi_test = Chi_Square()

        self.continue_ = True


    def perform_tokenization_(self, path: str) -> list:
        
        self.continue_ = False
        stop_words = set(stopwords.words('english'))
        corpus = ''
        for filename in glob.glob(os.path.join(path, '*.txt')):

            doc = io.open(filename, 'r', encoding='utf-8')  

            for lines in doc:
                corpus = corpus + lines

            doc.close()

        tokens = nltk.tokenize.word_tokenize(corpus)
        words = [w for w in tokens if not w.lower() in stop_words]
        words = [word.lower() for word in words if word.isalpha()]
        
        return words
    
    def print_top_nltk_bigrams_(self, words: dict):

        bigrams = nltk.bigrams(words)
        frequence = nltk.FreqDist(bigrams)

        print('######################## TOP BIGRAMS NLTK ##########################')

        print(frequence.most_common(100))

    def print_top_bigrams_(self, bigrams_dict: dict):
        
        i = self.top_bigrams_
        print('######################## TOP BIGRAMS COUNTED ##########################')
        print('    {:<20}                {:<20}'.format('Bigram', 'Value'))
        for values in bigrams_dict:
            if i != 0:
                print('{:<3} {:<20}                {:<21}'.format(self.top_bigrams_ - i + 1, values[0], values[1]))
            else:
                break
            i = i - 1
        
        print('\n\n######################## TOP BIGRAMS COUNTED ##########################\n\n')

        i = len(bigrams_dict) - 5
        print('######################## BOTTOM BIGRAMS COUNTED ##########################')
        print('    {:<20}                {:<20}'.format('Bigram', 'Value'))
        for values in bigrams_dict:
            if i < 1:
                print('{:<3} {:<20}                {:<21}'.format(len(bigrams_dict) + i + 1, values[0], values[1]))
            i = i - 1
        
        print('\n\n######################## BOTTOM BIGRAMS COUNTED ##########################\n\n')

    def __call__(self, file_localpath : str):

        dic_bigram = dict()
        chi_bigrams = dict()
        chi_test_dict = dict()
        inv_chi_test_dict = dict()
        bigram = ''

        # stores and returns counts of bigrams, first word occurences, and second word occurences
        # tokens are downcased and stop words are removed

        words = self.perform_tokenization_(file_localpath)

        # stores tokens on a dictionary organised by bigrams
        for token in range(len(words)) :
            if token != 0 :
                bigram = words[token-1] + ' ' + words[token]
                
                if bigram not in dic_bigram:
                    dic_bigram[bigram] = 1       
                else:
                    dic_bigram[bigram] = dic_bigram[bigram] + 1

        # uses the tokens stored to fill two dictionaries that contain the information to calculate
        # both chi_Sq and PMI. To do so and to keep complexity at N, we store the ocurrences of word1,word2
        # in one and an inverse dictionary to store word2,word1, so we can access it inmediately knowing the keys
        #from the bigrams      
        i = 0
        for bigram, value in dic_bigram.items() :
            i = i + 1
            if value > 1 and i > 0.1*len(dic_bigram) and 0.9 < len(dic_bigram):
                word1, word2 = bigram.split()

                chi_bigrams[bigram] = value

                if word1 not in chi_test_dict:
                    chi_test_dict[word1] = dict()
                
                if word2 not in chi_test_dict[word1]:
                    chi_test_dict[word1][word2] = value
                
                #inv chi value (to quickly access inverse information)
                if word2 not in inv_chi_test_dict:
                    inv_chi_test_dict[word2] = dict()
                
                if word1 not in inv_chi_test_dict[word2]:
                    inv_chi_test_dict[word2][word1] = value

        sorted_values = sorted(chi_bigrams.items(), key= lambda x : x[1], reverse=True)

        self.print_top_bigrams_(sorted_values)

        self.chi_test(chi_bigrams ,chi_test_dict, inv_chi_test_dict, len(chi_bigrams))

        #self.perform_nltk_bigrams_(words)
        

    


