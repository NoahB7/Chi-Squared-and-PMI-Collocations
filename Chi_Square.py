import sys,io,os,glob,nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import *
import math


class Chi_Square():        

    def __init__(self, top_chi_sq_ : int = 100):
        
        self.top_chi_sq_ : int = top_chi_sq_

    def probability_o11_(self, chi_test_dict, word1, word2) -> float:

        return chi_test_dict[word1][word2]
    
    def probability_o12_(self, inv_chi_test_dict, chi_test_dict, word1, word2) -> float:

        sum = 0
        for key, occurences in inv_chi_test_dict[word2].items():
            sum = sum + occurences
        
        sum = sum - chi_test_dict[word1][word2]
        
        return sum


    def probability_o21_(self, chi_test_dict, word1, word2) -> float:

        sum = 0
        for key, occurences in chi_test_dict[word1].items():
            sum = sum + occurences
        
        sum = sum - chi_test_dict[word1][word2]
        
        return sum


    def probability_o22_(self, prob_O11, prob_O12, prob_O21, n_tokens) -> float:

        return n_tokens - prob_O11 - prob_O12 - prob_O21

    
    def calculate_expentance_(self, prob_1, prob_2, prob_3, n_tokens) -> float:

        prob_a = (prob_1 + prob_2)/n_tokens

        prob_b = (prob_2 + prob_3)/n_tokens

        prob = prob_a * prob_b * n_tokens

        if prob < 0.000001:
            prob = 0.000001
        return prob
    
    def get_chi_(self, o_value, e_value) -> float:

        return ((o_value - e_value)*(o_value - e_value))/e_value
    
    def calculate_chi_value_(self, word1: str ,word2: str ,chi_test_dict: dict, inv_chi_test_dict: dict, n_tokens: int) -> float:
     
        o11 = self.probability_o11_(chi_test_dict, word1, word2)
        o12 = self.probability_o12_(inv_chi_test_dict, chi_test_dict, word1, word2)
        o21 = self.probability_o21_(chi_test_dict, word1, word2)
        o22 = self.probability_o22_(o11, o12, o21, n_tokens)

        #e11 = self.calculate_expentance_(o11, o12, o21, n_tokens)
        #e12 = self.calculate_expentance_(o11, o12, o22, n_tokens)
        #e21 = self.calculate_expentance_(o11, o21, o22, n_tokens)
        #e22 = self.calculate_expentance_(o12, o22, o21, n_tokens)

        #chi = self.get_chi_(o11, e11) + self.get_chi_(o12, e12) + self.get_chi_(o21, e21) + self.get_chi_(o22, e22)
        

        chi = (n_tokens * (o11*o22 - o12*o21) * (o11*o22 - o12*o21)) / ((o11+o12)*(o11+o21)*(o12+o22)*(o21+o22))

        return chi
    
    def mutual_information_(self, word1: str ,word2: str ,chi_test_dict: dict, inv_chi_test_dict: dict, n_tokens: int) -> float:

        o11 = self.probability_o11_(chi_test_dict, word1, word2)
        o12 = self.probability_o12_(inv_chi_test_dict, chi_test_dict, word1, word2)
        o21 = self.probability_o21_(chi_test_dict, word1, word2)
        o22 = self.probability_o22_(o11, o12, o21, n_tokens)

        p_i1j1 = (o11 - 1)/n_tokens
        p_j1 = (o11 + o21)/n_tokens
        p_i1 = (o11 + o12)/n_tokens

        return math.log2((p_i1j1)/(p_i1 * p_j1))

    def print_bigram_chi_sq_(self, bigrams: dict):

        i = self.top_chi_sq_
        sorted_values = sorted(bigrams.items(), key= lambda x : x[1][0], reverse=True)

        print('######################## TOP BIGRAMS COLLOCATIONS CHI-SQ ##########################')
        print('    {:<35}                {:<20}'.format('Bigram', 'Value'))
        for values in sorted_values:
            if i != 0:
                print('{:<3} {:<35}                {:<21}'.format(self.top_chi_sq_ - i + 1, values[0], values[1][0]))
            else:
                break
            i = i - 1
        
        print('\n\n######################## TOP BIGRAMS COLLOCATIONS CHI-SQ ##########################\n\n')

        print('######################## BOTTOM BIGRAMS COLLOCATIONS CHI-SQ ##########################')
        print('    {:<35}                {:<20}'.format('Bigram', 'Value'))
        
        i = len(sorted_values) - 5
        for values in sorted_values:
            if i < 1:
                print('{:<3} {:<35}                {:<21}'.format(len(sorted_values) + i + 1, values[0], values[1][0]))
            i = i - 1
        
        print('\n\n######################## BOTTOM BIGRAMS COLLOCATIONS CHI-SQ ##########################\n\n')

    def print_bigram_pmi_(self, bigrams: dict):

        i = self.top_chi_sq_
        sorted_values = sorted(bigrams.items(), key= lambda x : x[1][1], reverse=True)

        print('######################## TOP BIGRAMS COLLOCATIONS PMI ##########################')
        print('    {:<35}                {:<20}'.format('Bigram', 'Value'))
        for values in sorted_values:
            if i != 0:
                print('{:<3} {:<35}                {:<21}'.format(self.top_chi_sq_ - i + 1, values[0], values[1][1]))
            else:
                break
            i = i - 1
        
        print('\n\n######################## TOP BIGRAMS COLLOCATIONS PMI ##########################\n\n')

        print('######################## BOTTOM BIGRAMS COLLOCATIONS PMI ##########################')
        print('    {:<35}                {:<20}'.format('Bigram', 'Value'))
        
        i = len(sorted_values) - 5
        for values in sorted_values:
            if i < 1:
                print('{:<3} {:<35}                {:<21}'.format(len(sorted_values) + i + 1, values[0], values[1][1]))
            i = i - 1
        
        print('\n\n######################## BOTTOM BIGRAMS COLLOCATIONS PMI ##########################\n\n')
        

    def perform_nltk_bigram_chi_sq(self, words: dict, n : int = 1000):

        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, n)

        for bigram_0, bigram_1 in bigrams:
            if bigram_0[0] == 'a' and bigram_0[1] == 'b':
                print(bigram_0 + ' ' + bigram_1)    

    def __call__(self,bigram_list: dict ,chi_test_dict: dict, inv_chi_test_dict: dict, n_tokens: int):

        chi_dict = dict()

        # 3.841 critical value, not really relevant anyways since theyre being ranked by highest anyways which are well above 3.841
        for key,value in bigram_list.items():
            word1, word2 = key.split()
            chi = self.calculate_chi_value_(word1 ,word2, chi_test_dict, inv_chi_test_dict, n_tokens)
            mutual_information = self.mutual_information_(word1 ,word2, chi_test_dict, inv_chi_test_dict, n_tokens)

            if chi > 3.841:
                chi_dict[key] = chi,mutual_information
        
        self.print_bigram_chi_sq_(chi_dict)

        self.print_bigram_pmi_(chi_dict)

        #self.perform_nltk_bigram_chi_sq(sorted_values)
