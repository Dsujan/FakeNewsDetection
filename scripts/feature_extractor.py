from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim.models import Doc2Vec
import data_refractor as dr
import numpy as np

class FeatureXtractor:
    
    def __init__(self,train_data,test_data):

        train_data_for_pos, train_data = dr.perform_pre_processing(train_data)
        test_data_for_pos, test_data = dr.perform_pre_processing(test_data)


        train_data_detokenized = dr.detokenize_data(train_data)
        test_data_detokenized = dr.detokenize_data(test_data)

        # print('Pos tagging train tokens...')
        # pos_tagged_train_data = dr.tag_tokens_by_pos(train_data_for_pos)
        # print('Pos tagging test tokens')
        # pos_tagged_test_data = dr.tag_tokens_by_pos(test_data_for_pos)

        

        self.uni_tfidf_vector, self.uni_train_tf_feature,  self.uni_test_tf_feature =  self.calc_tfidf_vectors(train_data,test_data)
        self.char_tfidf_vector, self.char_train_tf_feature, self.char_test_tf_feature = self.calc_tfidf_vectors(train_data_detokenized,test_data_detokenized,lvl="char")
        self.ngram_tfidf_vector, self.ngram_train_tf_feature, self.ngram_test_tf_feature = self.calc_tfidf_vectors(train_data,test_data,lvl="ngram")

        self.count_vector, self.count_vector_train_feature, self.count_vector_test_feature = self.calc_count_vector(train_data,test_data)
        
        self.doc2vec_model,self.train_doc_vec,self.test_doc_vec = self.calc_doc_embedings(train_data,test_data)
        
        # self.pos_count_vector,self.pos_count_vector_train_feature,self.pos_count_vector_test_feature = self.calc_count_vector(pos_tagged_train_data,pos_tagged_test_data)

        self.lda_topic_model,self.lda_train_vec_feature,self.lda_test_vec_feature = self.lda_topic_modeling(self.count_vector_train_feature,self.count_vector_test_feature)

    def dummy_tokenizer(self, tokens):
        return tokens

    def calc_tfidf_vectors(self,train_data,test_data,lvl=None):
        
        if lvl == None:
            print("Extracting unigram features")
            tfidf_vector = TfidfVectorizer(min_df=1,stop_words="english",lowercase=False,tokenizer=self.dummy_tokenizer)

        elif lvl == "char":
            print("Extracting char features")
            tfidf_vector = TfidfVectorizer(min_df=1,stop_words="english",lowercase=False,analyzer="char_wb",ngram_range=(2,3))
        
        elif lvl == "ngram":
            print("Extracting ngram features")
            tfidf_vector = TfidfVectorizer(min_df=1,stop_words="english",lowercase=False,tokenizer=self.dummy_tokenizer,ngram_range=(2,2))



        train_data_vector = tfidf_vector.fit_transform(train_data)

        test_data_vector = tfidf_vector.transform(test_data)

        return tfidf_vector,train_data_vector,test_data_vector


    def calc_count_vector(self,train_data,test_data):
        print('Init count vector')
        count_vector = CountVectorizer(tokenizer=self.dummy_tokenizer,lowercase=False,analyzer="word")
        print('train transform vector')
        train_count_vec = count_vector.fit_transform(train_data)
        print('test transform vector')
        test_count_vec = count_vector.transform(test_data)

        return count_vector,train_count_vec,test_count_vec

    def calc_doc_embedings(self,train_data,test_data):
        
        #loading apnews pretrained do2vec gensim model
        filename = "../doc2vec_model/apnews_dbow/apnews_dbow/doc2vec.bin"
        model = Doc2Vec.load(filename)
        print('train doc2vec')
        inferred_vector_train = np.vstack(model.infer_vector(singledoc) for singledoc in train_data)
        print('test doc2vec')
        inferred_vector_test = np.vstack(model.infer_vector(singledoc) for singledoc in test_data)

        return model,inferred_vector_train,inferred_vector_test
    
    def lda_topic_modeling(self,train_count,test_count):

        model = LDA(n_components=20)
        print('train lda')
        lda_train_vec =model.fit_transform(train_count)
        print('test lda')
        lda_test_vec  = model.transform(test_count)

        return model,lda_train_vec,lda_test_vec







    

    



















