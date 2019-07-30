import matplotlib.pyplot as plt 
import ast
import pandas as pd

articles_number =  [value for value in range(1597,22358,1597)]

def plot_composite_feature_performance():
    all_features_naive_bayes_acc = []
    all_features_log_reg_acc = []
    all_features_svm_acc = []

    fetched_composite_acc= []
    #reading result of composite feature testing in all classifers
    with open('composite_results.txt','r') as f:
        for line in f.readlines():
            fetched_composite_acc.append(line.strip())

    for acc in fetched_composite_acc:
        accuracy_list=  list(map(float, acc.split(',')))
        #formatting each element by 2 decimal point
        accuracies = [ '%.2f' % elem for elem in accuracy_list ]    
        all_features_naive_bayes_acc.append(float(accuracies[0]))
        all_features_log_reg_acc.append(float(accuracies[1]))
        all_features_svm_acc.append(float(accuracies[2]))

    plt.figure(10)
    plt.plot(articles_number,all_features_naive_bayes_acc,label='Naive Bayes')
    plt.plot(articles_number,all_features_log_reg_acc,label='Logistic Regression')
    plt.plot(articles_number,all_features_svm_acc,label='SVM')
    plt.title("Composite Features performance")
    plt.xlabel('No of News articles')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../visuals/composite_features.png')

def plot_individual_features_performance():
    fetched_indivual_features_data =[]
    unigram_features = []
    ngram_features = []
    character_features = []
    count_features =[]
    lda_features = []

    with open('individual_results.txt') as f:
        for line in f.readlines():
            fetched_indivual_features_data.append(ast.literal_eval(line.strip()))

    for each_data in fetched_indivual_features_data:

        unigram_features.append(each_data['unigram'])
        ngram_features.append(each_data['ngram'])
        character_features.append(each_data['character'])
        count_features.append(each_data['count'])
        lda_features.append(each_data['lda'])

    #individual feature perfromance over news articels by naive bayes
    unigram_features_nb = []
    ngram_features_nb = []
    character_features_nb = []
    count_features_nb =[]
    lda_features_nb = []

    #individual feature perfromance over news articels by logistic regression
    unigram_features_lr = []
    ngram_features_lr = []
    character_features_lr = []
    count_features_lr =[]
    lda_features_lr = []

    #individual feature perfromance over news articels by svm
    unigram_features_sv = []
    ngram_features_sv = []
    character_features_sv = []
    count_features_sv =[]
    lda_features_sv = []



    for index,unigram in enumerate(unigram_features):

        unigram_features_nb.append(unigram[0])
        unigram_features_lr.append(unigram[1])
        unigram_features_sv.append(unigram[2])


        ngram_features_nb.append(ngram_features[index][0])
        ngram_features_lr.append(ngram_features[index][1])
        ngram_features_sv.append(ngram_features[index][2])

        character_features_nb.append(character_features[index][0])
        character_features_lr.append(character_features[index][1])
        character_features_sv.append(character_features[index][2])


        count_features_nb.append(count_features[index][0])
        count_features_lr.append(count_features[index][1])
        count_features_sv.append(count_features[index][2])


        lda_features_nb.append(lda_features[index][0])
        lda_features_lr.append(lda_features[index][1])
        lda_features_sv.append(lda_features[index][2])    


    plt.figure(1)
    plt.plot(articles_number,unigram_features_nb,label='Naive Bayes')
    plt.plot(articles_number,unigram_features_lr,label='Logistic Regression')
    plt.plot(articles_number,unigram_features_sv,label='SVM')
    plt.title('Unigram TFIDF Feature Performance')
    plt.xlabel('No of News articles')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../visuals/Unigram TFIDF Feature Performance.png')

    plt.figure(2)
    plt.plot(articles_number,ngram_features_nb,label='Naive Bayes')
    plt.plot(articles_number,ngram_features_lr,label='Logistic Regression')
    plt.plot(articles_number,ngram_features_sv,label='SVM')
    plt.title('Ngram TFIDF Feature Performance')
    plt.xlabel('No of News articles')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../visuals/Ngram TFIDF Feature Performance.png')

    plt.figure(3)
    plt.plot(articles_number,character_features_nb,label='Naive Bayes')
    plt.plot(articles_number,character_features_lr,label='Logistic Regression')
    plt.plot(articles_number,character_features_sv,label='SVM')
    plt.title('Character TFIDF Feature Performance')
    plt.xlabel('No of News articles')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../visuals/Character TFIDF Feature Performance.png')


    plt.figure(4)
    plt.plot(articles_number,count_features_nb,label='Naive Bayes')
    plt.plot(articles_number,count_features_lr,label='Logistic Regression')
    plt.plot(articles_number,count_features_sv,label='SVM')
    plt.title('Count Vector Feature Performance')
    plt.xlabel('No of News articles')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../visuals/Count Vector Feature Performance.png')





    plt.figure(5)
    plt.plot(articles_number,lda_features_nb,label='Naive Bayes')
    plt.plot(articles_number,lda_features_lr,label='Logistic Regression')
    plt.plot(articles_number,lda_features_sv,label='SVM')
    plt.title('LDA Topic Model Feature Performance')
    plt.xlabel('No of News articles')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../visuals/LDA Topic Model Feature Performance.png')




    plt.figure(6)
    plt.plot(articles_number,unigram_features_nb,label='Unigram TFIDF')
    plt.plot(articles_number,ngram_features_nb,label='Ngram TFIDF')
    plt.plot(articles_number,character_features_nb,label='Character TFIDF')
    plt.plot(articles_number,count_features_nb,label='Count Vector')
    plt.plot(articles_number,lda_features_nb,label='LDA Vector')
    plt.title('Individual Features performance in Naive Bayes')
    plt.xlabel('No of News articles')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../visuals/Individual Features performance in Naive Bayes.png')





    plt.figure(7)
    plt.plot(articles_number,unigram_features_lr,label='Unigram TFIDF')
    plt.plot(articles_number,ngram_features_lr,label='Ngram TFIDF')
    plt.plot(articles_number,character_features_lr,label='Character TFIDF')
    plt.plot(articles_number,count_features_lr,label='Count Vector')
    plt.plot(articles_number,lda_features_lr,label='LDA Vector')
    plt.title('Individual Features performance in Logistic Regression')
    plt.xlabel('No of News articles')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../visuals/Individual Features performance in Logistic Regression.png')




    plt.figure(8)
    plt.plot(articles_number,unigram_features_sv,label='Unigram TFIDF')
    plt.plot(articles_number,ngram_features_sv,label='Ngram TFIDF')
    plt.plot(articles_number,character_features_sv,label='Character TFIDF')
    plt.plot(articles_number,count_features_sv,label='Count Vector')
    plt.plot(articles_number,lda_features_sv,label='LDA Vector')
    plt.title('Individual Features performance in SVM')
    plt.xlabel('No of News articles')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../visuals/Individual Features performance in SVM')


def plot_data_distribution():
    df = pd.read_csv('../datasets/train.csv')
    main_df = df.loc[:,['id','text','label']]
    main_df = main_df.dropna()

    real_news = len(main_df[main_df['label']==1])
    fake_news = len(main_df[main_df['label']==0])

    plt.figure(9)
    plt.pie([real_news,fake_news],labels=['Real News','Fake News'],shadow=True, startangle=180,explode=(0,0.1),autopct='%1.1f%%',)
    plt.savefig('../visuals/Data distribution.png')








plot_data_distribution()
plot_composite_feature_performance()
plot_individual_features_performance()










    
   






