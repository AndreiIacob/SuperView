# State of the art & risk assessment

The first task that we accomplished together as a team was to search other relevant sources, articles for our future work. These sources represent a starting point in our project. 

# [Yelp challenge winners](https://www.yelp.com/dataset/challenge/winners)
  - ## [Round 1 - March 2013](https://engineeringblog.yelp.com/2013/10/yelp-dataset-challenge-winners-round-two-now-live.html)
    - ### [Clustered Layout Word Cloud for User Generated Review](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_WordCloud.pdf)
      They used dependency parsing to produce a word-cloud based on reviews that allows users to search based on words. Evaluation was done based on the amount of time a user took to decide between 2 restaurants.
    - ### [Hidden Factors and Hidden Topics: Understanding Rating Dimensions with Review Text](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_HiddenFactors.pdf)
      Combines [Latent-Factor Recommender Systems](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_5) and [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) for recommendation and genre discovery.
    - ### [Improving Restaurants by Extracting Subtopics from Yelp Reviews](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_ImprovingRestaurants.pdf)
      Used [Online LDA](https://github.com/blei-lab/onlineldavb) to extract 50 topics from reviews and analysed the most frequently occurring ones.
    - ### [Inferring Future Business Attention](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_InferringFuture.pdf)
      Used reviews to predict future attention of a business. Combined time-dependent features and text features formed from the top 300 keywords then filtered into 100 adjectives and noun groups. Feature selection was then used to reduce the number of features.
   
  - ## [Round 2 - February 10, 2014](https://engineeringblog.yelp.com/2014/02/yelp-dataset-challenge-round-2-winner-and-new-data.html)
    - ### [Valence Constrains the Information Density of Messages](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_InformationDensity.pdf)
      A study of how the affective state of a reviewer (the star rating) affects the information content of the words used as measured by Review-internal entropy, Average unigram information and Average conditional information.
      
  - ## [Round 3 - November 2014](https://engineeringblog.yelp.com/2014/11/yelp-dataset-challenge-round-3-winners-and-dataset-tools-for-round-4.html)
    - ### [Personalizing Yelp Star Ratings: A Semantic Topic Modeling Approach](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_PersonalizingRatings.pdf)
      Uses a modified [LDA]((https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)) that takes into account star ratings for the purpose of showing users a personalized review score as opposed to the average.
    - ### [On the Efficiency of Social Recommender Networks](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_NetworkEfficiency.pdf)
      An evaluation of the Yelp's social network as evaluated by the how friends reviews affect a user's decision to visit a business and how it affects information propagation. 
      
  - ## [Round 4 -  December 31, 2014](https://www.yelp.com/dataset/challenge/winners)
    - ### [Collective Factorization for Relational Data: An Evaluation on the Yelp Datasets](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_CollectiveFactorization.pdf)
      Combines relational databases to train a small representation of entities and predicts the probability that there is a relation between these 2 entities and suggestions for use for improving rating predictions for viewers.
    - ### [Mining Quality Phrases from Massive Text Corpora](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_MiningQualityPhrases.pdf)
      Proposes a method for quality phrase extraction, frequent phrase extraction and phrase quality evaluation leveraging the use of other work on Phrasal Segmentation. The quality of a phrase is based on popularity, concordance, informativeness and completeness.
    - ### [Oversampling with Bigram Multinomial Naive Bayes to Predict Yelp Review Star Classes](https://kevin11h.github.io/YelpDatasetChallengeDataScienceAndMachineLearningUCSD/)
      Outlines models for predicting review score based on text using Naive Bayes and Random Forest. Implemented improvements on Naive Bayes that makes it better suited to skewed datasets. The best model was Random Forest using bag of words bigrams, it that achieves mean precison of 0.73 and recall of 0.71.
      
  - ## [Round 5 - January 2016](https://engineeringblog.yelp.com/2016/01/yelp-dataset-challenge-round5-winner.html)
    - ### [From Group to Individual Labels using Deep Features](http://mdenil.com/media/papers/2015-deep-multi-instance-learning.pdf)
      Proposed a new method for classifying  instances within a group. It applies this method on Yelp reviews to label sentences within a review and words within sentences with the aid of a convolutional neural network, logistic regression and the bag of words model.
      
  - ## [Round 6 - August 2016](https://engineeringblog.yelp.com/2016/08/yelp-dataset-challenge-round6-winner.html)
    - ### Topic Regularized Matrix Factorization for Review Based Rating Prediction
      Predicts the star rating a user would give using Topic Regularized Matrix Factorization. This method uses topic learned through the use of an [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) to constrain the learning rate of matrix factorization.
      
  - ## [Round 7 - January 2017](https://engineeringblog.yelp.com/2017/01/dataset-round-7-winners-and-announcing-round-9.html)
    - ### [Semantic Scan: Detecting Subtle, Spatially Localized Events in Text Streams](https://arxiv.org/pdf/1602.04393.pdf)
      Describes a method for detecting anomalous events through text, such as emerging trends or problems. It makes use of a modified version of an [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) specifically designed to detect new topics alongside spatial scanning methods to detect anomalous events in emergency complaints and the Yelp dataset as measured by [Hellinger Distances](https://en.wikipedia.org/wiki/Hellinger_distance), spatial overlap and document overlap.
 
 - ## [Round 8 - June 2017](https://engineeringblog.yelp.com/2017/06/dataset-round-8-winners.html)
    - ### [Clustered Model Adaption for Personalized Sentiment Analysis](http://www.cs.virginia.edu/~hw5x/paper/fp1158-gongA.pdf)
      Used Dirichlet Priors combined with Markov chains to establish the group membership of a user then used global variables, group variables and user variables in a Expectation–Maximization algorithm to predict the user score arriving at a F1 score of 0.901 for positive Yelp reviews, 0.669 for negative Yelp reviews and 0.843 for positive Amazon reviews and 0.547 for negative Amazon reviews.
      
# [Other Studies using the Yelp dataset](https://scholar.google.com/scholar?q=citation%3A+Yelp+Dataset&btnG=&hl=en&as_sdt=0%2C5)
 - ## [Predicting Business Ratings on Yelp](http://cs229.stanford.edu/proj2015/013_report.pdf) 
        Describes ways to use matrix factorization to predict a user's review score of a business based on the business's average review score, the average score the user's reviews, global statistics, the categories the businesses are assigned to, data from the Yelp social network. The best model gives an error of 1.037 on the test set.
 - ## [Predicting the category of the restaurant using reviews and tips and recommend the restaurants based on cuisine preferences](http://cgi.soic.indiana.edu/~arunsank/Yelp-FinalProjectReport.pdf)
        This approach describes how the reviews are analysed based on ratings and stars in order to see which words are often used in good ratings (preprocessing step). In order to classify the category of the restaurant, they use Naïve Bayes, Locality Sensitive Hashing. For recommendation step, they use the data about an user and the “behaviour data” which is the data associated with the restaurants that the user has previously visited. In this part they use Matrix Factorization Recommender, Item Based Similarity Recommender, Popularity Recommender.
 - ## [A project made by students from Columbia University and Harvard University](http://www.columbia.edu/~yw2668/yelp.html)
        This project takes 3 approaches: using text mining, they find the most popular food in a restaurant, a food map where they locate the neighbourhood with the best restaurants in the city, and an analysis on how Yelp developed over years in USA.
 - ## [Restaurants Review Star Prediction for Yelp Dataset](https://cseweb.ucsd.edu/~jmcauley/cse255/reports/fa15/017.pdf)
        In this project, they use linear regression, random forest tree and latent factor model combined with the sentiment analysis.
 - ## [Eat, Rate, Love Project](https://www.springboard.com/blog/eat-rate-love-an-exploration-of-r-yelp-and-the-search-for-good-indian-food)
        This project relies on predicting the best Indian food restaurant using the Yelp dataset. It is used methods like generating weights on reviewers (it's analysed how many reviews wrote a reviewer) and calculating an "authenticity" rating (the author selects the Indian restaurants and analyses how many Indian people are visiting the location using reviewers data filtered by Indian names). In the end the steps are bound together, and the result is a filter that you can use to select the best restaurant with a type of food.
 - ## [Trends Found on Round 6 of the Yelp Dataset Challenge](http://www2.rmcil.edu/dataanalytics/v2016/papers/Trends_Found_on_Round_6_of_the_Yelp_Dataset_Challenge.pdf)
        This article is a study on demographic trends (which businesses are the most popular in a town). It is used Hadoop, Hive, PIG and Tableau for managing the big data.
 - ## [Yelp Dataset Challenge: Review Rating Prediction](https://www.researchgate.net/publication/303331726_Yelp_Dataset_Challenge_Review_Rating_Prediction)
        The author did a research on how can you predict a review rating based on some features that are drawn using unigrams, bigrams, trigrams, Latent Semantic Indexing. For training the model, it is used 4 algorithms: Logistic Regression, Naive Bayes classification, Perceptron and Support Vector Machines.
        The best result was done using Logistic Regression.
 - ## Other Projects
     - ### [Predicting a Business’ Star in Yelp from Its Reviews’ Text Alone](https://pdfs.semanticscholar.org/130e/cc92626b32b89a27dbcda7357cd4b18abdc5.pdf)
     - ### [Predicting the ambiance of restaurants using only the wording of Yelp reviews](https://medium.com/fun-with-data-and-stats/predicting-the-ambiance-of-restaurants-using-only-the-wording-of-yelp-reviews-954413b6d490)

The second accomplished task was to look for other sources (other databases) that we can use for our project. What we found are the following:

# Resources available
 - ## [AMAZON food reviews](http://snap.stanford.edu/data/web-FineFoods.html)
 - ## [AMAZON movie reviews](http://snap.stanford.edu/data/web-Movies.html)
 - ## [AMAZON reviews on different categories of products]
    ### [1](http://snap.stanford.edu/data/web-Amazon.html)
    ### [2](http://jmcauley.ucsd.edu/data/amazon/)
 - ## [Mining Quality Phrases from Massive Text Corpora] (https://aminer.org/citation)
    The authors used this dataset to create a review model.
 - ## [Movie review data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
    ### [Projects done using this dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/otherexperiments.html)
 - ## [Large Movie Review Dataset](http://ai.stanford.edu/~amaas//data/sentiment/)
 - ## [Skytrax User Reviews Dataset](https://github.com/quankiquanki/skytrax-reviews-dataset)
 - ## [Multi-Domain Sentiment Dataset](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/)
 - ## [Review data sets for "Latent Aspect Rating Analysis"](http://sifaka.cs.uiuc.edu/~wang296/Data/index.html)
 - ## [Opinosis Dataset - Topic related review sentences](http://kavita-ganesan.com/opinosis-opinion-dataset)
    This dataset contains reviews on hotels, cars, products.
 - ## [OpinRank Dataset - Reviews from TripAdvisor and Edmunds](http://kavita-ganesan.com/entity-ranking-data)
 - ## [MovieLens Reviews Dataset](https://grouplens.org/datasets/movielens/)
 - ## [From Amateurs to Connoisseurs: Modelling the Evolution of User Expertise through Online Reviews](https://cseweb.ucsd.edu/~jmcauley/pdfs/www13.pdf)
 - ## [BeerAdvocate Reviews](http://snap.stanford.edu/data/web-BeerAdvocate.html)
 - ## [RateBeer Reviews](http://snap.stanford.edu/data/web-RateBeer.html)
 - ## [CellarTracker Reviews](http://snap.stanford.edu/data/web-CellarTracker.html)
# Tools available 
 - ## [Google Trends](https://trends.google.com/trends/)
 - ## [City Search](http://www.citysearch.com/world)
