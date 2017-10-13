# [Yelp challenge winners](https://www.yelp.com/dataset/challenge/winners)
  - ## [Round 1 - March 2013](https://engineeringblog.yelp.com/2013/10/yelp-dataset-challenge-winners-round-two-now-live.html)
    - ### [Clustered Layout Word Cloud for User Generated Review](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_WordCloud.pdf)
      They used dependency parsing to produce a word-cloud based on reviews that allows users to search based on words. Evaluation was done based on the amount of time a user took to decide between 2 restaurants.
    - ### [Hidden Factors and Hidden Topics: Understanding Rating Dimensions with Review Text](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_HiddenFactors.pdf)
      Combines [Latent-Factor Recommender Systems](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_5) and [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) for recommendation and genre discovery.
    - ### [Improving Restaurants by Extracting Subtopics from Yelp Reviews](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_ImprovingRestaurants.pdf)
      Used [Online LDA](https://github.com/blei-lab/onlineldavb) to extract 50 topics from reviews and analyzed the most frequently occuring ones.
    - ### [Inferring Future Business Attention](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_InferringFuture.pdf)
      Used reviews to predict future attention of a buisness. Combined time-dependent features and text features formed from the top 300 keywords then filtered into 100 adjectives and noun groups. Feature selection was then used to reduce the number of features.
   
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
      Combines relational databases to train a small representation of enteties and predicts the probability that there is a relation between these 2 enteties and sugesstions for use for improving rating predictions for viewers.
    - ### [Mining Quality Phrases from Massive Text Corpora](https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_MiningQualityPhrases.pdf)
      Proposes a method for quality phrase extraction, frequent phrase extraction and phrase quality evaluation leveraging the use of other work on Phrasal Segmentation. The quality of a phrase is based on popularity, concordance, informativeness and completeness.
    - ### [Oversampling with Bigram Multinomial Naive Bayes to Predict Yelp Review Star Classes](https://kevin11h.github.io/YelpDatasetChallengeDataScienceAndMachineLearningUCSD/)
      Outlines models for predicting review score based on text using Naive Bayes and Random Forest. Implemented improvements on Naive Bayes that makes it better suited to skewed datasets. The best model was Random Foresst using bag of words bigrams, it that achieves mean precison of 0.73 and recall of 0.71.
      
  - ## [Round 5 - January 2016](https://engineeringblog.yelp.com/2016/01/yelp-dataset-challenge-round5-winner.html)
    - ### [From Group to Individual Labels using Deep Features](http://mdenil.com/media/papers/2015-deep-multi-instance-learning.pdf)
      Proposed a new method for classifying  instances within a group. It applies this method on Yelp reviews to label sentences within a review and words within sentences with the aid of a convolutional neural network, logistic regression and the bag of words model.
      
  - ## [Round 6 - August 2016](https://engineeringblog.yelp.com/2016/08/yelp-dataset-challenge-round6-winner.html)
    - ### Topic Regularized Matrix Factorization for Review Based Rating Prediction
      Predicts the star rating a user would give using Topic Regularized Matrix Factorization. This method uses topic learned through the use of an [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) to constrin the learning rate of matrix factorization.
      
  - ## [Round 7 - January 2017](https://engineeringblog.yelp.com/2017/01/dataset-round-7-winners-and-announcing-round-9.html)
    - ### [Semantic Scan: Detecting Subtle, Spatially Localized Events in Text Streams](https://arxiv.org/pdf/1602.04393.pdf)
      Describes a method for detecting anomalous events through text, such as emerging trends or problems. It makes use of a modified version of an [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) specifically designed to detect new topics alongside spatial scanning methods to detect anomalous events in emergency complaints and the Yelp dataset as measured by [Hellinger Distances](https://en.wikipedia.org/wiki/Hellinger_distance), spatial overlap and document overlap.
      
# Other Studies using the Yelp dataset
    
   - ### [Predicting Business Ratings on Yelp](http://cs229.stanford.edu/proj2015/013_report.pdf) 
    Describes ways to use matrix factorization to predict a user's review score of a buisness based on the buisness's average review score, the average score the user's reviews, global statistics, the categories the buisnesses are assigned to, data from the Yelp social network. The best model gives an error of 1.037 on the test set.
