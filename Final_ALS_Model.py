# import libraries
import os

from pyspark.shell import sqlContext
from pyspark.sql.types import *
from pyspark.sql import functions as F

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


# list directories
files_path = '../DataSet/'  # File path here ../DataSet/ #../MDM_Project/Dataset/

triplets_file = files_path + 'train_triplets.txt'
songs2tracks_file = files_path + 'song_to_tracks.txt'
metadata_file = files_path + 'track_metadata.csv'

# Handle Windows.
if os.path.sep != '/':
    triplets_file = triplets_file.replace('/', os.path.sep)
    songs2tracks_file = songs2tracks_file.replace('/', os.path.sep)
    metadata_file = metadata_file.replace('/', os.path.sep)

# Creating schema so the cluster only runs through the data once
triplets_schema = StructType(
    [StructField('userId', StringType()),
     StructField('songId', StringType()),
     StructField('Plays', IntegerType())]
)
songs2tracks_schema = StructType(
    [StructField('songId', StringType()),
     StructField('trackId', StringType())]
)
metadata_schema = StructType(
    [StructField('trackId', StringType()),
     StructField('title', StringType()),
     StructField('songId', StringType()),
     StructField('release', StringType()),
     StructField('artist_id', StringType()),
     StructField('artist_mbid', StringType()),
     StructField('artist_name', StringType()),
     StructField('duration', DoubleType()),
     StructField('artist_familiarity', DoubleType()),
     StructField('artist_hotttness', DoubleType()),
     StructField('year', IntegerType()),
     StructField('track_7digitalid', IntegerType()),
     StructField('shs_perf', DoubleType()),
     StructField('shs_work', DoubleType())]
)

# load the data into DataFrames
plays_df = sqlContext.read.format('com.databricks.spark.csv') \
    .options(delimiter='\t', header=True, inferSchema=False) \
    .schema(triplets_schema) \
    .load(triplets_file)

songs2tracks_df = sqlContext.read.format('com.databricks.spark.csv') \
    .options(delimiter=',', header=True, inferSchema=False) \
    .schema(songs2tracks_schema) \
    .load(songs2tracks_file)

metadata_df = sqlContext.read.format('com.databricks.spark.csv') \
    .options(delimiter=',', header=True, inferSchema=False) \
    .schema(metadata_schema) \
    .load(metadata_file)

# change ids from strings to integers
userId_change = plays_df.select('userId').distinct().select('userId',F.monotonically_increasing_id().alias('new_userId'))
user_als_id_LUT = sqlContext.createDataFrame(userId_change.rdd.map(lambda x: x[0]).zipWithIndex(), StructType([StructField("userId", StringType(), True),StructField("user_als_id", IntegerType(), True)]))

songId_change = plays_df.select('songId').distinct().select('songId', F.monotonically_increasing_id().alias('new_songId'))
song_als_id_LUT = sqlContext.createDataFrame(songId_change.rdd.map(lambda x: x[0]).zipWithIndex(), StructType([StructField("songId", StringType(), True),StructField("song_als_id", IntegerType(), True)]))

# RUN BELOW TWO LINES TO CHECK IF THE  NEW USER_ID, SONG_ID GENERATED PROPERLY
# user_als_id_LUT.show(5)
# song_als_id_LUT.show(5)

# Get total unique users and songs
unique_users = user_als_id_LUT.count()
unique_songs = song_als_id_LUT.count()
print('Number of unique users: {0}'.format(unique_users))
print('Number of unique songs: {0}'.format(unique_songs))

# Joining the new ID's to the Plays_df
plays_df_2 = plays_df.join(user_als_id_LUT,'userId').join(song_als_id_LUT,'songId')

# remove half users to make more manageable
plays_df_2 = plays_df_2.filter(plays_df_2.user_als_id < unique_users / 2)

# Summary of each DataFrame
plays_df_2.cache()
plays_df_2.show(5)

songs2tracks_df.cache()
songs2tracks_df.show(5)

metadata_df.cache()
metadata_df.show(5)

#Total Listens(plays) of Each SongID
Total_listens = plays_df_2.groupBy('songId') \
                                              .agg(F.count(plays_df_2.Plays).alias('User_Count'),
                                                            F.sum(plays_df_2.Plays).alias('Total_Plays')) \
                                                       .orderBy('Total_Plays', ascending = False)

print('Total Listens of Each SONG_ID:')
Total_listens.show(3, truncate=False)

# Joining with metadata to get artist and song title for the Total_Listens
Song_names = Total_listens.join(metadata_df, 'songId' ) \
                                                      .filter('User_Count >= 200') \
                                                      .select('artist_name', 'title', 'songId', 'User_Count','Total_Plays') \
                                                      .orderBy('Total_Plays', ascending = False)

print('Complete Details of Songs Listened')
Song_names.show(20, truncate = False)

# We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
seed = 180229192
(split_1, split_2, split_3) = plays_df_2.randomSplit([0.6, 0.2, 0.2], seed = seed)

# Let's cache these datasets for performance
train_set = split_1.cache()
validation_set = split_2.cache()
test_set = split_3.cache()

print('Training: {0}, validation: {1}, test: {2}\n'.format(
  train_set.count(), validation_set.count(), test_set.count())
)
train_set.show(5)
validation_set.show(5)
test_set.show(5)

# Number of plays needs to be double type
validation_set = validation_set.withColumn("Plays", validation_set["Plays"].cast(DoubleType()))
validation_set.show(5)

## MODEL GENERATION (Alternating Least Squares)

# initialising our First ALS learner
als_01 = ALS()
# Setting the parameters for the method
als_01.setMaxIter(5)\
   .setSeed(seed)\
   .setItemCol("song_als_id")\
   .setRatingCol("Plays")\
   .setUserCol("user_als_id")

# computing an evaluation metric for our test dataset
# We Create an RMSE evaluator using the label and predicted columns

reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="Plays", metricName="rmse")

tolerance = 0.03
ranks = [4, 8, 12, 16]
regParams = [0.15, 0.2, 0.25]
errors = [[0]*len(ranks)]*len(regParams)
models = [[0]*len(ranks)]*len(regParams)
err = 0
min_error = float('inf')
best_rank = -1

i = 0
for regParam in regParams:
  j = 0
  for rank in ranks:
    # Set the rank here:
    als_01.setParams(rank = rank, regParam = regParam)
    # Create the model with these parameters.
    model = als_01.fit(train_set)
    # Run the model to create a prediction. Predict against the validation_df.
    predictions = model.transform(validation_set)

    # Remove NaN values from prediction (due to SPARK-14489)
    predicted_plays = predictions.filter(predictions.prediction != float('nan'))
    predicted_plays = predicted_plays.withColumn("prediction", F.abs(F.round(predicted_plays["prediction"],0)))

    # Run the previously created RMSE evaluator, reg_eval, on the predicted_plays DataFrame
    error = reg_eval.evaluate(predicted_plays)
    errors[i][j] = error
    models[i][j] = model
    print ('For rank :',rank, ' regularization parameter:', regParam,' the RMSE is', error)
    if error < min_error:
      min_error = error
      best_params = [i,j]
    j += 1
  i += 1

als_01.setRegParam(regParams[best_params[0]])
als_01.setRank(ranks[best_params[1]])
print ('The best model was trained with regularization parameter %s' % regParams[best_params[0]])
print ('The best model was trained with rank %s' % ranks[best_params[1]])
my_model = models[best_params[0]][best_params[1]]

#predicted plays
predicted_plays.show(10)

## TESTING THE MODEL

test_set = test_set.withColumn("Plays", test_set["Plays"].cast(DoubleType()))
predict_df = my_model.transform(test_set)

# Remove NaN values from prediction (due to SPARK-14489)
Test_predictions = predict_df.filter(predict_df.prediction != float('nan'))

# Round floats to whole numbers
Test_predictions = Test_predictions.withColumn("prediction", F.abs(F.round(Test_predictions["prediction"],0)))
# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
Test_RMSE = reg_eval.evaluate(Test_predictions)
print('The model had a RMSE on the test set of {0}'.format(Test_RMSE))

# Comparing the Model
avg_plays = train_set.groupBy().avg('Plays').select(F.round('avg(Plays)'))
avg_plays.show(3)
train_avg_plays = avg_plays.collect()[0][0]
print('The average number of plays in the dataset is {0}'.format(train_avg_plays))

# Add a column with the average rating
test_avg = test_set.withColumn('prediction', F.lit(train_avg_plays))

# Run the previously created RMSE evaluator, reg_eval, on the test_for_avg_df DataFrame
test_avg_RMSE = reg_eval.evaluate(test_avg)
print("The RMSE on the average set is {0}".format(test_avg_RMSE))

## PREDICTION FOR AN USER

UserID = 13
songs_listened = plays_df_2.filter(plays_df_2.user_als_id == UserID) \
    .join(metadata_df, 'songId') \
    .select('song_als_id', 'artist_name', 'title') \
 \
# Generating List of Listened Songs
listened_songs_list = []
for song in songs_listened.collect():
    listened_songs_list.append(song['song_als_id'])

print('Songs user has listened to:')
songs_listened.select('artist_name', 'title').show()

# generate dataframe of unlistened songs
songs_unlistened = plays_df_2.filter( ~ plays_df_2['song_als_id'].isin(listened_songs_list)) \
    .select('song_als_id').withColumn('user_als_id', F.lit(UserID)).distinct()

# feed unlistened songs into model
predicted_listens = my_model.transform(songs_unlistened)

# remove NaNs
predicted_listens = predicted_listens.filter(predicted_listens['prediction'] != float('nan'))

# print output
print('Predicted Songs:')
predicted_listens.join(plays_df_2, 'song_als_id') \
    .join(metadata_df, 'songId') \
    .select('artist_name', 'title', 'prediction') \
    .distinct() \
    .orderBy('prediction', ascending=False) \
    .show(10)

## MAKING PREDICTIONS BASED ON  'SONGS LISTENED TO' AT LEAST TWICE
plays_df_2more_plays = plays_df.join(user_als_id_LUT, 'userId') \
                                       .join(song_als_id_LUT, 'songId') \
                                       .filter(plays_df.Plays >= 2)\
                                       .distinct()

total_entries_2more = plays_df_2more_plays.count()
print('Total enties with two or more plays: {0}'.format(total_entries_2more))

plays_df_2more_plays = plays_df_2more_plays.filter(plays_df_2more_plays.user_als_id < (unique_users)*0.8) \
                                                   .select('user_als_id', 'song_als_id', 'Plays')
plays_df_2more_plays.cache()

# We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
seed = 1800083193
(split_01, split_02, split_03) = plays_df_2more_plays.randomSplit([0.6, 0.2, 0.2], seed = seed)

# Let's cache these datasets for performance
trainset_2more = split_01.cache()
validationset_2more = split_02.cache()
testset_2more = split_03.cache()

print('Training: {0}, validation: {1}, test: {2}\n'.format(
  trainset_2more.count(), validationset_2more.count(), testset_2more.count())
)
validationset_2more = validationset_2more.withColumn("Plays", validationset_2more["Plays"].cast(DoubleType()))
test_2more = testset_2more.withColumn("Plays", testset_2more["Plays"].cast(DoubleType()))

trainset_2more.show(3)
validationset_2more.show(3)
testset_2more.show(3)

# Let's initialize our ALS learner
als_2more = ALS()

# Now set the parameters for the method
als_2more.setMaxIter(2)\
   .setSeed(seed)\
   .setItemCol("song_als_id")\
   .setRatingCol("Plays")\
   .setUserCol("user_als_id")

# Now let's compute an evaluation metric for our test dataset
# We Create an RMSE evaluator using the label and predicted columns
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="Plays", metricName="rmse")

tolerance = 0.03
ranks = [4, 8, 12, 16]
regParams = [0.1, 0.15, 0.2, 0.25]
errors = [[0]*len(ranks)]*len(regParams)
models = [[0]*len(ranks)]*len(regParams)
err = 0
min_error = float('inf')
best_rank = -1
i = 0
for regParam in regParams:
  j = 0
  for rank in ranks:
    # Set the rank here:
    als_2more.setParams(rank = rank, regParam = regParam)
    # Create the model with these parameters.
    model = als_2more.fit(trainset_2more)
    # Run the model to create a prediction. Predict against the validation_df.
    predict_df = model.transform(validationset_2more)

    # Remove NaN values from prediction (due to SPARK-14489)
    predicted_plays_df = predict_df.filter(predict_df.prediction != float('nan'))
    predicted_plays_df = predicted_plays_df.withColumn("prediction", F.abs(F.round(predicted_plays_df["prediction"],0)))
    # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
    error = reg_eval.evaluate(predicted_plays_df)
    errors[i][j] = error
    models[i][j] = model
    print ('For rank %s, regularization parameter %s the RMSE is %s' % (rank, regParam, error))
    if error < min_error:
      min_error = error
      best_params = [i,j]
    j += 1
  i += 1

als_2more.setRegParam(regParams[best_params[0]])
als_2more.setRank(ranks[best_params[1]])
print ('The best model was trained with regularization parameter %s' % regParams[best_params[0]])
print ('The best model was trained with rank %s' % ranks[best_params[1]])
my_model_2more = models[best_params[0]][best_params[1]]

#Testing the Model on the Test_2more Dataset
predict_2more = my_model_2more.transform(test_2more)

# Remove NaN values from prediction (due to SPARK-14489)
predicted_test_2more = predict_2more.filter(predict_2more.prediction != float('nan'))

# Round floats to whole numbers
predicted_test_2more = predicted_test_2more.withColumn("prediction", F.abs(F.round(predicted_test_2more["prediction"],0)))
# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
test2more_RMSE = reg_eval.evaluate(predicted_test_2more)

print('The model had a RMSE on the test set of {0}'.format(test2more_RMSE))

#Comparing the Model
##We again compare to selecting the average number of plays from the training dataset
avg_plays_2more = trainset_2more.groupBy().avg('Plays').select(F.round('avg(Plays)'))

avg_plays_2more.show(3)
# Extract the average rating value. (This is row 0, column 0.)
train_avg_plays2more = avg_plays_2more.collect()[0][0]

print('The average number of plays in the dataset is {0}'.format(train_avg_plays2more))

# Add a column with the average rating
test_for_avg_2more = test_2more.withColumn('prediction', F.lit(train_avg_plays2more))

# Run the previously created RMSE evaluator, reg_eval, on the test_for_avg_df DataFrame
test_avg_RMSE_2more = reg_eval.evaluate(test_for_avg_2more)

print("The RMSE on the average set is {0}".format(test_avg_RMSE_2more))

#PREDICTION FOR THE USER - 02
UserID = 13
songs_listened = plays_df_2.filter(plays_df_2.user_als_id == UserID) \
    .join(metadata_df, 'songId') \
    .select('song_als_id', 'artist_name', 'title') \
 \
# Generating List of Listened Songs
listened_songs_list = []
for song in songs_listened.collect():
    listened_songs_list.append(song['song_als_id'])

print('Songs user has listened to:')
songs_listened.select('artist_name', 'title').show()

# generate dataframe of unlistened songs
songs_unlistened = plays_df_2.filter( ~ plays_df_2['song_als_id'].isin(listened_songs_list)) \
    .select('song_als_id').withColumn('user_als_id', F.lit(UserID)).distinct()

# feed unlistened songs into model
predicted_listens = my_model_2more.transform(songs_unlistened)

# remove NaNs
predicted_listens = predicted_listens.filter(predicted_listens['prediction'] != float('nan'))

# print output
print('Predicted Songs:')
predicted_listens.join(plays_df_2, 'song_als_id') \
    .join(metadata_df, 'songId') \
    .select('artist_name', 'title', 'prediction') \
    .distinct() \
    .orderBy('prediction', ascending=False) \
    .show(10)
