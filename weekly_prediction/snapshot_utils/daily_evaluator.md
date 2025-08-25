lets refactor the daily evaluator. 
1. it should read the latest snashot file if file is not provided
2. refresh the full data set from table if refresh_prices=true
3. --date take the eval_date for which file needs to be generated, if date is empty use the current data
3. create the data frame for the stocks which are in snapshot and only take symbol name from this dataframe
4. now combine snapshot data frame to price dataframe using inner join and where date >= eval_date
5. pass this dataframe as input to feature_engineering.py which should return a dataframe with features calculated. 
6. save that file as data_eval_input.csv in sanshot.

As program is not generating the expected output, I want to do the things in steps. 
comment the existing code, so that one above is working we will try to complete the reaming steps. 
