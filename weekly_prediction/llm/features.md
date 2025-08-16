## Features
using langchain these are the features I want to develop. 
1. identify earning is in next 3 weeks. 
2. in last 2 quarters stock has done well . 
3. in last 3 weeks stocks has continus showed a upward movement . 
4. I want is_upward_trending to be more complex where it crosses the historical resistance level
6.I want to create a one more feature , when a good quater result anounced price goes up and after some day it comes down  and then again goes up. during this time stock has a bullish moment. I want to caputure this bullish moment as a another feature.

## Model
1. I should able to predict in next one week these stock will reach at price x

Note : 
## The Problem: Using Price to Predict Price
The culprit is the price feature itself.

Right now, we are asking the model: "Given today's price of $100 (and other features), what will the price be in 5 days?"

The single best predictor of the price in 5 days is... the price today. Stock prices have extremely high auto-correlation. The model is learning a very simple and not very useful rule: future_price ≈ current_price. This is why the R² is so high—the model's predictions are almost perfectly correlated with the actual values, but it's not learning from our complex features. The $4.12 MAE likely represents the average 5-day price drift that the model is learning, but the R² is dominated by the trivial price-is-close-to-price relationship.

## The Solution: Predict Price Change, Not Price Level
To make our model truly useful, we need to ask it a harder, more meaningful question. Instead of predicting the future price level, let's make it predict the future price change.

We will change our target (y) from price_in_5_days to price_change_in_5_days.

This forces the model to ignore the easy answer ("the price will be about the same") and instead focus on all the other features we built (broke_resistance, rsi_14d, post_earnings_dip_rally, etc.) to predict how much the price will move.

## Goal: Find the "20 Best Stocks"
The goal is to create a module, stock_selector.py, that uses our model's predictions and our engineered features to identify the most promising stocks for the next week.

Here’s our strategy for ranking them:

Load the most recent set of features for all our stocks.

Use our trained model to predict the 5-day price change for each one.

Calculate a predicted return percentage.

Apply a "confidence filter" – we'll focus only on stocks that are showing strong bullish signals (broke_resistance is True OR post_earnings_dip_rally is True).

Rank these high-confidence stocks by their predicted return percentage.

Select the top 20 from this ranked list.

##UI

5. I want to develop a UI using stream lit to show 20 best stocks for this week, 
6. I should have chatbot in UI that allows me to ask questions

can you help to generate the code for me step by step. code must be as modular as possible which allows me to enhance its feature separately without impact any existing functionality

