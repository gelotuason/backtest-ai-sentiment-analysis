import os
import asyncio
import pandas as pd
import dateutil.parser
import json
import yfinance as yf
import vectorbt as vbt
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from xai_sdk import AsyncClient
from xai_sdk.chat import user, system
from xai_sdk.search import SearchParameters, news_source, rss_source, web_source, x_source

load_dotenv(dotenv_path=".env")

api_key = os.getenv("XAI_API_KEY")

date = datetime(2025, 9, 19, tzinfo=timezone.utc)
ticker = "TSLA"

system_prompt = """
You are a financial news sentiment analysis assistant integrated with algorithmic trading systems.
Your role is to analyze provided news articles, headlines, and social media posts for a specified ticker and time window, and return a structured sentiment score and label.
Only use information available up to the provided date; never use any data or context not present in the input.

Output must be in standardized JSON format with the following fields.

Example output (JSON only):
[
  {
    "date": "2024-01-15",
    "ticker": "TSLA",
    "summary": "Tesla shares rose after the company's strong earnings...",
    "key_factors": [
      {
        "source_type": "web",
        "link": "https://**",
        "description": "Tesla earnings surpass expectations in Q4 report"
      },
      {
        "source_type": "social",
        "link": "https://**",
        "description": "Elon Musk tweets: 'Great quarter for Tesla!'"
      },
      {
        "source_type": "news",
        "link": "https://**",
        "description": "Tesla expands its market share in Europe"
      }
    ],
    "sentiment_label": "bullish",
    "sentiment_score": 0.78,
    "confidence": 0.92
  },
  ...
]

Be objective and consistent, and always provide clear justification for each sentiment judgement.
"""

user_prompt_template = """
Analyze sentiment for {ticker} on {date}.
Search relevant news, web, and social platforms for major developments affecting the stock on this day.
List the main drivers with their types and links, and output sentiment analysis results in this JSON format:
{{
  "date": {date},
  "ticker": {ticker},
  "summary": "Clear, concise synthesis of market events and sentiment drivers",
  "key_factors": [
    {{"source_type": "news|web|social", "link": "...", "description": "..."}},
    {{"source_type": "news|web|social", "link": "...", "description": "..."}}
  ],
  "sentiment_label": "bullish|bearish|neutral",
  "sentiment_score": float (from -1 to 1),
  "confidence": float (from 0 to 1)
}}
"""

async def live_search_with_web_and_news(api_key, ticker: str, date: datetime):
    client = AsyncClient(api_key=api_key)

    user_prompt = user_prompt_template.format(
        ticker=ticker,
        date=date.strftime("%Y-%m-%d"),
    )
    chat = client.chat.create(
        model="grok-3-mini",
        search_parameters=SearchParameters(
            mode="auto",
            from_date=date - timedelta(days=2),
            to_date=date,
            # sources=[web_source(), news_source(), x_source()],
            sources=[x_source()],
        ),
    )

    chat.append(system(system_prompt))
    chat.append(user(user_prompt))

    response = await chat.sample()
    sentiment_data = parse_ai_sentiment_response(response.content)
    df = pd.DataFrame(sentiment_data)
    print(f"Grok Response Content:\n{response.content}")
    return df

    print(f"Citations: {response.citations}")
    print(f"Unique search sources: {response.usage.num_sources_used}")

import pandas as pd

def calc_sentiment_entries_exits(
    df: pd.DataFrame,
    entry_threshold: float = 0.6,
    exit_threshold: float = -0.6,
    sentiment_col: str = "sentiment_score"
) -> pd.DataFrame:
    """
    Adds entry and exit signal columns to the DataFrame based on sentiment thresholds.

    Parameters:
        df: DataFrame with at least a 'sentiment_score' column (or custom)
        entry_threshold: float, sentiment score threshold for entry (buy/long)
        exit_threshold: float, sentiment score threshold for exit (sell/close)
        sentiment_col: str, column name to use for scoring

    Returns:
        df: DataFrame with added 'entry_signal' and 'exit_signal' columns (True/False).
    """
    # Signal: True where sentiment crosses above entry_threshold (and wasn't already True)
    df = df.copy()
    entry = df[sentiment_col] > entry_threshold
    exit = df[sentiment_col] < exit_threshold

    # Only trigger a new entry when not previously in entry (avoids stacked entries in uptrend)
    df["entry_signal"] = entry & (~entry.shift(1, fill_value=False))
    df["exit_signal"] = exit & (~exit.shift(1, fill_value=False))
    return df

def parse_ai_sentiment_response(raw_response):
    """
    Transforms the AI JSON response into a DataFrame-ready dict/list.
    - Ensures field names, types, and structures are consistent
    - Handles date/time, float conversions, missing fields
    """

    result = []
    # If response is a string, parse as json, otherwise treat as dict/list directly
    data = raw_response
    if isinstance(raw_response, str):
        data = json.loads(raw_response)
    if isinstance(data, dict):  # Single entry
        data = [data]

    for entry in data:
        # Normalize and assign defaults as needed
        result.append({
            "date": entry.get("date") or entry.get("timestamp"),
            "ticker": entry.get("ticker", "").upper(),
            "summary": entry.get("summary") or "",
            "key_factors": entry.get("key_factors", []),
            "sentiment_label": entry.get("sentiment_label", "").lower(),
            "sentiment_score": float(entry.get("sentiment_score", 0)),
            "confidence": float(entry.get("confidence", 0))
        })
    return result

def fetch_historical_data(ticker: str, end_date: datetime, min_days: int = 0) -> pd.DataFrame:
    """
    Fetch historical market price data using yfinance library.

    Parameters:
        ticker: str, stock ticker symbol (e.g., 'AAPL', 'TSLA')
        end_date: datetime, end date for historical data
        min_days: int, minimum number of days to fetch (default: 30 for one month)

    Returns:
        pd.DataFrame: Historical price data with columns: Open, High, Low, Close, Volume, Adj Close
    """
    # Ensure minimum window of at least one month
    # if min_days < 30:
    #     min_days = 30

    # Calculate start date (add a few extra days to ensure we get enough trading days)
    start_date = end_date - timedelta(days=min_days + 5)

    # Download historical data
    stock = yf.Ticker(ticker)
    hist_data = stock.history(
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d')
    )

    if hist_data.empty:
        raise ValueError(f"No data found for ticker {ticker} in the specified date range")

    # Reset index to make Date a column
    hist_data.reset_index(inplace=True)

    # Ensure we have at least the minimum number of trading days
    if len(hist_data) < min_days * 0.7:  # Account for weekends/holidays (roughly 70% trading days)
        # If not enough data, extend the start date
        extended_start = end_date - timedelta(days=min_days * 3)
        hist_data = stock.history(
            start=extended_start.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        hist_data.reset_index(inplace=True)

    return hist_data

def calculate_portfolio(df):

    pass

def prepare_data_for_backtest(sentiment_df, price_df):
    """
    Merges sentiment data with historical price data for backtesting.

    Parameters:
        sentiment_df: pd.DataFrame with sentiment analysis results
        price_df: pd.DataFrame with historical price data

    Returns:
        tuple: (price_clean, entries, exits) - price data, entry signals, and exit signals
    """
    # Transform price_df to have Date as index and only Close column
    price_clean = price_df[['Date', 'Close']].copy()
    price_clean.set_index('Date', inplace=True)

    # Create entries and exits as boolean series with all dates from price_df, False by default
    entries = pd.Series(False, index=price_clean.index, name='entry_signal')
    exits = pd.Series(False, index=price_clean.index, name='exit_signal')
    entries.index.name = 'Date'
    exits.index.name = 'Date'

    # If sentiment_df is provided, update entries and exits for matching dates
    if sentiment_df is not None:
        # Convert sentiment dates to datetime and create temporary index
        sentiment_temp = sentiment_df[['date', 'entry_signal', 'exit_signal']].copy()
        sentiment_temp['date'] = pd.to_datetime(sentiment_temp['date'])
        sentiment_temp.set_index('date', inplace=True)

        # Update entries and exits where dates match and signals are True
        for sentiment_date, row in sentiment_temp.iterrows():
            for price_date in entries.index:
                if sentiment_date.date() == price_date.date():
                    if row['entry_signal']:
                        entries.loc[price_date] = True
                        print(f"Entry signal set for {price_date}")
                    if row['exit_signal']:
                        exits.loc[price_date] = True
                        print(f"Exit signal set for {price_date}")
                    break

    return price_clean, entries, exits

if __name__ == "__main__":
    # Fetch historical price data first
    price_df = fetch_historical_data(ticker, date)

    # Extract dates from price_df and get AI analysis for each date using concurrent requests
    all_sentiment_data = []

    async def analyze_single_date(ticker, analysis_date):
        """Analyze sentiment for a single date"""
        try:
            print(f"Analyzing sentiment for {ticker} on {analysis_date.strftime('%Y-%m-%d')}...")
            df_single = await live_search_with_web_and_news(api_key, ticker, analysis_date)

            if not df_single.empty:
                sentiment_single = calc_sentiment_entries_exits(df_single)
                return sentiment_single
            return None
        except Exception as e:
            print(f"Error analyzing {analysis_date.strftime('%Y-%m-%d')}: {e}")
            return None

    async def analyze_concurrent_dates(ticker, dates, max_concurrent=3):
        """Analyze multiple dates concurrently with limited concurrency"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_analyze(date):
            async with semaphore:
                return await analyze_single_date(ticker, date)

        # Convert dates to datetime with timezone
        analysis_dates = [date.to_pydatetime().replace(tzinfo=timezone.utc) for date in dates]

        # Run analyses concurrently with max_concurrent limit
        tasks = [bounded_analyze(date) for date in analysis_dates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        valid_results = []
        for result in results:
            if result is not None and not isinstance(result, Exception):
                valid_results.append(result)
            elif isinstance(result, Exception):
                print(f"Exception in concurrent analysis: {result}")

        return valid_results

    # Run concurrent analysis
    print(f"Starting concurrent analysis for {len(price_df)} dates with max 3 concurrent requests...")
    all_sentiment_data = asyncio.run(analyze_concurrent_dates(ticker, price_df['Date'], max_concurrent=3))

    print(f"Completed analysis. Got {len(all_sentiment_data)} valid sentiment analyses.")

    # Combine all sentiment data
    if all_sentiment_data:
        # Combine all individual sentiment DataFrames
        combined_sentiment_df = pd.concat(all_sentiment_data, ignore_index=True)
        print(f"Combined sentiment data:\n{combined_sentiment_df[['date', 'sentiment_score', 'entry_signal', 'exit_signal']]}")

        # Prepare data for backtesting using the COMBINED data
        clean_price_df, entries, exits = prepare_data_for_backtest(combined_sentiment_df, price_df)

        portfolio = vbt.Portfolio.from_signals(
            close=clean_price_df['Close'],
            entries=entries,
            exits=exits,
            init_cash=100000,
            freq='1d',
        )

        print(portfolio.stats())
        print(portfolio.trades.records_readable)
        portfolio.plot().show()
    else:
        print("No sentiment data found for any dates")