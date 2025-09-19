import os
import asyncio
import pandas as pd
import dateutil.parser
import json
import yfinance as yf
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

async def live_search_batch_dates(api_key, ticker: str, dates: list[datetime]):
    """
    Batch analysis for multiple dates in a single API request
    More cost-effective than individual requests
    """
    client = AsyncClient(api_key=api_key)

    # Format dates for batch prompt
    date_list = [d.strftime("%Y-%m-%d") for d in dates]
    date_range_str = f"from {date_list[0]} to {date_list[-1]}"

    batch_prompt = f"""
    Analyze sentiment for {ticker} across multiple dates: {', '.join(date_list)}.

    For each date, search relevant news, web, and social platforms for major developments affecting the stock.

    Return a valid JSON array with one object for each date, in this exact format:
    [
      {{
        "date": "YYYY-MM-DD",
        "ticker": "{ticker}",
        "summary": "Clear, concise synthesis of market events and sentiment drivers for this specific date",
        "key_factors": [
          {{"source_type": "news|web|social", "link": "...", "description": "..."}},
          {{"source_type": "news|web|social", "link": "...", "description": "..."}}
        ],
        "sentiment_label": "bullish|bearish|neutral",
        "sentiment_score": float (from -1 to 1),
        "confidence": float (from 0 to 1)
      }}
    ]

    IMPORTANT:
    - Return valid JSON only, no comments or extra text
    - Include an entry for each date even if no significant events found
    - Use neutral sentiment (score: 0.0) for dates with no major developments
    """

    # Use the earliest and latest dates for search window
    earliest_date = min(dates)
    latest_date = max(dates)

    chat = client.chat.create(
        model="grok-3-mini",
        search_parameters=SearchParameters(
            mode="on",  # Force search instead of auto
            from_date=earliest_date,  # Wider window
            to_date=latest_date,
            sources=[x_source()],
        ),
    )

    chat.append(system(system_prompt))
    chat.append(user(batch_prompt))

    response = await chat.sample()
    sentiment_data = parse_ai_sentiment_response(response.content)
    df = pd.DataFrame(sentiment_data)
    print(f"Grok Batch Response Content:\n{response.content}")
    return df

async def live_search_with_web_and_news(api_key, ticker: str, date: datetime):
    client = AsyncClient(api_key=api_key)

    user_prompt = user_prompt_template.format(
        ticker=ticker,
        date=date.strftime("%Y-%m-%d"),
    )
    chat = client.chat.create(
        model="grok-3-mini-fast",
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
    start_date = end_date - timedelta(days=min_days + 1)

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
                        # print(f"Entry signal set for {price_date}")
                    if row['exit_signal']:
                        exits.loc[price_date] = True
                        # print(f"Exit signal set for {price_date}")
                    break

    return price_clean, entries, exits

if __name__ == "__main__":
    # Fetch historical price data first
    price_df = fetch_historical_data(ticker, date)

    # Extract dates from price_df and get AI analysis using BATCH processing
    all_sentiment_data = []

    def chunk_dates(dates, chunk_size=3):
        """Split dates into chunks for batch processing"""
        for i in range(0, len(dates), chunk_size):
            yield dates[i:i + chunk_size]

    async def analyze_batch_dates(ticker, dates, batch_size=3):
        """Analyze dates in batches for cost optimization"""
        results = []

        # Convert dates to datetime with timezone
        analysis_dates = [date.to_pydatetime().replace(tzinfo=timezone.utc) for date in dates]

        # Process in batches
        date_chunks = list(chunk_dates(analysis_dates, batch_size))

        for i, chunk in enumerate(date_chunks):
            try:
                print(f"Batch {i+1}/{len(date_chunks)}: Analyzing {len(chunk)} dates from {chunk[0].strftime('%Y-%m-%d')} to {chunk[-1].strftime('%Y-%m-%d')}")

                # Use batch analysis for the chunk
                df_batch = await live_search_batch_dates(api_key, ticker, chunk)

                if not df_batch.empty:
                    sentiment_batch = calc_sentiment_entries_exits(df_batch)
                    results.append(sentiment_batch)

            except Exception as e:
                print(f"Error in batch {i+1}: {e}")
                # Fallback to individual analysis for this batch
                print(f"Falling back to individual analysis for batch {i+1}")
                for date in chunk:
                    try:
                        df_single = await live_search_with_web_and_news(api_key, ticker, date)
                        if not df_single.empty:
                            sentiment_single = calc_sentiment_entries_exits(df_single)
                            results.append(sentiment_single)
                    except Exception as individual_error:
                        print(f"Individual analysis failed for {date.strftime('%Y-%m-%d')}: {individual_error}")

        return results

    # Run batch analysis (3 dates per batch = ~66% cost reduction)
    print(f"Starting BATCH analysis for {len(price_df)} dates (3 dates per batch)...")
    all_sentiment_data = asyncio.run(analyze_batch_dates(ticker, price_df['Date'], batch_size=3))

    # print(f"Completed batch analysis. Got {len(all_sentiment_data)} sentiment batches.")
    print(f"All Sentiment Data:\n{all_sentiment_data}")

    # Combine all sentiment data
    if all_sentiment_data:
        # Combine all individual sentiment DataFrames
        combined_sentiment_df = pd.concat(all_sentiment_data, ignore_index=True)
        # print(f"Combined sentiment data:\n{combined_sentiment_df[['date', 'sentiment_score', 'entry_signal', 'exit_signal']]}")

        # Prepare data for backtesting using the COMBINED data
        clean_price_df, entries, exits = prepare_data_for_backtest(combined_sentiment_df, price_df)

        portfolio = vbt.Portfolio.from_signals(
            close=clean_price_df['Close'],
            entries=entries,
            exits=exits,
            init_cash=100000,
            freq='1d',
        )

        # print(clean_price_df['Close'])
        # print(portfolio.stats())
        print(portfolio.trades.records_readable)

        # Create enhanced Plotly chart with dual y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Get portfolio equity curve
        equity = portfolio.value()
        
        # Add strategy equity curve (primary y-axis)
        if equity is not None and not equity.empty:
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    name='Strategy Equity',
                    mode='lines',
                    line=dict(color='#ffffff', width=2),
                    text=[f"${v:,.2f}<br>{d.strftime('%Y-%m-%d')}" for v, d in zip(equity.values, equity.index)],
                    hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>"
                ),
                secondary_y=False
            )

        # Add close price curve (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=clean_price_df.index,
                y=clean_price_df['Close'],
                name='Close Price',
                mode='lines',
                line=dict(color='#84cc16', width=2),
                text=[f"${v:.2f}<br>{d.strftime('%Y-%m-%d')}" for v, d in zip(clean_price_df['Close'], clean_price_df.index)],
                hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>"
            ),
            secondary_y=True
        )

        # Get trade records and add entry/exit signals
        trades_df = portfolio.trades.records_readable
        
        if not trades_df.empty:
            shown_entry = False
            shown_profit = False
            shown_loss = False
            
            for i in range(len(trades_df)):
                trade = trades_df.iloc[i]
                entry_time = pd.to_datetime(trade['Entry Timestamp'])
                entry_price = trade['Avg Entry Price']
                
                # Entry marker (on price axis)
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time],
                        y=[entry_price],
                        name='Entry',
                        mode='markers',
                        marker=dict(
                            symbol='square',
                            size=8,
                            color='#3b82f6',
                            line=dict(width=1, color='#3b82f6')
                        ),
                        text=[f"${entry_price:,.2f}<br>{entry_time.strftime('%Y-%m-%d')}"],
                        hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>",
                        showlegend=not shown_entry,
                        legendgroup='Entry'
                    ),
                    secondary_y=True
                )
                shown_entry = True
                
                # Exit marker (treat all trades the same - use exit data directly)
                if pd.notna(trade['Exit Timestamp']):
                    exit_time = pd.to_datetime(trade['Exit Timestamp'])
                    exit_price = trade['Avg Exit Price']
                    pnl = trade['PnL']
                    is_profit = exit_price > entry_price
                    
                    exit_type = 'Exit - Profit' if is_profit else 'Exit - Loss'
                    show_exit_legend = (is_profit and not shown_profit) or (not is_profit and not shown_loss)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[exit_time],
                            y=[exit_price],
                            name=exit_type,
                            mode='markers',
                            marker=dict(
                                symbol='square',
                                size=8,
                                color="#22c55e" if is_profit else '#ef4444',
                                line=dict(width=1, color="#22c55e" if is_profit else '#ef4444')
                            ),
                            text=[f"${exit_price:,.2f}<br>{exit_time.strftime('%Y-%m-%d')}<br>PnL: ${pnl:,.2f}"],
                            hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>",
                            showlegend=show_exit_legend,
                            legendgroup=exit_type
                        ),
                        secondary_y=True
                    )
                    
                    if is_profit:
                        shown_profit = True
                    else:
                        shown_loss = True
                    
                    # Add shaded trade area (green for profit, red for loss only)
                    fig.add_shape(
                        type="rect",
                        x0=entry_time,
                        x1=exit_time,
                        yref="paper",
                        y0=0,
                        y1=1,
                        fillcolor='#22c55e' if is_profit else '#ef4444',
                        opacity=0.2,
                        line=dict(width=0),
                        layer="below"
                    )

        # Update layout with dark theme and minimal grid lines
        fig.update_layout(
            title=f'{ticker} - Sentiment Strategy Performance',
            xaxis_title='Date',
            height=700,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='#ffffff'),
            # Reduce grid line opacity and customize
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',  # Minimal opacity for vertical grid
                gridwidth=1
            )
        )
        
        # Set y-axes titles and grid settings
        fig.update_yaxes(
            title_text="Portfolio Value ($)", 
            secondary_y=False, 
            color='#ffffff',
            showgrid=False  # Remove horizontal grid lines from equity axis
        )
        fig.update_yaxes(
            title_text="Stock Price ($)", 
            secondary_y=True, 
            color='#84cc16',
            gridcolor='rgba(132,204,22,0.1)',  # Minimal opacity horizontal grid from price axis
            gridwidth=1
        )

        fig.show()

    else:
        print("No sentiment data found for any dates")