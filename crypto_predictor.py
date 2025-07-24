import requests
import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
import ta

def fetch_crypto_data():
    sources = {
        "CoinGecko": "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false",
        "CoinCap": "https://api.coincap.io/v2/assets?limit=10"
    }

    all_data = []
    for source, url in sources.items():
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()

            if source == "CoinGecko":
                all_data.extend(data)
            elif source == "CoinCap":
                # Normalize CoinCap data to match CoinGecko's format
                if isinstance(data, dict) and "data" in data:
                    for asset in data["data"]:
                        all_data.append({
                            "id": asset.get("id"),
                            "symbol": asset.get("symbol", "").lower(),
                            "name": asset.get("name"),
                            "current_price": float(asset.get("priceUsd", 0)),
                            "market_cap": float(asset.get("marketCapUsd", 0)),
                            "total_volume": float(asset.get("volumeUsd24Hr", 0)),
                            "price_change_percentage_24h": float(asset.get("changePercent24Hr", 0))
                        })
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch data from {source}: {e}")

    return all_data

def calculate_indicators(df):
    # Calculate Simple Moving Averages (SMA)
    df['sma_7'] = df['current_price'].rolling(window=7).mean()
    df['sma_30'] = df['current_price'].rolling(window=30).mean()

    # Calculate Relative Strength Index (RSI)
    df['rsi'] = ta.momentum.RSIIndicator(df['current_price'], window=14).rsi()

    # Calculate Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(df['current_price'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    return df

def calculate_risk_metrics(df):
    # Calculate daily returns
    df['daily_return'] = df['current_price'].pct_change()

    # Calculate volatility (standard deviation of daily returns)
    df['volatility'] = df['daily_return'].rolling(window=30).std()

    # Calculate Sharpe ratio (assuming risk-free rate is 0 for simplicity)
    df['sharpe_ratio'] = df['daily_return'].rolling(window=30).mean() / df['volatility']

    return df

def analyze_data(data):
    df = pd.DataFrame(data)
    df['price_change_percentage_24h'] = df['price_change_percentage_24h'].astype(float)
    df['market_cap'] = df['market_cap'].astype(float)
    df['total_volume'] = df['total_volume'].astype(float)
    df['current_price'] = df['current_price'].astype(float)

    # Calculate technical indicators
    df = calculate_indicators(df)

    # Calculate risk metrics
    df = calculate_risk_metrics(df)

    # Filter criteria: highest 24h change, high market cap, high volume, price above SMA, RSI below 70, MACD above signal, high Sharpe ratio
    filtered_df = df[(df['price_change_percentage_24h'] > 0) &
                     (df['market_cap'] > df['market_cap'].mean()) &
                     (df['total_volume'] > df['total_volume'].mean()) &
                     (df['current_price'] > df['sma_7']) &
                     (df['rsi'] < 70) &
                     (df['macd'] > df['macd_signal']) &
                     (df['sharpe_ratio'] > df['sharpe_ratio'].mean())]

    if not filtered_df.empty:
        best_crypto = filtered_df.loc[filtered_df['price_change_percentage_24h'].idxmax()]
    else:
        best_crypto = df.loc[df['price_change_percentage_24h'].idxmax()]

    return best_crypto

def get_user_portfolio():
    portfolio = {}
    while True:
        crypto_name = input("Enter the name of a cryptocurrency you own (or 'done' to finish): ")
        if crypto_name.lower() == 'done':
            break

        try:
            amount = float(input(f"Enter the amount of {crypto_name} you own: "))
            portfolio[crypto_name] = amount
        except ValueError:
            print("Invalid amount. Please enter a number.")

    return portfolio

def assess_portfolio(portfolio, all_crypto_data):
    if not portfolio:
        return

    portfolio_value = 0
    portfolio_df = pd.DataFrame(list(portfolio.items()), columns=['name', 'amount'])

    for index, row in portfolio_df.iterrows():
        crypto_name = row['name']
        amount = row['amount']

        # Find the crypto in the fetched data
        crypto_data = next((item for item in all_crypto_data if item['name'].lower() == crypto_name.lower()), None)

        if crypto_data:
            current_price = crypto_data['current_price']
            value = amount * current_price
            portfolio_value += value
            print(f"\nAssessing {crypto_name}:")
            print(f"  Amount: {amount}")
            print(f"  Current Price: ${current_price:,.2f}")
            print(f"  Value: ${value:,.2f}")
        else:
            print(f"\nCould not find data for {crypto_name}.")

    print(f"\nTotal Portfolio Value: ${portfolio_value:,.2f}")

def ai_bot_advisor(portfolio, all_crypto_data):
    print("\n🤖 AI Bot Advisor:")

    if not portfolio:
        print("Your portfolio is empty. Here's a general tip:")
        print("- Diversify your investments across different types of cryptocurrencies.")
        return

    # Simple advice based on portfolio
    if len(portfolio) == 1:
        print("- Your portfolio is not diversified. Consider adding other cryptocurrencies to spread risk.")

    # Advice based on market data
    best_crypto = analyze_data(all_crypto_data)
    print(f"- The market analysis suggests that {best_crypto['name']} is a strong candidate to buy right now.")

    # Cross-reference with user's portfolio
    if best_crypto['name'] in portfolio:
        print(f"- You already own {best_crypto['name']}, which is great! Consider if you want to increase your position.")
    else:
        print(f"- You do not own {best_crypto['name']}. You might want to research it as a potential addition to your portfolio.")

def create_test_portfolio(all_crypto_data):
    print("\nCreating a test portfolio with random cryptocurrencies and fake funds...")
    test_portfolio = {}

    # Ensure we have data to create a portfolio from
    if not all_crypto_data:
        print("No crypto data available to create a test portfolio.")
        return test_portfolio

    # Select 3 random cryptocurrencies from the fetched data
    num_to_select = min(3, len(all_crypto_data))
    random_cryptos = np.random.choice(all_crypto_data, num_to_select, replace=False)

    for crypto in random_cryptos:
        # Assign a random amount between 1 and 1000
        amount = np.random.uniform(1, 1000)
        test_portfolio[crypto['name']] = amount

    print("Test portfolio created successfully.")
    return test_portfolio

def test_accuracy_of_picks(past_data, current_data):
    print("\nTesting accuracy of AI bot's future picks...")

    if not past_data or not current_data:
        print("Insufficient data to test accuracy.")
        return

    # Simulate a pick from the past
    past_pick = analyze_data(past_data)

    # Find the same cryptocurrency in the current data
    current_pick_data = next((item for item in current_data if item['id'] == past_pick['id']), None)

    if not current_pick_data:
        print(f"Could not find {past_pick['name']} in the current data to verify its performance.")
        return

    # Compare the price change
    past_price = past_pick['current_price']
    current_price = current_pick_data['current_price']
    price_change = ((current_price - past_price) / past_price) * 100

    print(f"AI Bot's pick from the past: {past_pick['name']}")
    print(f"Past Price: ${past_price:,.2f}")
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Performance: {price_change:+.2f}%")

    if price_change > 0:
        print("Conclusion: The pick was a winner!")
    else:
        print("Conclusion: The pick did not perform well.")

def strategy_section(portfolio, all_crypto_data):
    print("\n💡 Strategy Section to Maximize Earnings:")

    if not portfolio:
        print("- Start by building a diversified portfolio. A good starting point is to invest in a mix of large-cap and mid-cap cryptocurrencies.")
        return

    # Strategy 1: Rebalancing
    print("\n1. Portfolio Rebalancing:")
    print("- Periodically review your portfolio to ensure it aligns with your risk tolerance. If one asset has grown significantly, consider taking some profits and reallocating to other assets to maintain diversification.")

    # Strategy 2: Dollar-Cost Averaging (DCA)
    print("\n2. Dollar-Cost Averaging (DCA):")
    print("- Instead of investing a lump sum, consider investing smaller amounts regularly over time. This can help reduce the impact of volatility.")

    # Strategy 3: Focus on High-Quality Assets
    best_crypto = analyze_data(all_crypto_data)
    print("\n3. Focus on High-Quality Assets:")
    print(f"- Our analysis indicates that {best_crypto['name']} is a strong asset. Continuously look for assets with strong fundamentals, high market cap, and good trading volume.")

def main():
    # Ask user if they want to use a test portfolio
    use_test_portfolio = input("Do you want to use a test portfolio? (yes/no): ").lower()

    data = fetch_crypto_data()
    if not data:
        print("Could not fetch cryptocurrency data. Exiting.")
        return

    if use_test_portfolio == 'yes':
        portfolio = create_test_portfolio(data)
        # For demonstration, we'll use the same data for past and current
        # In a real scenario, you'd fetch data at different times
        test_accuracy_of_picks(data, data)
    else:
        portfolio = get_user_portfolio()

    assess_portfolio(portfolio, data)
    ai_bot_advisor(portfolio, data)
    strategy_section(portfolio, data)

    best_crypto = analyze_data(data)

    print("\nBest cryptocurrency to buy based on enhanced criteria and risk assessment:")
    print(f"Name: {best_crypto['name']}")
    print(f"Symbol: {best_crypto['symbol']}")
    print(f"Current Price: ${best_crypto['current_price']:.2f}")
    print(f"24h Change: {best_crypto['price_change_percentage_24h']:.2f}%")
    print(f"Market Cap: ${best_crypto['market_cap']:,.2f}")
    print(f"Volume: ${best_crypto['total_volume']:,.2f}")

    # The following will only be printed if the values are not NaN
    if pd.notna(best_crypto['sma_7']):
        print(f"SMA 7: ${best_crypto['sma_7']:.2f}")
    if pd.notna(best_crypto['sma_30']):
        print(f"SMA 30: ${best_crypto['sma_30']:.2f}")
    if pd.notna(best_crypto['rsi']):
        print(f"RSI: {best_crypto['rsi']:.2f}")
    if pd.notna(best_crypto['macd']):
        print(f"MACD: {best_crypto['macd']:.2f}")
    if pd.notna(best_crypto['macd_signal']):
        print(f"MACD Signal: {best_crypto['macd_signal']:.2f}")
    if pd.notna(best_crypto['volatility']):
        print(f"Volatility: {best_crypto['volatility']:.4f}")
    if pd.notna(best_crypto['sharpe_ratio']):
        print(f"Sharpe Ratio: {best_crypto['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    main()
