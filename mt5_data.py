import os
import asyncio
import pandas as pd
from metaapi_cloud_sdk import MetaApi

# Retrieve tokens from environment variables (set in Render Dashboard)
META_API_TOKEN = os.getenv("META_API_TOKEN", "YOUR_META_API_TOKEN_HERE")
ACCOUNT_ID = os.getenv("META_ACCOUNT_ID", "YOUR_META_ACCOUNT_ID_HERE")


async def get_rpc_connection():
    """Initializes MetaApi lazily inside an active event loop and connects."""
    api = MetaApi(META_API_TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()
    return connection


def fetch_rates(symbol: str, timeframe: str = "15m", count: int = 300) -> pd.DataFrame:
    """
    Synchronous wrapper to fetch rates for strategy analysis, 
    compatible with existing engine pipelines.
    """
    try:
        return asyncio.run(_async_fetch_rates(symbol, timeframe, count))
    except Exception as e:
        print(f"Error running fetch_rates: {e}")
        return None


async def _async_fetch_rates(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Fetches OHLCV rate data asynchronously via MetaApi RPC connection."""
    try:
        connection = await get_rpc_connection()
        candles = await connection.get_historical_candles(symbol, timeframe, None, count)
        
        if not candles:
            print(f"No candles returned for {symbol}")
            return None

        df = pd.DataFrame(candles)

        # Standardize column names to match structure_engine & strategy_engine
        df.rename(columns={
            'time': 'time',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tickVolume': 'Volume'
        }, inplace=True)

        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

        return df
    except Exception as e:
        print(f"MetaApi fetch rates error for {symbol}: {e}")
        return None


async def execute_trade(symbol: str, action: str, volume: float, sl: float = None, tp: float = None):
    """
    Executes trades directly on MT5 via MetaApi after setup confirmation.
    
    action: "BUY", "SELL", "BUY_STOP", "SELL_STOP"
    """
    try:
        connection = await get_rpc_connection()
        action_upper = action.upper().strip()

        if action_upper == "BUY":
            result = await connection.create_market_buy_order(
                symbol=symbol, volume=volume, stop_loss=sl, take_profit=tp
            )
        elif action_upper == "SELL":
            result = await connection.create_market_sell_order(
                symbol=symbol, volume=volume, stop_loss=sl, take_profit=tp
            )
        elif action_upper == "BUY_STOP":
            result = await connection.create_limit_buy_order(
                symbol=symbol, volume=volume, open_price=sl, stop_loss=sl, take_profit=tp
            )
        elif action_upper == "SELL_STOP":
            result = await connection.create_limit_sell_order(
                symbol=symbol, volume=volume, open_price=sl, stop_loss=sl, take_profit=tp
            )
        else:
            print(f"Invalid trade action: {action}")
            return None

        print(f"✅ Trade successfully sent to MT5: {result}")
        return result

    except Exception as e:
        print(f"❌ Failed to execute trade on MT5: {e}")
        return None
