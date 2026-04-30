import json 
import asyncio
import openai
from openai import AsyncOpenAI
from data_provider import fetch_news_yfinance

def create_sentiment_prompt(ticker: str, headlines: list) -> str:
    """
    Creates a strict prompt to force the LLM to output a JSON object.
    """
    headlines_text = "\n".join([f"- {h}" for h in headlines])
    
    prompt = f"""
    You are an expert financial quantitative analyst. 
    Analyze the overall sentiment of the following recent news headlines for the asset '{ticker}'.
    
    Headlines:
    {headlines_text}
    
    Respond STRICTLY with a valid JSON object. Do not include markdown formatting (like ```json), explanations, or any other text.
    The JSON must have exactly two keys:
    - "sentiment": A float between -1.0 (extreme negative/bearish) and 1.0 (extreme positive/bullish).
    - "confidence": A float between 0.0 and 1.0 indicating your confidence in this assessment.
    """
    return prompt

async def run_sentiment_analysis(client: AsyncOpenAI, tickers: list) -> dict:
    """
    Asynchronously aggregates news and fetches sentiment from LLM for all tickers at once.
    """
    async def process_single_ticker(ticker):
        headlines = fetch_news_yfinance(ticker) 
        sentiment_data = await get_llm_sentiment(client, ticker, headlines)
        return ticker, sentiment_data.get('sentiment', 0.0)

    tasks = [process_single_ticker(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    return dict(results)

async def get_llm_sentiment(client: AsyncOpenAI, ticker: str, headlines: list) -> dict:
    """
    Sends the prompt to OpenAI API and forces a structured JSON response.
    """
    if not headlines:
        return {"sentiment": 0.0, "confidence": 0.0}
    
    prompt = create_sentiment_prompt(ticker, headlines)
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial analyst algorithm. You strictly output JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }, 
            temperature=0.0 
        )
        result_str = response.choices[0].message.content
        result_json = json.loads(result_str)
        
        if "sentiment" not in result_json:
            raise KeyError(f"LLM hallucinated keys. Response: {result_json}")
        
        return result_json
        
    except openai.APIError as e:
        print(f"OpenAI Network/API Error for {ticker}: {e}")
        return {"sentiment": 0.0, "confidence": 0.0}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Data format error for {ticker}: {e}")
        return {"sentiment": 0.0, "confidence": 0.0}
