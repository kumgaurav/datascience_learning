import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

def create_chatbot_agent(df: pd.DataFrame):
    """
    Creates a LangChain agent that can answer questions about a Pandas DataFrame.
    Uses proper LangChain configuration for Google Gemini.
    """
    try:
        # Check if Google API key is available
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise Exception("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        
        print(f"Google API key found: {api_key[:10]}...")
        
        # Initialize the Google Gemini model
        llm = ChatGoogleGenerativeAI(
            temperature=0.1,  # Slightly higher for better responses
            model="gemini-1.5-flash",
            google_api_key=api_key
        )
        
        # Test the connection with a simple query
        try:
            test_response = llm.invoke("Hello")
            print("Google Gemini API connection successful!")
        except Exception as test_error:
            print(f"API test failed: {test_error}")
            # Try with a different model
            try:
                llm = ChatGoogleGenerativeAI(
                    temperature=0.1,
                    model="gemini-pro",
                    google_api_key=api_key
                )
                test_response = llm.invoke("Hello")
                print("Google Gemini Pro API connection successful!")
            except Exception as fallback_error:
                print(f"Fallback model also failed: {fallback_error}")
                raise Exception(f"Google Gemini API not accessible. Please check your API key permissions. Error: {fallback_error}")
        
        # Use the SimpleGeminiAgent which works reliably with LangChain
        print("Creating LangChain-based stock analysis agent...")
        return SimpleGeminiAgent(llm, df)
        
    except Exception as e:
        print(f"Google Gemini API not available: {str(e)}")
        print("Falling back to simple analysis mode...")
        return SimpleAnalysisAgent(df)


class SimpleGeminiAgent:
    """
    A simple agent that directly uses Google Gemini for stock analysis.
    More reliable than the complex LangChain agent.
    """
    
    def __init__(self, llm, df):
        self.llm = llm
        self.df = df
        
        # Create a context string with the data
        self.context = f"""
        You are a stock market analyst. You have access to a dataset with {len(df)} stocks.
        
        The dataset contains these columns: {', '.join(df.columns.tolist())}
        
        Here's a summary of the top stocks:
        {df.head().to_string()}
        
        Please provide helpful, accurate analysis based on this data.
        """
    
    def invoke(self, input_dict):
        """
        Process user input and provide responses using Google Gemini.
        """
        query = input_dict.get('input', '')
        
        if not query:
            return {'output': "Please ask a question about the stock recommendations."}
        
        # Create a comprehensive prompt
        prompt = f"""
        {self.context}
        
        User Question: {query}
        
        Please provide a detailed, helpful response about the stock data. 
        Focus on the specific question asked and provide actionable insights.
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # Handle LangChain response format
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
            
            # Clean up the response
            if not response_text or response_text.strip() == "":
                response_text = "I'm sorry, I couldn't generate a response. Please try asking a different question."
            
            return {'output': response_text}
            
        except Exception as e:
            print(f"SimpleGeminiAgent error: {e}")
            return {'output': f"Sorry, I encountered an error: {str(e)}"}


class SimpleAnalysisAgent:
    """
    A simple fallback agent that provides basic analysis when the main LLM is not available.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.available_questions = [
            "What are the top stocks?",
            "Show me the best performing stocks",
            "What is the average predicted return?",
            "Which stocks have the highest confidence?",
            "Show me low risk stocks",
            "What are the technical signals?",
            "Help me understand the data"
        ]
    
    def invoke(self, input_dict):
        """
        Process user input and provide helpful responses.
        """
        query = input_dict.get('input', '').lower()
        
        if not query:
            return {'output': "Please ask a question about the stock recommendations."}
        
        # Basic analysis responses
        if any(word in query for word in ['top', 'best', 'recommend']):
            return self._get_top_stocks_analysis()
        
        elif any(word in query for word in ['average', 'mean', 'return']):
            return self._get_average_metrics()
        
        elif any(word in query for word in ['confidence', 'score']):
            return self._get_confidence_analysis()
        
        elif any(word in query for word in ['risk', 'volatility']):
            return self._get_risk_analysis()
        
        elif any(word in query for word in ['technical', 'signal', 'indicator']):
            return self._get_technical_analysis()
        
        elif any(word in query for word in ['help', 'understand', 'explain']):
            return self._get_help_info()
        
        else:
            return self._get_general_analysis()
    
    def _get_top_stocks_analysis(self):
        """Analyze top stocks by predicted return."""
        if self.df.empty:
            return {'output': "No stock data available for analysis."}
        
        top_stocks = self.df.nlargest(5, 'predicted_return_pct')
        
        response = "**Top 5 Stocks by Predicted Return:**\n\n"
        for i, (_, stock) in enumerate(top_stocks.iterrows(), 1):
            response += f"{i}. **{stock['ticker']}**: {stock['predicted_return_pct']:.2f}% return\n"
            response += f"   - Confidence: {stock['confidence_score']:.1f}/100\n"
            response += f"   - Risk Score: {stock['risk_score']:.1f}/100\n"
            response += f"   - Current Price: ${stock['close']:.2f}\n\n"
        
        return {'output': response}
    
    def _get_average_metrics(self):
        """Calculate and display average metrics."""
        if self.df.empty:
            return {'output': "No stock data available for analysis."}
        
        avg_return = self.df['predicted_return_pct'].mean()
        avg_confidence = self.df['confidence_score'].mean()
        avg_risk = self.df['risk_score'].mean()
        
        response = f"**Portfolio Summary:**\n\n"
        response += f"• **Average Predicted Return**: {avg_return:.2f}%\n"
        response += f"• **Average Confidence Score**: {avg_confidence:.1f}/100\n"
        response += f"• **Average Risk Score**: {avg_risk:.1f}/100\n"
        response += f"• **Total Stocks Analyzed**: {len(self.df)}\n"
        
        return {'output': response}
    
    def _get_confidence_analysis(self):
        """Analyze confidence scores."""
        if self.df.empty:
            return {'output': "No stock data available for analysis."}
        
        high_confidence = self.df[self.df['confidence_score'] >= 50]
        low_confidence = self.df[self.df['confidence_score'] < 30]
        
        response = "**Confidence Analysis:**\n\n"
        response += f"• **High Confidence Stocks** (≥50): {len(high_confidence)}\n"
        response += f"• **Low Confidence Stocks** (<30): {len(low_confidence)}\n"
        response += f"• **Average Confidence**: {self.df['confidence_score'].mean():.1f}/100\n\n"
        
        if not high_confidence.empty:
            response += "**Highest Confidence Stocks:**\n"
            for _, stock in high_confidence.nlargest(3, 'confidence_score').iterrows():
                response += f"• {stock['ticker']}: {stock['confidence_score']:.1f}/100\n"
        
        return {'output': response}
    
    def _get_risk_analysis(self):
        """Analyze risk metrics."""
        if self.df.empty:
            return {'output': "No stock data available for analysis."}
        
        low_risk = self.df[self.df['risk_score'] <= 40]
        high_risk = self.df[self.df['risk_score'] >= 70]
        
        response = "**Risk Analysis:**\n\n"
        response += f"• **Low Risk Stocks** (≤40): {len(low_risk)}\n"
        response += f"• **High Risk Stocks** (≥70): {len(high_risk)}\n"
        response += f"• **Average Risk Score**: {self.df['risk_score'].mean():.1f}/100\n\n"
        
        if not low_risk.empty:
            response += "**Lowest Risk Stocks:**\n"
            for _, stock in low_risk.nsmallest(3, 'risk_score').iterrows():
                response += f"• {stock['ticker']}: Risk {stock['risk_score']:.1f}/100\n"
        
        return {'output': response}
    
    def _get_technical_analysis(self):
        """Analyze technical signals."""
        if self.df.empty:
            return {'output': "No stock data available for analysis."}
        
        response = "**Technical Signals Summary:**\n\n"
        
        # Count various technical signals
        signals = {
            'Broke Resistance': 'broke_resistance',
            'Strong Momentum': 'strong_momentum',
            'Breakout Confirmed': 'breakout_confirmed',
            'Earnings in 3 Weeks': 'earnings_in_3_weeks',
            'Post-Earnings Rally': 'post_earnings_dip_rally'
        }
        
        for signal_name, signal_col in signals.items():
            if signal_col in self.df.columns:
                count = self.df[signal_col].sum()
                response += f"• **{signal_name}**: {count} stocks\n"
        
        response += f"\n**RSI Analysis:**\n"
        response += f"• Average RSI: {self.df['rsi_14d'].mean():.1f}\n"
        response += f"• Stocks with RSI > 70: {(self.df['rsi_14d'] > 70).sum()}\n"
        response += f"• Stocks with RSI < 30: {(self.df['rsi_14d'] < 30).sum()}\n"
        
        return {'output': response}
    
    def _get_help_info(self):
        """Provide help information."""
        response = "**How to Use This Stock Screener:**\n\n"
        response += "**Available Questions:**\n"
        for question in self.available_questions:
            response += f"• {question}\n"
        
        response += "\n**Understanding the Data:**\n"
        response += "• **Predicted Return**: Expected 5-day price change percentage\n"
        response += "• **Confidence Score**: 0-100 rating of prediction confidence\n"
        response += "• **Risk Score**: 0-100 rating of stock risk (lower is better)\n"
        response += "• **Technical Signals**: Various technical indicators\n"
        
        response += "\n**Note**: This is a simplified analysis mode. For full AI-powered analysis, please configure your Google Gemini API key."
        
        return {'output': response}
    
    def _get_general_analysis(self):
        """Provide general analysis when query is not recognized."""
        response = "I can help you analyze the stock recommendations! Here are some things you can ask:\n\n"
        response += "• **Top stocks** and their performance\n"
        response += "• **Average metrics** across the portfolio\n"
        response += "• **Confidence analysis** of predictions\n"
        response += "• **Risk assessment** of stocks\n"
        response += "• **Technical signals** and indicators\n"
        response += "• **Help** with understanding the data\n\n"
        response += "Try asking about specific aspects of the stock recommendations!"
        
        return {'output': response}