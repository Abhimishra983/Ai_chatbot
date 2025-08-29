# *********** Phase-1 (Create AI Agent) *********************

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Step-1 --> Setup API Keys for Groq, OpenAI, and Tavily
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Debug: print API keys to ensure they are loaded (remove later!)
print("OPENAI_API_KEY:", "Loaded" if OPENAI_API_KEY else "Missing")
print("GROQ_API_KEY:", "Loaded" if GROQ_API_KEY else "Missing")
print("TAVILY_API_KEY:", "Loaded" if TAVILY_API_KEY else "Missing")

# Step-2 --> Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

try:
    openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
except Exception as e:
    print("Error initializing LLMs:", e)

# Optional: comment out search tool for now to isolate errors
try:
    search_tool = TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY)
except Exception as e:
    print("Error initializing search tool:", e)
    search_tool = None

# Step-3 --> Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt = "Act as an AI chatbot who is smart and friendly"

# response function
def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    try:
        if provider == "Groq":
            llm = ChatGroq(model=llm_id, api_key=GROQ_API_KEY)
        elif provider == "OpenAI":
            llm = ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)
        else:
            raise ValueError("Invalid provider")

        tools = [TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY)] if allow_search else []

        # creating AI agent
        agent = create_react_agent(
            model=llm,
            tools=tools
        )

        # Add system prompt as the first message
        state = {"messages": [system_prompt] + query}

        response = agent.invoke(state)
        messages = response.get("messages")
        ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
        return ai_messages[-1] if ai_messages else "No response from agent"

    except Exception as e:
        print("Error in get_response_from_ai_agent:", e)
        return f"Error: {e}"

