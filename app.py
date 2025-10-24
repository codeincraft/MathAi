import streamlit as st
import numexpr as ne
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Math & Knowledge Assistant", page_icon="🧮")
st.title("⚡ Fast AI Math & Knowledge Assistant")

# -------------------- API KEY --------------------
groq_api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")
if not groq_api_key:
    st.warning("Please enter your Groq API Key to continue.")
    st.stop()

# -------------------- MODEL --------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=512,
)

# -------------------- TOOLS --------------------
@tool
def calculator(expression: str) -> str:
    """Evaluates mathematical expressions. Input should be a valid Python math expression like '2+2' or '10*5+3'."""
    try:
        result = ne.evaluate(expression)
        return f"✅ {expression} = {result}"
    except Exception as e:
        return f"⚠️ Could not compute. Error: {str(e)}"

@tool
def wikipedia_search(query: str) -> str:
    """Searches Wikipedia for factual information. Input should be a search query."""
    try:
        wiki = WikipediaAPIWrapper()
        result = wiki.run(query)
        return result[:500]  # Limit to 500 chars
    except Exception as e:
        return f"⚠️ Wikipedia search failed: {str(e)}"

# -------------------- SIMPLE AGENT LOGIC --------------------
def process_question(question: str) -> str:
    """Process user question using simple routing logic."""
    
    # Check if it's a math question
    math_keywords = ['calculate', 'solve', 'compute', '+', '-', '*', '/', 'math', 'equation']
    if any(keyword in question.lower() for keyword in math_keywords):
        # Extract math expression
        try:
            prompt = f"Extract ONLY the mathematical expression from this question. Return just the numbers and operators, nothing else:\n{question}"
            response = llm.invoke(prompt)
            expr = response.content.strip()
            return calculator.invoke(expr)
        except:
            pass
    
    # Check if it's a factual/Wikipedia question
    wiki_keywords = ['who is', 'what is', 'when was', 'where is', 'tell me about', 'information about']
    if any(keyword in question.lower() for keyword in wiki_keywords):
        try:
            return wikipedia_search.invoke(question)
        except:
            pass
    
    # Default: Use LLM for reasoning
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer questions clearly and concisely in 2-3 sentences."),
            ("user", "{question}")
        ])
        chain = prompt | llm
        response = chain.invoke({"question": question})
        return response.content
    except Exception as e:
        return f"⚠️ Error processing question: {str(e)}"

# -------------------- CHAT --------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hi! I'm your AI assistant. I can help with:\n- Math calculations\n- Wikipedia facts\n- General questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------- INPUT --------------------
if question := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking... ⚡"):
            try:
                response = process_question(question)
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
