import streamlit as st
import numexpr as ne
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler
from langchain import hub

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Math & Knowledge Assistant", page_icon="🧮")
st.title("⚡ Fast AI Math & Knowledge Assistant")

# -------------------- API KEY --------------------
groq_api_key = st.sidebar.text_input("🔑 Enter API Key", type="password")
if not groq_api_key:
    st.warning("Please enter your API Key to continue.")
    st.stop()

# -------------------- MODEL --------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=512,
)

# -------------------- TOOLS --------------------
def fast_math_solver(question: str) -> str:
    """Extracts and evaluates math expressions quickly."""
    try:
        expr = llm.invoke(
            f"Extract only the pure math expression from this question:\n{question}"
        ).content.strip()
        result = ne.evaluate(expr)
        return f"✅ {expr} = {result.item()}"
    except Exception as e:
        return f"⚠️ Could not compute. Error: {e}"

calculator = Tool(
    name="Calculator",
    func=fast_math_solver,
    description="Solves simple or compound math expressions quickly.",
)

wikipedia_tool = Tool(
    name="Wikipedia",
    func=WikipediaAPIWrapper().run,
    description="Searches Wikipedia for factual information.",
)

# reasoning tool
prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "Answer the question clearly in 2 short steps:\n"
        "1️⃣ Explain briefly (max 2 sentences)\n"
        "2️⃣ Give final answer.\n\n"
        "Question: {question}\nAnswer:"
    ),
)

reasoning_chain = prompt | llm

def reasoning_func(question: str) -> str:
    """Provides reasoning for questions."""
    try:
        response = reasoning_chain.invoke({"question": question})
        return response.content
    except Exception as e:
        return f"⚠️ Reasoning error: {e}"

reasoning_tool = Tool(
    name="Reasoning",
    func=reasoning_func,
    description="Provides short reasoning and final answer.",
)

# -------------------- AGENT --------------------
tools = [calculator, wikipedia_tool, reasoning_tool]

# Get the ReAct prompt template
react_prompt = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(llm, tools, react_prompt)

# Create the agent executor
assistant = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    max_iterations=2,
    verbose=True
)

# -------------------- CHAT --------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hi, I'm your fast AI assistant! Ask me any math or knowledge question."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------- INPUT --------------------
if question := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    with st.chat_message("assistant"), st.spinner("Thinking... ⚡"):
        try:
            cb = StreamlitCallbackHandler(st.container())
            response = assistant.invoke({"input": question}, callbacks=[cb])
            answer = response["output"]
        except Exception as e:
            answer = f"⚠️ Error: {e}"
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)
