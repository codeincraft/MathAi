import streamlit as st
import numexpr as ne
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler

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
    temperature=0.2,           # reduce creativity for factual accuracy
    max_tokens=512,            # limit response length
)

# -------------------- TOOLS --------------------
def fast_math_solver(question: str) -> str:
    """Extracts and evaluates math expressions quickly."""
    try:
        expr = llm.predict(
            f"Extract only the pure math expression from this question:\n{question}"
        ).strip()
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
reasoning_chain = LLMChain(llm=llm, prompt=prompt)

reasoning_tool = Tool(
    name="Reasoning",
    func=reasoning_chain.run,
    description="Provides short reasoning and final answer.",
)

# -------------------- AGENT --------------------
assistant = initialize_agent(
    tools=[calculator, wikipedia_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=2,      # prevent multiple loops
    early_stopping_method="generate",
)

# -------------------- CHAT --------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hi, I’m your fast AI assistant! Ask me any math or knowledge question."}
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
            response = assistant.run(question, callbacks=[cb])
        except Exception as e:
            response = f"⚠️ Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
