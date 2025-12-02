import streamlit as st
import openai
import re
import time
import csv
import os
from datetime import datetime

# Import các hàm cần thiết từ module retrieval của chúng ta
import retrieval 

# --- CONFIGURATION ---
st.set_page_config(page_title="PC Assistant Chatbot", layout="wide")

# --- UTILITY FUNCTIONS (For UI and OpenAI) ---

def parse_response(response: str):
    """Tách phần suy nghĩ của LLM ra khỏi câu trả lời cuối cùng."""
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, response, re.DOTALL)
    if match:
        thought_content = match.group(1).strip()
        clean_response = re.sub(think_pattern, '', response, count=1, flags=re.DOTALL).strip()
        return thought_content, clean_response
    return None, response.strip()

def get_openai_client(api_key: str, base_url: str):
    """Khởi tạo OpenAI client."""
    if not api_key or not base_url: return None
    try:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

def rewrite_query_with_llm(client: openai.OpenAI, query: str, model: str):
    """Sử dụng LLM để biến câu hỏi của người dùng thành truy vấn tìm kiếm tối ưu."""
    if not client:
        return query # Trả về truy vấn gốc nếu client không hợp lệ

    system_prompt = (
        "You are an expert AI assistant that rewrites user queries for a vector database search. "
        "The database contains ONLY **pre-built / assembled PCs** (ready-to-use systems). "
        "There are NO individual components in the database (no CPU-only, GPU-only, RAM-only, etc.).\n\n"

        "Your task is to rewrite the user's natural-language query into a concise, keyword-rich search query "
        "that best captures the user's intent **for a full PC system**.\n\n"

        "Guidelines:\n"
        "- Always focus on **pre-built PCs** ('PC nguyên bộ', 'PC lắp sẵn', 'ready-to-use PC', 'prebuilt PC').\n"
        "- NEVER include or search for individual components (CPU, GPU, RAM, SSD, PSU, mainboard, case…).\n"
        "- If the user mentions a budget, include it in the rewritten query (e.g., 'PC gaming 20 triệu').\n"
        "- Expand vague requests into clearer intent (e.g., 'máy mạnh' → 'PC gaming mạnh', "
        "'máy để render' → 'workstation PC render').\n"
        "- Always include the usage scenario if present: gaming, office, editing, rendering, study, etc.\n"
        "- Include both Vietnamese and English keywords for better retrieval.\n"
        "- All currency is VND (Vietnamese Dong).\n"
        "- Do NOT answer or explain anything — only output the rewritten search query."
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite the following query: '{query}'"}
            ],
            temperature=0.0
        )
        rewritten_query = response.choices[0].message.content.strip()
        return rewritten_query.replace('"', '')
    except openai.APIError:
        return query

def generate_response_stream(client: openai.OpenAI, messages: list, retrieved_info: str, model: str):
    """Yields response chunks from the LLM API stream."""
    if not client:
        yield "Error: OpenAI client not initialized."
        return

    try:
        # <<< THAY ĐỔI 1: SỬA LỖI F-STRING >>>
        # Đã xóa cặp ngoặc nhọn thừa xung quanh `retrieved_info`.
        # TRƯỚC ĐÂY: f"Retrieved Information:\n{{retrieved_info}}"
        # BÂY GIỜ:   f"Retrieved Information:\n{retrieved_info}"
        system_prompt = (
            "You are a helpful Vietnamese assistant for a PC parts store. "
            "Use the retrieved information below to answer the user's latest question. "
            "Be concise, natural, and friendly.\n\n"
            f"Retrieved Information:\n{retrieved_info}"
        )
        final_messages = [{"role": "system", "content": system_prompt}] + messages

        stream = client.chat.completions.create(
            model=model,
            messages=final_messages,
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except openai.APIError as e:
        yield f"Error generating response: {e}"

# --- MODEL LOADING (with Streamlit Caching) ---

@st.cache_resource(show_spinner="Setting up device...")
def cached_setup_device():
    return retrieval.setup_device()

@st.cache_resource(show_spinner="Loading embedding model (Qwen-Embedding)...")
def cached_load_embedding_model():
    return retrieval.load_embedding_model()

@st.cache_resource(show_spinner="Loading reranker model (Qwen-Reranker)...")
def cached_load_reranker_data(device):
    return retrieval.load_reranker_data(device)

# --- STREAMLIT UI ---
st.title("PC Assistant Chatbot 💬")

device = cached_setup_device()
embedding_model = cached_load_embedding_model()
reranker_data = cached_load_reranker_data(device)

with st.sidebar:
    st.header("⚙️ LLM Configuration")
    st.info("Configure the API endpoint for your **generation** LLM below.")
    
    generation_api = st.text_input("API Endpoint Base URL", value="http://127.0.0.1:1234/v1")
    api_key = st.text_input("API Key", type="password", value="not-needed")
    client = get_openai_client(api_key, generation_api)
    selected_model = st.text_input("Select Generation Model", value="qwen/qwen3-4b-thinking-2507")

if "messages" not in st.session_state:
    st.session_state.messages = []

# <<< THAY ĐỔI 2: CẬP NHẬT GIAO DIỆN LỊCH SỬ TRÒ CHUYỆN >>>
# Sử dụng st.expander để hiển thị khối <think> trong lịch sử.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "thoughts" in message and message["thoughts"]:
            with st.expander("Show Thought Process"):
                st.markdown(message["thoughts"])
        st.markdown(message["content"])

if prompt := st.chat_input("Hỏi về linh kiện hoặc PC dựng sẵn..."):
    start_time = time.time()  # Start measuring total latency
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Phân tích và tối ưu hóa câu hỏi..."):
            rewritten_query = rewrite_query_with_llm(client, prompt, selected_model)
            with st.expander("Show Optimized Query for Retrieval"):
                st.info(f"**Original:** {prompt}\n\n**Optimized:** {rewritten_query}")

        with st.spinner("Đang tìm kiếm và xếp hạng thông tin..."):
            retrieved_info = retrieval.perform_retrieval_and_reranking(rewritten_query, embedding_model, reranker_data)
            with st.expander("Show Reranked Context"):
                st.info(retrieved_info or "No context found.")
        
        if not client:
            st.warning("Cannot generate response. LLM API client not configured.")
        else:
            # <<< THAY ĐỔI 3: CẬP NHẬT LUỒNG STREAMING VỚI DROPDOWN >>>
            # Sử dụng placeholder cho câu trả lời và tạo expander khi có nội dung suy nghĩ.
            answer_placeholder = st.empty()
            
            full_raw_response = ""
            thought_content = ""
            clean_response = ""
            
            is_thinking_parsed = False

            response_stream = generate_response_stream(client, st.session_state.messages, retrieved_info, model=selected_model)

            # Tạo expander trước, nhưng chưa điền nội dung.
            thought_expander = st.expander("Show Thought Process")
            
            for chunk in response_stream:
                full_raw_response += chunk
                
                if "</think>" in full_raw_response and not is_thinking_parsed:
                    temp_thought, temp_clean = parse_response(full_raw_response)
                    if temp_thought:
                        thought_content = temp_thought
                        clean_response = temp_clean
                        
                        # Điền nội dung vào expander đã tạo
                        thought_expander.markdown(thought_content)
                        
                        answer_placeholder.markdown(clean_response)
                        is_thinking_parsed = True
                elif is_thinking_parsed:
                    clean_response += chunk
                    answer_placeholder.markdown(clean_response)
                else:
                    answer_placeholder.markdown(full_raw_response)
            
            _, final_clean_response = parse_response(full_raw_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_clean_response,
                "thoughts": thought_content
            })
    
    # Calculate and display total latency
    end_time = time.time()
    total_latency = end_time - start_time
    st.info(f"Thời gian xử lý: {total_latency:.2f} giây")
    
    # Feedback form
    with st.form(key=f"feedback_{len(st.session_state.messages)}"):
        st.subheader("Phản hồi về câu trả lời")
        accuracy = st.radio(
            "Độ chính xác của cấu hình tư vấn:",
            ["correct_or_acceptable", "incorrect_or_not_suitable"],
            index=0,
            help="Chọn 'correct_or_acceptable' nếu cấu hình đúng/chấp nhận được, ngược lại chọn 'incorrect_or_not_suitable'."
        )
        context_score = st.slider(
            "Điểm hiểu ngữ cảnh (1-5):",
            min_value=1,
            max_value=5,
            value=3,
            help="1: hiểu sai hoàn toàn, 5: hiểu đúng và phản hồi hợp lý"
        )
        personalization_score = st.slider(
            "Điểm tính cá nhân hóa (1-5):",
            min_value=1,
            max_value=5,
            value=3,
            help="Mức độ phù hợp với sở thích, ngân sách, ràng buộc cá nhân"
        )
        submitted = st.form_submit_button("Gửi phản hồi")
        
        if submitted:
            # Save to CSV
            csv_file = "./feedback.csv"
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["timestamp", "question", "answer", "accuracy", "context_score", "personalization_score", "total_latency"])
                writer.writerow([
                    datetime.now().isoformat(),
                    prompt,
                    final_clean_response,
                    accuracy,
                    context_score,
                    personalization_score,
                    round(total_latency, 2)
                ])
            st.success("Cảm ơn phản hồi của bạn!")