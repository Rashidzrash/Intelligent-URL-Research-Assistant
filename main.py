import streamlit as st
from RAG import processed,generate_answers



st.title('Real Estate Research Tool')

if "urls" not in st.session_state:
    st.session_state.urls = 1
urls = []
for i in range(st.session_state.urls):
    url=st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    urls.append(url)

if st.sidebar.button("Add URL"):
    st.session_state.urls += 1

process_url_button=st.sidebar.button('Process URL')
placeholder=st.empty()
if process_url_button:
    valid_urls = [url for url in urls if url.strip() != ""]
    if not valid_urls:
        placeholder.text("URL is empty")
    else:
        for status in processed(valid_urls):
            placeholder.text(status)
query=placeholder.text_input("Enter Your Question")
if query:
    try:
        answer,sources=generate_answers(query)
        st.subheader('Answer:')
        st.write(answer)
        if sources:
            st.subheader('Sources:')
            for source in sources:
                st.write(source)
    except RuntimeError as e:
        placeholder.text("You must process urls first")