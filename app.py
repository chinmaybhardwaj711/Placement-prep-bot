import streamlit as st
import tempfile
import os


from rag_core import (
    load_and_chunk,
    build_vectorstore,
    get_chain_with_memory,
    retrieve_with_scores,
    get_pdf_hash

)

# page config
st.set_page_config(
    page_title="Placement Prep Bot",
    page_icon="🎯",
    layout="wide"
)

#Title
st.title("🎯 Placement Prep Bot")

st.caption(
    "Upload your notes → Select subject → Ask anything"
)

#sidebar
with st.sidebar:

    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload your study material",
        type="pdf",
        accept_multiple_files=True
    )

    st.divider()

    st.header("🎓 Subject Mode")

    subject = st.selectbox(
        "Select subject",
        [
            "General",
            "DSA",
            "OOPS",
            "DBMS",
            "OS",
            "CN"
        ]
    )

    st.divider()

    process_btn = st.button(
        "⚡ Process PDFs",
        use_container_width=True
    )

#session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = None

if "chain" not in st.session_state:
    st.session_state.chain = None


if process_btn:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing PDFs..."):

            temp_files = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    temp_files.append(
                        {
                            "path": tmp.name,
                            "name": uploaded_file.name
                        }
                    )

            # chunks = load_and_chunk(temp_files)

            # vectorstore = build_vectorstore(
            #     chunks,
            #     force_rebuild=True
            # )
           
            pdf_hash = get_pdf_hash(temp_files)
            chunks = load_and_chunk(temp_files)

            vectorstore = build_vectorstore(
                chunks,
                pdf_hash=pdf_hash
            )


            # vectorstore = load_cached_vectorstore(pdf_hash)

            # if vectorstore is None:
            #     chunks = load_and_chunk(temp_files)
            #     vectorstore = build_vectorstore(
            #         chunks,
            #         pdf_hash=pdf_hash
            #     )

            st.session_state.vectorstore = vectorstore

            chain, memory = get_chain_with_memory(
                vectorstore,
                subject=subject
            )

            st.session_state.chain = chain
            st.session_state.memory = memory            

            for file in temp_files:
                os.unlink(file["path"])

        st.sidebar.success(
            f"✅ {len(uploaded_files)} PDF(s) processed!"
        )


        st.divider()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
query = st.chat_input("Ask a placement question...")

if query:
    if st.session_state.chain is None:
        st.warning("⚠️ Please upload and process PDFs first.")
    else:
        # User message
        with st.chat_message("user"):
            st.write(query)

        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        # AI response
        with st.chat_message("assistant"):
           
               

                response = st.session_state.chain.invoke({
                    "question": query
                })


                answer = response["answer"]
                source_docs = response["source_documents"]

                sources = list(set([
                    doc.metadata["source"]
                    for doc in source_docs
                ]))
                st.write(answer)
                if sources:
                    st.caption(
                        f"📄 Sources: {', '.join(sources)}"
                    ) 
                    
                     
                with st.expander("🔍 View Retrieved Chunks"):

                    chunks_info = retrieve_with_scores(
                        st.session_state.vectorstore,
                        query
                    )

                    for i, chunk in enumerate(chunks_info):

                        st.markdown(f"### Chunk {i+1}")

                        st.markdown(
                            f"📄 **Source:** `{chunk['source']}`"
                        )

                        st.markdown(
                            f"📊 **Similarity Score:** `{chunk['score']}`"
                        )

                        st.text(
                            chunk["content"][:300] + "..."
                        )

                        st.divider()             

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
