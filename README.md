# Chat with Websites Application

## Overview

This project is a Streamlit-based web application that allows users to interact with information from websites via a conversational chatbot. The chatbot uses **LangChain**, **FAISS**, and **Google Generative AI** to retrieve, process, and respond to user queries based on the content from specified URLs.

---

## Features

- **Website Integration:** Users can specify up to three website URLs to load and process their content.
- **Vector Search:** The content from the websites is embedded into a FAISS vector database for efficient similarity-based retrieval.
- **Context-Aware Conversations:** Supports a chat system that retains history, allowing for seamless and context-aware query responses.
- **Generative AI Responses:** Uses Google's generative AI models to generate concise and relevant answers.
- **Interactive UI:** Built with Streamlit, providing a user-friendly chat interface with configurable settings.

---

## How It Works

1. **Input Website URLs:**  
   The user specifies up to three website URLs in the sidebar. These are loaded, and their content is processed into a vector database.
   
2. **Create Vector Database:**  
   The `vectordb_creator` function uses `UnstructuredURLLoader` to extract content, splits the text into chunks, and embeds it using **Google Generative AI Embeddings**. The chunks are then stored in a FAISS vector store.

3. **Context Retrieval Chain:**  
   A context-aware retriever is created using the vector database. It reformulates user queries into standalone questions based on chat history.

4. **Conversational Question-Answering:**  
   A retrieval-augmented generation (RAG) pipeline combines the retriever's results with a generative model to provide concise, contextually accurate answers.

5. **Chat Interface:**  
   The conversation is displayed in a Streamlit chat interface, preserving the chat history for continuity.

---

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
