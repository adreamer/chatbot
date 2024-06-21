| 파일명 | 설명 |
|------|-----|
| `docs/*.pdf` | 임베딩할 문서들. 전자금융업 관련 문서, 외국환거래법 관련 문서, 여신전문금융업 관련 문서들이 PDF로 있음 |
| `llm/*.py` | langchain으로 LLM 모델 만드는 라이브러리. Bedrock(Antrhopic Claude)와 OpenAI 만들어놓음 |
| `vector_store/*.py` | langchain으로 벡터DB 만드는 라이브러리. FAISS(로컬)과 Pinecone(클라우드) 만들어놓음 |
| `chatbot.py` | Streamlit 챗봇 스크립트. `streamlit run chatbot.py` 로 실행 |
| `search_text.py` | 벡터DB에서 retrieval하는 테스트 스크립트 |
| `store_to_vectordb.py` | 벡터DB에 문서 읽어서 임베딩 ingest하는 스크립트 |
