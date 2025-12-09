'''
FAST API ENDPOINT 
'''

# import logging
# import uvicorn
# from MedAge import rag as rag_module
# from MedAge.rag import build_summary_chain, load_pdf_text_elements, build_vectorstore_from_texts
# from MedAge.config import GROQ_API_KEY

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def initialize_rag():
#     pdf_path = "lung_disease.pdf"
#     try:
#         texts = load_pdf_text_elements(pdf_path)
#         if texts:
#             vectorstore = build_vectorstore_from_texts(texts)
#             summary_chain = build_summary_chain(groq_api_key=GROQ_API_KEY)
#             rag_module.vectorstore = vectorstore
#             rag_module.summary_chain = summary_chain
#             logger.info("✅ RAG initialized and attached to rag module.")
#         else:
#             logger.warning("No text elements found in PDF; skipping RAG initialization.")
#     except Exception as e:
#         logger.exception("RAG initialization failed (this is non-fatal)")

# if __name__ == "__main__":
#     initialize_rag()
#     # Run via module string to avoid import-time problems
#     uvicorn.run("MedAge.api:app", host="0.0.0.0", port=8000, reload=True)
