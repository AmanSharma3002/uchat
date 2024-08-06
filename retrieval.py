# Updated query_retrieval function to use memory
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils import pretty_print


def query_retrieval(db, question, groq_api_key, memory):
    """
    Perform query retrieval using LangChain modules.

    Args:
    - db: Database or data structure for retrieval.
    - question (str): Query question to retrieve answers for.
    - groq_api_key (str): API key for Groq service.
    - memory: ConversationBufferMemory instance to maintain conversation history.

    Returns:
    - result (dict): Result containing the answer and source documents.
    """

    model1 = "llama-3.1-70b-versatile"
    llm = ChatGroq(temperature=0, model=model1, api_key=groq_api_key)
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description=f"The link of the page the chunk is from",
            type="String"
        )
    ]
    document_content_description = "Text from Website"
    retriever = SelfQueryRetriever.from_llm(
        llm,
        db,
        document_content_description,
        metadata_field_info,
        search_type='mmr',
        search_kwargs={"k": 5},
        verbose=True
    )

    retrieved_docs = retriever.invoke(question)
    print("Retrieved documents:")
    for doc in retrieved_docs:
        pretty_print(doc)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
        memory=memory,
    )

    try:
        result = qa_chain.invoke({"question": question})
    except Exception as e:
        print(f"Error during query retrieval: {e}")
        return None

    return result