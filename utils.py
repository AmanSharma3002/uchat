def pretty_print(docs):
    if type(docs)==list:
        print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
    elif type(docs)==dict:
        print(docs["answer"])
    else:
        print(docs)