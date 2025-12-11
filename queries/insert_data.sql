INSERT INTO mekari_chatbot_log (
    input_prompt,
    output_llm,
    input_token,
    output_token,
    retrieved_knowledge_base,
    embedding_similarity,
    created_at
) VALUES (%s, %s, %s, %s, %s, %s, %s)