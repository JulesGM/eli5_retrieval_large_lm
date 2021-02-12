# Retriever JSON Configuration Files

- **query_embedder_path**: "gs://realm-data/cc_news_pretrained/embedder"
- **text_records**: A tfrecord containing a single block with the textual form 
  of the corpus to be retrieved. Should use the same format as the ORQA file.
  See `gs://orqa-data/enwiki-20181220/blocks.tfr`. 
  See 
  https://github.com/google-research/language/tree/master/language/orqa#modeling 
  for more details.
- **num_block_records**: Quantity of entries in the `blocks.tfr` file.
- **description**: A short description of the purpose of the specific config 
  file.