# Configuration for the `query_cacher_tfrecord.py` script
- **batch_size**: Batch size to use during processing.
- **dataset_root**: If you have a local copy of the ELI5 dataset, the path
  to that. Otherwise, if left emtpy, will default to using 
- **retriever_config_path**: See the `./retriever_configs/` configuration files. 
  Contains the paths to use for the files linked to retrieval.
- **logger_levels**: Named logger levels, allows to modulate the logging level 
  per module. `"__main__:DEBUG,utils:DEBUG,tf_utils:DEBUG,retrievers:DEBUG"` 
  is a dood default.
- **output_dir**: Where the output files should be saved.
