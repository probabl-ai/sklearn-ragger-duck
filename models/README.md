# Models folder

This folder contains the retriever index resulting from the command
`pixi run train-retrievers`. When training the model, the pre-trained embeddings
are downloaded here as well.

In addition, this is the location where the LLM model can be stored. For instance,
the command `pixi run fetch-mistral` will download the Mistral 7b model and store it
in this folder.
