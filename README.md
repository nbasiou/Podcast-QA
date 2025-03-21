# Podcast Q&A using RAG
**Perform Q&A on a podcast audio**

This repository implements a Retrieval Augmented Generation (RAG) pipeline to support Q&A on a podcast audio:

- Setup the development environent
- Processing the audio input: transcribe the audion using an ASR conformer model.
- RAG based retrieval on the transcribed audio 

## Dependencies
Install all the necessary package requirements.

````python
pip install -r requirements.txt
````

You also need to have the podcast audio in *.wav format under a separate folder (e.g. `audio_files`). The transcribed output is stored in the sepcified `asr_output` folder. This demo uses a Conformer-CTC model, [`nvidia/stt_en_conformer_ctc_small`](https://huggingface.co/nvidia/stt_en_conformer_ctc_small) released by NVIDIA to trascribe the podcast speech. You can experiment with other ASR models. For the RAG pipeline, the demo uses the [teknium/OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) LLM model to generate the answer based on the question and the retrieved context. Finally, for the audio transcripts embeddings represenations the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) sentence embedding model is used. FAISS library is used for efficient similarity search and fast retrieval of vector embeddings.


## Notes
You need to provide the podcast audio file(s) for processing.
The models included are intended for demonstration purposes only. Feel free to experiment with alternative models that better suit your specific needs.

## References
[Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)


