# Fake News Classification Using the Fakeddit Dataset

## Introduction
Fakeddit is a fine-grained multimodal fake news detection dataset, designed to advance efforts to combat the spread of misinformation in multiple modalities.
I worked on classifying the data in Fakeddit into 6 pre-defined classes: 
- Authentic/true news content
- Satire/Parody
- Content with false connection
- Imposter content
- Manipulated content
- Misleading content
  
For the Image-Feature Extractor, I used a pre-trained ```ResNet50 model``` trained on the ImageNet dataset for image classification tasks.  
For the Text-Feature Extractor, I used a pre-trained ```Bertmodel``` trained on the English Wikipedia and Toronto Book Corpus in lower cased letters.

- Base Reference Paper: [r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection](https://aclanthology.org/2020.lrec-1.755/)
 
## Problem Statement
Fake news has altered society in negative ways in politics and culture. It has adversely affected both online social network systems as well as offline communities and conversations. Using automatic machine learning classification models is an efficient way to combat the widespread dissemination of fake news. However, a lack of effective, comprehensive datasets has been a problem for fake news research and detection model development. Prior fake news datasets do not provide multimodal text and image data, metadata, comment data, and fine-grained fake news categorization at the scale and breadth of our dataset. Fakeddit is a novel multimodal dataset consisting of over 1 million samples from multiple categories of fake news. The goal is to construct hybrid text+image models and perform extensive experiments for multiple variations of classification, demonstrating the importance of the novel aspect of multimodality and fine-grained classification unique to Fakeddit.
 
## Model Architecture


#### BERT and BERT embeddings
- BERT uses a bi-directional approach considering both the left and right context of words in a sentence, instead of analyzing the text sequentially.
- These vectors are used as high-quality feature inputs to downstream models. NLP models such as LSTMs or CNNs require inputs in the form of numerical vectors, hence BERT is a good option for encoding variable length text strings.

#### ResNet50
- ResNet50 is a deep learning model launched in 2015 by Microsoft Research for the purpose of visual recognition. The model is 50 layers deep.
- ResNet50's architecture (including shortcut connections between layers) significantly improves on the vanishing gradient problems that arise during backpropagation which allows for higher accuracy.
- The skip connections in ResNet50 facilitate smoother training and faster convergence. Thus making it easier for the model to learn and update weights during training.

#### Late Fusion
-  Late fusion processes the data of each sensor independently to make a local prediction. These individual results are then combined at a higher level to make the final fused prediction.
- The advantage of late fusion is its simplicity and isolation. Each model gets to learn super rich information on its modality.


## Running the Streamlit App
From the repository root, install the app dependencies and start Streamlit:

```powershell
cd Fakeddit-WebApp
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first prediction may download the BERT tokenizer files. The included model
checkpoint is loaded locally.

### Optional OpenAI second opinion

Install dependencies as above, then set your OpenAI API key in the environment
before launching the app. In PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python -m streamlit run app.py
```

The app only sends the headline and a resized copy of the uploaded image to
OpenAI when **Get an OpenAI second opinion** is selected. The local model always
runs independently. The second opinion is an additional assessment, not a
verified fact-check or a guaranteed accuracy improvement. Do not commit API keys
to the repository.
---

The multi-modal feature extraction by Text-Image-Model yielded an overall performance of *77.46%* accuracy on test results.  
(Originally completed for Web Enthusiasts Club Recruitments 2024)
