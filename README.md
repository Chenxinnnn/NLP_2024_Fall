# NLP_2024_Fall
This is a repository for NLP course during 2024 fall semester.

Homework code includes four projects. Homework 1 is to implement functions to compute dense word vectors 
from a word co-occurrencematrix using SVD. Homework 2 is to create a RNN pipeline. Homework 3 
is to finetune a BERT-base model forsentiment analysis using the IMDB dataset. Homework 4 is to conduct 
prompt energeering for math computation and test in context learning.

Our final project centers on enhancing the Inverse Cooking model, which employs a dual-encoder to
extract ingredients and generate cooking instructions from food images. We have focused on
improving ingredient prediction accuracy and producing contextually relevant, coherent recipes.
This work sets the foundation for expanding into more advanced models like ChefFusion, which
leverages multimodal embeddings to connect vision and language, and FIRE, which integrates Vision
Transformers with language models for richer output. These future steps aim to push recipe generation
capabilities even further.

## Motivation

Recipe generation from images or text has advanced with the rise of transformer
models, but multimodal approaches that integrate both text and images to produce coherent, accurate
recipes still face significant challenges. This gap motivates us to implement, evaluate, and enhance
existing models, aiming to push the boundaries of state-of-the-art performance in this domain.

## Data
We train and evaluate our models on the Recipe1M dataset, composed of 1,029,720 recipes
scraped from cooking websites[8]. The dataset contains 720,639 training, 155 036 validation, and
154,045 test recipes, containing a title, a list of ingredients, a list of cooking instructions, and
(optionally) an image.

## Results

**Evaluation Metrics for all models**

| Model                     | SacreBLEU | ROUGE-L |
|---------------------------|----------|---------|
| Baseline: Inverse Cooking | 3.11     | 20.02   |
| FIRE                      | 3.99     | **20.74** |
| ChefFusion                | **4.88** | 20.12   |

Table: Evaluation Metrics for all models.

The table above shows the performance of different models, measured by the SacreBLEU and ROUGE-L metrics.  
The key takeaway is that the transformer-based ChefFusion model achieves the highest scores on both metrics, indicating superior performance compared to the other approaches.  
However, the baseline Inverse Cooking only shows slightly lower scores on these metrics, suggesting that it is able to establish similar levels of performance to the more advanced transformer-based ChefFusion model.



