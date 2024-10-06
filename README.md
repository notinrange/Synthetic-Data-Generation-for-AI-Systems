
# Synthetic Data generation from pandas dataframe with LLM

### Model used : [nemotron-4-340b-instruct](https://build.nvidia.com/nvidia/nemotron-4-340b-instruct)

### Synthetic data generation : unsuccessful

### library used : langchain, openai

### API used : nvidia opensource API because openai and Vertex AI are paid

I designed whole architecture of input prompt template data preprocessing and cleaning also but after using synthetic_data_generator of langchain i got suspicious Inference error Which I am trying now... to get Synthetic Data


#### Step 1 Amazon product review dataset cleaning and Converting into dictionary
```
df = df.head(20)
# Converting the DataFrame to a list of dictionaries
examples = df.to_dict(orient='records')

examples[0]
{'title_x': 'Bariatric Fusion Bariatric Multivitamin Soft Chew | Tropical Fruit Flavor | Chewy for Post Bariatric Surgery Patients Including Gastric Bypass and Sleeve Gastrectomy | 60 Count | 1 Month Supply',
 'parent_asin': 'B0BKC2WYWB',
 'categories': 'Health & Household',
 'cat1': ' Heart Health Event',
 'cat2': nan,
 'cat3': nan,
 'cat4': nan,
 'cat5': nan,
 'rating': 5,
 'title_y': 'Good vitamins',
 'text': 'Is a bariatric patient, my vitamins are really important!! These Taste pretty good and seems to work.',
 'asin': 'B07JH63HWS',
 'user_id': 'AF2HY3SCRK45T2ATV7FKRYUJCCTA',
 'helpful_vote': 0,
 'verified_purchase': True,
 'date': Timestamp('2019-12-12 00:00:00'),
 'time': datetime.time(12, 43)}
```


#### Step 2 Defining Prompt template

```
template_str = """
Product Review:
- Product Title: {title_x}
- Parent ASIN: {parent_asin}
- Categories: {categories}
- Sub-Category 1: {cat1}
- Sub-Category 2: {cat2}
- Sub-Category 3: {cat3}
- Sub-Category 4: {cat4}
- Sub-Category 5: {cat5}
- Rating: {rating}/5
- Review Title: {title_y}
- Review Text: "{text}"
- Product ASIN: {asin}
- User ID: {user_id}
- Helpful Votes: {helpful_vote}
- Verified Purchase: {verified_purchase}
- Review Date: {date}
- Review Time: {time}
"""

# Template for the synthetic data example
SYNTHETIC_DATA_EXAMPLE = PromptTemplate(
    input_variables=["title_x", "parent_asin", "categories", "cat1", "cat2", "cat3", "cat4", "cat5", "rating", "title_y", "text", "asin", "user_id", "helpful_vote", "verified_purchase", "date", "time"],
    template=template_str
)

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples, 
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject","extra"],
    example_prompt=SYNTHETIC_DATA_EXAMPLE,
)
```

#### Step 4 Defining synthetic_data_generator
```
synthetic_data_generator = create_openai_data_generator(
    output_schema=AmazonProductReviews,
    llm=ChatNVIDIA(model="nvidia/nemotron-4-340b-instruct",temperature=0.2),
    prompt=prompt_template
)
```


#### Step 5 Trying to generate result
```
synthetic_results = synthetic_data_generator.generate(
    subject="amazon_product_reviews",
    extra = "the name must be chosen at random. Make it something you wouldn't normally choose.",
    runs=10,
)
```
## Why this Model ?
- Nvidia's nemotron-4-340b-instruct LLM model is specially designed for synthetic data generation.It is a fine-tuned version of the Nemotron-4-340B-Base model, optimized for English-based single and multi-turn chat use-cases. It supports a context length of 4,096 tokens.

- The base model was pre-trained on a corpus of 9 trillion tokens consisting of a diverse assortment of English based texts, 50+ natural languages, and 40+ coding languages. 

## Efficiency measures and testing 
- We can measure efficiency of dataset by testing synthetic data as testing data against input data as train and validation data and check confusion Matrix for that.

## Challenges

- Lack of proper documentations
- almost all LLM services are paid
- Synthetic Data generation as csv is need to be developed 


## References

[nemotron-4-340b-instruct LLM Model](https://build.nvidia.com/nvidia/nemotron-4-340b-instruct)

[Synthetic Data generation YouTube](https://www.youtube.com/watch?v=hMjtdECXlYo)

[Langchain Documentation on ChatNVIDIA](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/)

[Amazon Product Reviews Dataset](https://drive.google.com/file/d/19eTFRj2ctWYOmdYHuC7h7qlBBDYqSVVM/view)


