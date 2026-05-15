streamlit run main.py



5/3/2026 -- This has llama 3.2 prompt for diagnosis omop started working on grounded vs non grounded pipeline


What it does : 

1. LLM text notes
2. NON Grounded - Process and give ICD and conceptID mapping from LLM raw result
3. It checks ICD is real/valid using athena look up
4. It does NOT check - ICD is real but is it the clinically corretect? 
   For that we need LLM judge or symentic layer
5. It check and give concept id from Athena for that ICD



TO DO : 

1. Add database since its messing up results --- fix the benchmark
2. Get the results from database table for benchmark
2. Validate 10 simple notes with LLM
2. Correctly fix the ICD look up


2. Make Pipeline Faster
4. Then create DATA notes 
5. batch processing
6. Poster
7. Brief report
1. Append LLM ICD judge
-----------------------------------------------------------------------------------------------------