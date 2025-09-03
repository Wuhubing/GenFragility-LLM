# flake8: noqa
EXPANSION_PROMPT_TEMPLATE = """
You are a meticulous knowledge base constructor focused on creating a "pure knowledge" graph. Your task is to expand the graph from a given entity with high-quality, verifiable facts.
For the entity "{entity}", generate a list of factual triples.

You must follow these new rules strictly:
1.  **JSON Output Only**: Your entire output must be a single JSON object containing a "triples" key, which holds a list of triple objects.
2.  **Relation Whitelist**: You must only use relations from this list: {relation_whitelist}.
3.  **Entity-Centric Triples**: The "tail" of the triple MUST be a specific named entity (e.g., a person, city, organization, industry, occupation). The `tail_type` must be "entity", except for dates.
4.  **No Literals (Except Dates)**: Do NOT generate triples with tails that are URLs, coordinates, or technical IDs. Focus on knowledge, not data points.
5.  **Strictly Temporal Relations**: For relations like `HeadquarteredIn` or any role (`Role_...`), you MUST provide the relevant time fields (`start_time`, `end_time`, `as_of_date`). Failure to provide a date for a temporal relation means the triple is invalid.
6.  **Controlled Vocabularies**: For the `PrimaryIndustry` relation, the "tail" MUST be one of the following official industry names: {industry_whitelist}.
7.  **Generate a Question**: For each triple, you MUST also generate a natural language question that the triple's tail would answer.

Here is the required JSON structure for each object in the list:
{{
 "head": "{entity}",
 "relation": "<A relation from the provided whitelist>",
 "tail": "<A specific named entity (or a date for date relations)>",
 "tail_type": "<'entity' or 'literal' for dates>",
 "question": "<The natural language question>",
 "as_of_date": "<YYYY-MM-DD, optional but required for temporal relations>",
 "start_time": "<YYYY-MM-DD, optional but required for temporal relations>",
 "end_time": "<YYYY-MM-DD, optional>",
 "evidence": ["<A verifiable URL or identifier>"]
}}

Generate a single JSON object with a key "triples" containing a list of these objects.

Example of a good "pure knowledge" triple with a question:
{{
  "triples": [
    {{
      "head": "Satya Nadella",
      "relation": "EmploymentRole_atOrg",
      "tail": "Microsoft Corporation",
      "tail_type": "entity",
      "question": "At which company does Satya Nadella hold an employment role?",
      "start_time": "2014-02-04",
      "evidence": ["https://en.wikipedia.org/wiki/Satya_Nadella"]
    }},
    {{
      "head": "Microsoft Corporation",
      "relation": "PrimaryIndustry",
      "tail": "Software",
      "tail_type": "entity",
      "question": "What is the primary industry of Microsoft Corporation?",
      "evidence": ["Official company reports"]
    }}
  ]
}}

Example of a bad triple to AVOID (contains a URL as a literal tail):
{{
  "triples": [
    {{
      "head": "Apple Inc.",
      "relation": "Website",
      "tail": "https://www.apple.com",
      "question": "What is the official website for Apple Inc.?"
    }}
  ]
}}

Now, generate the JSON object for the entity "{entity}".
"""
