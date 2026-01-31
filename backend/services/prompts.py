"""
LLM prompt templates for recommendations, explanations, and cold-start.
Optimized for concise, relevant responses.
"""


# User Insight Generation
USER_INSIGHT_PROMPT = """Given a customer's shopping behavior, generate a concise 2-3 sentence insight.

Customer Profile:
- Total Spend: £{total_spend}
- Number of Purchases: {purchase_count}
- Top Categories: {top_categories}
- Average Order Value: £{avg_order_value}
- Price Sensitivity: {price_sensitivity}

Provide insight about:
1. Shopping personality/preferences
2. Price sensitivity level
3. Best recommendation strategy

Keep it concise and actionable."""


# Recommendation Explanation
RECOMMENDATION_EXPLANATION_PROMPT = """Explain in 1-2 sentences why we recommend this product to the customer.

Product: {product_name}
Category: {product_category}
Price: £{product_price}

Customer Profile:
- Top Categories: {user_categories}
- Average Price Range: £{user_avg_price}
- Price Sensitivity: {price_sensitivity}

Match Score: {match_score:.2f}

Provide a brief, compelling explanation focusing on relevance to their interests."""


# Natural Language Query Understanding
QUERY_UNDERSTANDING_PROMPT = """Extract structured information from this product search query.

Query: "{query}"

Return a JSON object with:
{{
  "category": "extracted product category or null",
  "intent": "what the user is looking for",
  "max_price": extracted maximum price as number or null,
  "min_price": extracted minimum price as number or null,
  "features": ["list", "of", "desired", "features"],
  "constraints": ["any", "constraints", "or", "requirements"]
}}

Only return the JSON, no additional text."""


# Cold Start - Initial Questions
COLD_START_QUESTIONS_PROMPT = """A new customer has no purchase history. Generate 3-4 brief questions to understand their preferences.

Focus on:
1. Product categories of interest
2. Price range/budget
3. Shopping purpose (gift, personal, business)
4. Style preferences

Keep questions natural and conversational. Return as a JSON array of strings."""


# Cold Start - Recommendation Reasoning
COLD_START_REASONING_PROMPT = """Generate product recommendations for a new customer based on their responses.

Customer Responses:
{user_responses}

Available Product Categories:
{available_categories}

Popular Products:
{popular_products}

Provide reasoning for 3-5 product recommendations that match their stated preferences.
Format as JSON array:
[
  {{
    "product_category": "category",
    "reasoning": "why this matches their preferences",
    "priority": "high/medium/low"
  }}
]"""


# Search Result Explanation
SEARCH_EXPLANATION_PROMPT = """Explain why this product matches the search query.

Query: "{query}"
Product: {product_name}
Category: {category}
Price: £{price}

Provide a 1-sentence explanation of relevance."""


# Batch Explanation (for efficiency)
BATCH_EXPLANATION_PROMPT = """Generate brief explanations for why these products are recommended to the customer.

Customer Profile:
- Top Categories: {user_categories}
- Price Sensitivity: {price_sensitivity}

Products:
{products_list}

For each product, provide a 1-sentence explanation. Return as JSON array:
[
  {{"product_id": "id", "explanation": "brief explanation"}},
  ...
]"""


def format_user_insight_prompt(user_profile: dict) -> str:
    """Format user insight prompt with profile data."""
    return USER_INSIGHT_PROMPT.format(
        total_spend=user_profile.get('total_spend', 0),
        purchase_count=user_profile.get('purchase_count', 0),
        top_categories=', '.join(user_profile.get('top_categories', [])),
        avg_order_value=user_profile.get('avg_order_value', 0),
        price_sensitivity=user_profile.get('price_sensitivity', 'unknown')
    )


def format_recommendation_explanation_prompt(product: dict, user_profile: dict, match_score: float) -> str:
    """Format recommendation explanation prompt."""
    return RECOMMENDATION_EXPLANATION_PROMPT.format(
        product_name=product.get('product_name', ''),
        product_category=product.get('category', ''),
        product_price=product.get('price', 0),
        user_categories=', '.join(user_profile.get('top_categories', [])),
        user_avg_price=user_profile.get('avg_price', 0),
        price_sensitivity=user_profile.get('price_sensitivity', 'unknown'),
        match_score=match_score
    )


def format_query_understanding_prompt(query: str) -> str:
    """Format query understanding prompt."""
    return QUERY_UNDERSTANDING_PROMPT.format(query=query)


def format_search_explanation_prompt(query: str, product: dict) -> str:
    """Format search explanation prompt."""
    return SEARCH_EXPLANATION_PROMPT.format(
        query=query,
        product_name=product.get('product_name', ''),
        category=product.get('category', ''),
        price=product.get('price', 0)
    )


def format_batch_explanation_prompt(user_profile: dict, products: list) -> str:
    """Format batch explanation prompt for multiple products."""
    products_list = "\n".join([
        f"- {p.get('product_id')}: {p.get('product_name')} (£{p.get('price')}, {p.get('category')})"
        for p in products
    ])
    
    return BATCH_EXPLANATION_PROMPT.format(
        user_categories=', '.join(user_profile.get('top_categories', [])),
        price_sensitivity=user_profile.get('price_sensitivity', 'unknown'),
        products_list=products_list
    )
