response_template = """
**Question**: {question}<br>
**Answer**: {answer}<br>
**Sources**: Pages {sources}<br>
====================================<br>
"""

prompt_template_str = """
You are a world class state of the art agent.

Your task is to extract information from the annual report of the given company that can be used to craft a sales pitch for providing analytical and & data science services.
You provide summaries related to the topic below and enrich them by answering the listed questions.
Topic: {topic}
Questions: {questions}
"""
main_title = "# Summary for {company}<br>"
summary_title = "**#{title}**<br>"

summary_questions = {
    "Company Overview & Evolution": "What are the company's core business, industry, mission, key products or services, employee size, office locations, and notable evolution over the recent years?",
    "Financial Performance & Investment Focus": "What are the highlights of the company's financial performance, including revenue, profit/loss, main financial challenges, investment areas during the reporting period, and alignment with long-term goals?",
    "Strategies, Development, and Sustainability": "What key strategies, product development plans, and sustainability initiatives are in place for the company's growth, market adaptation, and technological advancement?",
    "Data-Driven Initiatives & Business Priorities": "Can you describe the company's data-driven initiatives, the role of data in decision-making, key business priorities for the upcoming period, and how these align with overall goals?",
    "Process Optimization & Competitive Landscape": "What efforts are made towards process optimization, specific improvements, impact on performance, and the company's strategies to differentiate from competitors and respond to market trends?",
}
