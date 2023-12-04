# Databricks notebook source
pip install llama_index llama_hub pypdf


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import nest_asyncio
import openai
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

# COMMAND ----------

from agent import PdfSummaryAgent

# COMMAND ----------

model = "gpt-3.5-turbo-0613"
pdf_agent = PdfSummaryAgent(model)

# COMMAND ----------

pdf_dict = {"Macquarie" : "data/annualreport.pdf",
                   # "Henkel": "data/OTC_HENKY_2022.pdf"
                    }

# COMMAND ----------

pdf_agent.load_pdfs(pdfs=pdf_dict)

# COMMAND ----------

summaries = pdf_agent.generate_summary()

# COMMAND ----------

pdf_agent.get_chat_engine()

# COMMAND ----------


