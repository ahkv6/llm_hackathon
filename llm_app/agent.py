from llama_index import VectorStoreIndex, ServiceContext
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings import OpenAIEmbedding
from llama_hub.file.pdf.base import PDFReader
from llama_index import StorageContext
from llama_index import load_index_from_storage
from llama_index.vector_stores import SimpleVectorStore
from llama_index.storage.index_store import SimpleIndexStore
from IPython.display import Markdown
from templates import summary_questions, main_title
from utils import (
    INDEX,
    QUERY_ENGINE,
    generate_responses,
    get_query_engine,
    get_chat_engine_tools,
)

GPT4 = "gpt-4-1106-preview"

from llama_index.callbacks import CallbackManager, WandbCallbackHandler


class PdfSummaryAgent:
    def __init__(self, openai_model: str = GPT4, print_trace: bool = False):
        """
        Initializes the PDF Summary Agent with a specified language model.

        Args:
            openai_model (str): Identifier for the language model to be used.
             Defaults to 'gpt-4-1106-preview'.
        """
        self._init_llama(openai_model=openai_model, print_trace=print_trace)

    def _init_llama(self, openai_model: str = GPT4, print_trace: bool = False):
        """
        Initializes Llama index with the specified model.

        Args:
            openai_model (str): The OpenAI model to be used. Defaults to GPT4.
        """
        self.embed_model = OpenAIEmbedding(embed_batch_size=10)
        self.llm = OpenAI(model=openai_model, temperature=0.1)
        wandb_args = {"project": "llama-index-report"}
        self.llama_debug = WandbCallbackHandler(run_args=wandb_args)
        # self.llama_debug = LlamaDebugHandler(print_trace_on_end=print_trace)
        self.callback_manager = CallbackManager([self.llama_debug])
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            callback_manager=self.callback_manager,
            embed_model=self.embed_model,
        )

    def load_pdfs(self, pdfs: dict[str, str], load_saved: bool = False):
        """
        Loads PDFs and prepares their indexes for query processing.

        Args:
            pdfs (dict[str, str]): A dictionary mapping company names to PDF file paths.
        """
        self.index_dict = {}
        for company_name, pdf in pdfs.items():
            self.index_dict[company_name] = {
                INDEX: self.load_store(location=pdf, load_saved=load_saved)
            }
            self.index_dict[company_name][QUERY_ENGINE] = get_query_engine(
                self.index_dict[company_name][INDEX], verbose=True
            )

    def load_store(self, location: str = "", load_saved: bool = False):
        if load_saved:
            storage_context = StorageContext.from_defaults(
                vector_store=SimpleVectorStore.from_persist_dir(
                    persist_dir=location, namespace="default"
                ),
                index_store=SimpleIndexStore.from_persist_dir(persist_dir=location),
            )
            index = load_index_from_storage(storage_context)
        else:
            index = self._get_vector_store(location)
        return index

    def _get_vector_store(self, pdf_location: str) -> VectorStoreIndex:
        """
        Converts a PDF file into a vector store for indexing.
        Applies SentenceWindow processing to the document.

        Args:
            pdf_location (str): The file path of the PDF document.

        Returns:
            VectorStoreIndex: An index created from the PDF document for querying.
        """
        pdf_reader = PDFReader()
        doc = pdf_reader.load_data(pdf_location)

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        nodes = node_parser.get_nodes_from_documents(doc)
        index = VectorStoreIndex(nodes, service_context=self.service_context)

        return index

    def generate_summary(self, topic_questions: dict = summary_questions):
        """
        Generates a summary for each company in the index dictionary.

        Args:
            topic_questions (dict[str, str]): A dictionary of questions to guide the summary generation. Defaults to concise_guiding_questions.

        Returns:
            str: A concatenated string of responses forming the summary.
        """
        summary_title = {}
        for company, index_items in self.index_dict.items():
            responses = generate_responses(
                engine=index_items[QUERY_ENGINE],
                topic_questions=topic_questions,
                llama_debug=self.callback_manager,
            )
            full_response = main_title.format(company=company)
            for _, response in responses.items():
                full_response += response["response"]
            summary_title[company] = full_response
            display(Markdown(full_response))
        return summary_title

    def get_chat_engine(self, model: str = GPT4, verbose: bool = False):
        """
        Creates a chat engine for interactive queries.

        Args:
            verbose (bool): If True, enables verbose logging.

        Usage:
            This function enters an interactive chat mode which can be exited by typing 'exit'.
        """
        chat_llm = OpenAI(
            temperature=0.1,
            model=model,
            streaming=True,
        )

        chat_engine = OpenAIAgent.from_tools(
            tools=get_chat_engine_tools(
                index_info=self.index_dict,
                service_context=self.service_context,
                verbose=verbose,
            ),
            llm=chat_llm,
            verbose=verbose,
            callback_manager=self.callback_manager,
            max_function_calls=3,
        )

        while True:
            text_input = input("User: ")
            if text_input == "exit":
                break
            response = chat_engine.chat(text_input)
            print(f"Agent: {response}")
