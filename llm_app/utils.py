from llama_index import (
    StorageContext,
    ServiceContext,
    VectorStoreIndex,
)
from llama_index.tools import ToolMetadata, QueryEngineTool
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.prompts import PromptTemplate
from llama_index.response import Response
from llama_index.callbacks import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.postprocessor import MetadataReplacementPostProcessor

from templates import prompt_template_str, summary_title, response_template

INDEX = "index"
QUERY_ENGINE = "query_engine"


def get_query_engine(index, verbose: bool = False):
    """Create and return a query engine based on the given index."""
    return index.as_query_engine(
        similarity_top_k=2,
        verbose=verbose,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )


def get_chat_engine_tools(index_info: dict, service_context, verbose: bool = False):
    """Create and return a list of top-level query engine tools based on the provided index info."""
    # Create individual query engine tools for each company
    query_engine_tools = [
        QueryEngineTool(
            query_engine=get_query_engine(index[INDEX], verbose=verbose),
            metadata=ToolMetadata(
                name=f"{company_name} Annual Report Summarizer",
                description=f"Useful for answering questions related to {company_name}.",
            ),
        )
        for company_name, index in index_info.items()
    ]

    # Create a subquestion engine
    subquestion_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        service_context=service_context,
        verbose=True,
    )

    # Wrap the subquestion engine in a tool
    subquestion_wrapper = QueryEngineTool.from_defaults(
        query_engine=subquestion_engine,
        description="Useful for answering complex questions involving multiple steps of reasoning.",
    )

    # Create top-level tools for each company and include the subquestion wrapper
    top_level_tools = [
        QueryEngineTool.from_defaults(
            query_engine=get_query_engine(index[INDEX], verbose=verbose),
            description=f"Useful for answering questions related to {company_name}'s annual report.",
        )
        for company_name, index in index_info.items()
    ]
    top_level_tools.append(subquestion_wrapper)

    return top_level_tools


def get_page_metadata(response: Response):
    """
    Extracts page metadata from a given response
    """
    return [
        metadata.get("page_label")
        for _, metadata in response.metadata.items()
        if "page_label" in metadata
    ]


join_page_labels = lambda page_label: ", ".join(map(str, page_label))


def get_response_callback(
    llama_debug: CallbackManager, response_template: str = response_template
):
    responses = []
    for i, (start_event, end_event) in enumerate(
        llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
    ):
        qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
        page_labels = [source.metadata["page_label"] for source in qa_pair.sources]
        response = response_template.format(
            question=qa_pair.sub_q.sub_question.strip(),
            answer=qa_pair.answer.strip(),
            sources=join_page_labels(page_labels),
        )
        responses.append(response)
    llama_debug.flush_event_logs()

    return "\n".join(responses)


def generate_responses(
    engine,
    llama_debug: CallbackManager,
    topic_questions: dict,
    template: str = prompt_template_str,
):
    """
    Generates responses on questions in the topic_questions dict over the provided index,
    extracts page metadata and stores the question,response and metadata in a dict.
    """
    responses = {}

    for prompt_topic, questions in topic_questions.items():
        prompt_template = PromptTemplate(template=template)
        fmt_prompt = prompt_template.format(topic=prompt_topic, questions=questions)

        response = engine.query(fmt_prompt)

        responses[prompt_topic] = {}

        title = summary_title.format(title=prompt_topic)
        main_response = response_template.format(
            question=questions,
            answer=response.response.strip(),
            sources=join_page_labels(get_page_metadata(response)),
        )
        responses[prompt_topic]["response"] = (
            title + main_response  # + get_response_callback(llama_debug)
        )

    return responses
